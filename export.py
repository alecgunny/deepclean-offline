import logging
import os
import shutil
import sys
import typing

import torch
from google.api_core import exceptions
from google.cloud import container_v1 as container
from google.cloud import storage

import cloud_utils
import exportlib
import typeo
from deepclean_prod.nn.net import DeepClean


class StreamingDeepClean(torch.nn.Module):
    def __init__(self, dc, update_size):
        super().__init__()
        self.dc = dc
        self.update_size = update_size

    def forward(self, x):
        x = self.dc(x)
        return x[:, -self.update_size:]


def main(
    service_account_key_file: str,
    project: str,
    zone: str,
    cluster_name: str,
    model_repo_bucket_name: str,
    kernel_size: float,
    fs: float,
    kernel_stride: float,
    channels: str,
    weights_path: typing.Optional[str] = None,
    inference_gpu: str = "t4",
    use_fp16: bool = False
):
    manager = cloud_utils.GKEClusterManager(
        project=project, zone=zone, credentials=service_account_key_file
    )
    try:
        cluster = manager.resources[cluster_name]
    except KeyError:
        cluster_resource = container.Cluster(
            name=cluster_name,
            node_pools=[container.NodePool(
                name="default-pool",
                initial_node_count=2,
                config=container.NodeConfig()
            )]
        )
        cluster = manager.create_resource(cluster_resource)
        cluster.deploy_gpu_drivers()

    repo_dir = "tmprepo"
    if not os.path.exists(repo_dir):
        os.makedirs(repo_dir)
    repo = exportlib.ModelRepository(repo_dir)
    model = repo.create_model("deepclean", platform="tensorrt_plan")

    with open(channels, "r") as f:
        channels = [i for i in f.read().splitlines() if i][1:]

    deepclean = DeepClean(len(channels))
    if weights_path is not None:
        state = torch.load(weights_path, map_location=torch.device("cpu"))
        deepclean.load_state_dict(state)
    deepclean.eval()
    deepclean = StreamingDeepClean(deepclean, int(kernel_stride * fs))

    node_pool_config = cloud_utils.gke.create_gpu_node_pool_config(
        vcpus=4, gpus=1, gpu_type=inference_gpu, labels={"trtconverter": "true"}
    )
    node_pool_resource = container.NodePool(
        name="trt-converter-pool",
        initial_node_count=1,
        config=node_pool_config
    )
    node_pool = cluster.create_resource(node_pool_resource)

    wait_for_delete = False
    try:
        cluster.deploy(
            os.path.join("apps", "trt-converter.yaml"),
            gpu=inference_gpu
        )
        cluster.k8s_client.wait_for_deployment("trt-converter")
        ip = cluster.k8s_client.wait_for_service("trt-converter")

        input_shape = (1, len(channels), int(kernel_size * fs))

        logging.info("Exporting model")
        model.export_version(
            deepclean,
            input_shapes={"witness": input_shape},
            use_fp16=use_fp16,
            url=f"http://{ip}:5000/onnx"
        )
    except Exception as e:
        logging.error(f"Encountered error: {e}")
        wait_for_delete = True
        raise
    finally:
        # carry on without waiting for delete to complete,
        # double check for that at the end
        # cluster.k8s_client.remove_deployment("trt-converter")
        cloud_utils.utils.wait_for(
            node_pool.submit_delete,
            "Waiting to delete node pool",
            "Node pool delete request submitted"
        )
        if wait_for_delete:
            cloud_utils.utils.wait_for(
                node_pool.is_deleted,
                "Waiting for node pool to delete",
                "Node pool deleted"
            )
        pass

    ensemble = repo.create_model("deepclean_stream", platform="ensemble")
    ensemble.add_streaming_inputs(
        inputs=[model.inputs["witness"]],
        stream_size=int(kernel_stride * fs),
        name="snapshotter"
    )

    output = list(model.outputs.values())[0]
    ensemble.add_output(output)
    ensemble.export_version()

    # now export everything to the GCS bucket
    storage_client = storage.Client(credentials=manager.client.credentials)
    try:
        bucket = storage_client.get_bucket(model_repo_bucket_name)
    except exceptions.NotFound:
        bucket = storage_client.create_bucket(model_repo_bucket_name)

    for root, _, files in os.walk(repo_dir):
        for f in files:
            path = os.path.join(root, f)
            blob_path = path.replace(os.path.join(repo_dir, ""), "")

            # change path separaters in case we're on Windows
            blob_path = blob_path.replace("\\", "/")
            logging.info(f"Copying {path} to {blob_path}")

            blob = bucket.blob(blob_path)
            blob.upload_from_filename(path)

    cloud_utils.utils.wait_for(
        node_pool.is_deleted,
        "Waiting for node pool to delete",
        "Node pool deleted"
    )


if __name__ == "__main__":
    parser = typeo.make_parser(main)
    flags = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d - %(levelname)-8s %(message)s",
        stream=sys.stdout,
        datefmt="%H:%M:%S",
        level=logging.INFO
    )

    try:
        main(**vars(flags))
    finally:
        logging.info("Deleting local repository")
        shutil.rmtree("tmprepo")
