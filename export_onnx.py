import logging
import os
import shutil
import sys
import typing

import torch
from google.api_core import exceptions
from google.cloud import storage

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
    model_repo_bucket_name: str,
    kernel_size: float,
    fs: float,
    kernel_stride: float,
    channels: str,
    weights_path: typing.Optional[str] = None,
):
    repo_dir = "tmprepo"
    if not os.path.exists(repo_dir):
        os.makedirs(repo_dir)
    repo = exportlib.ModelRepository(repo_dir)
    model = repo.create_model("deepclean", platform="onnxruntime_onnx")

    with open(channels, "r") as f:
        channels = [i for i in f.read().splitlines() if i][1:]

    deepclean = DeepClean(len(channels))
    if weights_path is not None:
        state = torch.load(weights_path, map_location=torch.device("cpu"))
        deepclean.load_state_dict(state)
    deepclean.eval()
    deepclean = StreamingDeepClean(deepclean, int(kernel_stride * fs))

    input_shape = (1, len(channels), int(kernel_size * fs))
    model.export_version(deepclean, input_shapes={"witness": input_shape})

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
    storage_client = storage.Client.from_service_account_json(
        service_account_key_file
    )
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
