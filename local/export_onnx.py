import logging
import os
import shutil
import sys
import typing

import exportlib
import torch
import typeo
from deepclean_prod.nn.net import DeepClean
from google.api_core import exceptions
from google.cloud import storage


class StreamingDeepClean(torch.nn.Module):
    def __init__(self, dc, update_size):
        super().__init__()
        self.dc = dc
        self.update_size = update_size

    def forward(self, x):
        x = self.dc(x)
        return x[:, -self.update_size :]


def main(
    service_account_key_file: str,
    model_repo_bucket_name: str,
    kernel_size: float,
    fs: float,
    kernel_stride: float,
    channels: str,
    weights_path: typing.Optional[str] = None,
):
    # create a temporary local repo to export our model to
    repo_dir = "tmprepo"
    repo = exportlib.ModelRepository(repo_dir)
    model = repo.create_model("deepclean", platform="onnxruntime_onnx")

    # get the channel dimension of the model
    # from the specified channel list
    with open(channels, "r") as f:
        num_channels = len([i for i in f.read().splitlines() if i]) - 1

    # instantiate the Torch model and
    # load in trained weights if specified
    deepclean = DeepClean(num_channels)
    if weights_path is not None:
        state = torch.load(weights_path, map_location=torch.device("cpu"))
        deepclean.load_state_dict(state)

    # create a streaming version that slices
    # the last `kernel_stride * fs` samples off
    # and set to `eval` mode
    deepclean = StreamingDeepClean(deepclean, int(kernel_stride * fs))
    deepclean.eval()

    # export a version of this model to
    # the model repository
    input_shape = (1, num_channels, int(kernel_size * fs))
    model.export_version(deepclean, input_shapes={"witness": input_shape})

    # now expose a streaming input for this
    # model by creating an ensemble model
    # that has a snapshotter at the front
    ensemble = repo.create_model("deepclean_stream", platform="ensemble")
    ensemble.add_streaming_inputs(
        inputs=[model.inputs["witness"]],
        stream_size=int(kernel_stride * fs),
        name="snapshotter",
    )

    # expose the DeepClean output as the
    # output of the ensemble
    output = list(model.outputs.values())[0]
    ensemble.add_output(output)
    ensemble.export_version()

    # instantiate a GCP client using
    # a local credentials file
    storage_client = storage.Client.from_service_account_json(
        service_account_key_file
    )

    # create the specified bucket if it doesn't already exist
    try:
        bucket = storage_client.get_bucket(model_repo_bucket_name)
    except exceptions.NotFound:
        bucket = storage_client.create_bucket(model_repo_bucket_name)

    # now copy everything from the local temp repo
    # to the GCP repo, keeping the same structure
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
        level=logging.INFO,
    )

    try:
        main(**vars(flags))
    finally:
        logging.info("Deleting local repository")
        shutil.rmtree("tmprepo")
