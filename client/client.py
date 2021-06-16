import logging
import time
import typing

import typeo
from stillwater import ExceptionWrapper, StreamingInferenceClient
from frame_reader import (
    GCPFrameDownloader,
    GwfFrameDataGenerator,
    GwfFrameWriter
)


def main(
    url: str,
    model_name: str,
    model_version: int,
    generation_rate: float,
    sequence_id: int,
    bucket_name: str,
    output_bucket_name: str,
    fnames: typing.List[str],
    kernel_stride: float,
    sample_rate: float = 4000,
    chunk_size: float = 1024,
    start_time: float = None
):
    client = StreamingInferenceClient(
        url=url,
        model_name=model_name,
        model_version=model_version,
        qps_limit=generation_rate,
        name="client"
    )

    with open("../channels.txt", "r") as f:
        channels = [i for i in f.read().splitlines() if i]

    # downloader pulls the frames down from GCP to local files
    downloader = GCPFrameDownloader(bucket_name, fnames)

    # data generator iterates through these local
    # files and prepares streaming updates from them
    source = GwfFrameDataGenerator(
        sample_rate=sample_rate,
        channels=channels,
        kernel_stride=kernel_stride,
        chunk_size=chunk_size,
        generation_rate=generation_rate
    )

    # route the downloader to the data generator,
    # to which it will pass its saved filenames
    downloader.add_child(source)

    # the writer combines the outputs from the
    # client and subtracts them from the strain
    # channel to produce cleaned strains
    writer = GwfFrameWriter(
        source.writer_q,
        output_bucket_name,
        channels[0],
        sample_rate
    )

    # need to route the outputs from the data
    # generator into the client, the outputs
    # of which will be passed to the writer
    # for aggregation
    client.add_source(source, writer, sequence_id)

    # now get the written blob names from
    # the writer for logging purposes
    pipe = writer.add_child("output")

    while time.time() < start_time:
        time.sleep(1e-3)

    logging.info("Starting processes")
    timeout, stopped = 20, False
    with client, downloader, writer:
        while True:
            tick = time.time()
            while (time.time() - tick) < timeout:
                if pipe.poll():
                    fname = pipe.recv()
                    break
            else:
                if stopped:
                    logging.info("Timed out, breaking")
                    break

            try:
                if isinstance(fname, ExceptionWrapper):
                    fname.reraise()
            except StopIteration:
                logging.info("StopIteration raised")
                stopped = True
                continue

            logging.info(f"Wrote frame {fname}")


if __name__ == "__main__":
    parser = typeo.make_parser(main)
    parser.add_argument(
        "--log-file",
        type=str,
        default="client.log"
    )
    flags = vars(parser.parse_args())
    logging.basicConfig(filename=flags.pop("log_file"), level=logging.INFO)

    main(**flags)
