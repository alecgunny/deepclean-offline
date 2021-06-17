import multiprocessing as mp
import os
import queue
import time
import typing
import yaml
from contextlib import redirect_stderr
from io import StringIO

import numpy as np
from google.api_core import exceptions
from google.cloud import storage
from gwpy.timeseries import TimeSeries, TimeSeriesDict
from requests import HTTPError
from wurlitzer import sys_pipes

from stillwater import StreamingInferenceProcess
from stillwater.utils import ExceptionWrapper, Package


def _parse_blob_fname(name):
    name = name.split(".")[0]
    timestamp, length = tuple(map(int, name.split("-")[-2:]))
    return timestamp, length


def _get_bucket(bucket_name):
    client = storage.Client()
    try:
        bucket = client.get_bucket(bucket_name)
    except exceptions.NotFound:
        bucket = client.create_bucket(bucket_name)
    except HTTPError as e:
        content = yaml.safe_load(e.response.content.decode("utf-8"))
        raise RuntimeError(
            f"Encountered HTTPError with code {e.code} "
            "and message: {}".format(content["error"]["message"])
        )
    return bucket


class GCPFrameDownloader(StreamingInferenceProcess):
    def __init__(self, bucket_name, fnames):
        bucket = _get_bucket(bucket_name)
        blobs = [blob for blob in bucket.list_blobs() if blob.name in fnames]
        self.blobs = iter(blobs)
        super().__init__(name="downloader")

    def _get_data(self):
        blob = next(self.blobs)
        fname = blob.name.split("/")[-1]
        timestamp, length = _parse_blob_fname(fname)
        fname = f"{timestamp}-{length}.gwf"

        blob.download_to_filename(fname)
        return fname, blob.name

    def _do_stuff_with_data(self, stuff):
        self._children.loader.send(stuff)


class GwfFrameDataGenerator(StreamingInferenceProcess):
    def __init__(
        self,
        sample_rate: float,
        channels: typing.List[str],
        kernel_stride: float,
        chunk_size: float,
        generation_rate: typing.Optional[float] = None,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.kernel_stride = kernel_stride
        self.chunk_size = chunk_size
        self.generation_rate = generation_rate

        self._last_time = time.time()
        self._frame = None
        self._idx = 0
        self._step = int(self.kernel_stride * self.sample_rate)
        self._self_q = mp.Queue()
        self.writer_q = mp.Queue()

        super().__init__(name="loader")

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                package = self._self_q.get_nowait()
                if isinstance(package, ExceptionWrapper):
                    package.reraise()
                return package
            except queue.Empty:
                time.sleep(1e-6)

    def _do_stuff_with_data(self, package):
        self._self_q.put(package)

    def _load_frame(self):
        while not self.stopped:
            if not self._parents.downloader.poll():
                time.sleep(1e-6)
                continue
            stuff = self._parents.downloader.recv()
            break

        if isinstance(stuff, ExceptionWrapper):
            try:
                stuff.reraise()
            except StopIteration as e:
                self._self_q.put(stuff)
                raise

        fname, blob_name = stuff
        with redirect_stderr(StringIO()), sys_pipes():
            timeseries = TimeSeriesDict.read(
                fname, channels=list(set(self.channels))
            )

        timeseries.resample(self.sample_rate)
        arrays = [timeseries[channel].value for channel in self.channels]

        strain = arrays.pop(0).astype("float32")
        frame = np.stack(arrays).astype("float32")

        self.writer_q.put((strain, blob_name, fname))
        os.remove(fname)
        return frame

    def _get_data(self):
        self._idx += 1
        start = self._idx * self._step
        stop = (self._idx + 1) * self._step

        # if we're about to exhaust the current
        # frame, try to get another from the queue
        if self._frame is None or stop > self._frame.shape[1]:
            frame = self._load_frame()

            # check if we have any data left from the old frame
            # and if so tack it to the start of the new frame
            if self._frame is not None and start < self._frame.shape[1]:
                leftover = self._frame[:, -start:]
                frame = np.concatenate([leftover, frame], axis=1)

            # reset the frame and index and update
            # the start and stop to match
            self._frame, self._idx = frame, 0
            start, stop = 0, self._step

        # pause a beat if we have a throttle
        if self.generation_rate is not None:
            wait_time = (1. / self.generation_rate - 2e-4)
            while (time.time() - self._last_time) < wait_time:
                time.sleep(1e-6)

        # create a package from the
        # current slice of the frame
        x = self._frame[:, start:stop]
        package = Package(x=x, t0=time.time())

        self._last_time = package.t0
        return package


class GwfFrameWriter(StreamingInferenceProcess):
    def __init__(
        self,
        strain_q,
        output_bucket,
        channel_name,
        sample_rate,
    ):
        self.strain_q = strain_q
        self.bucket = _get_bucket(output_bucket)
        self.channel_name = channel_name
        self.sample_rate = sample_rate

        self._strains = []
        self._noise = np.array([])
        super().__init__(name="writer")

    def _try_recv_and_check(self, conn):
        if conn.poll():
            package = conn.recv()
            if isinstance(package, ExceptionWrapper):
                package.reraise()
            return package

    def _get_data(self):
        try:
            stuff = self.strain_q.get_nowait()
        except queue.Empty:
            pass
        else:
            if isinstance(stuff, ExceptionWrapper):
                stuff.reraise()
            self._strains.append(stuff)

        return self._try_recv_and_check(self._parents.client)

    def _do_stuff_with_data(self, package):
        # add the new inferences to the
        # running noise estimate array
        self._noise = np.append(self._noise, package["output_0"].x[0])
        print(self._noise.shape)
        if len(self._noise) >= self._strains[0][0].shape[0]:
            # if we've accumulated a frame's worth of
            # noise, split it off and subtract it from
            # its corresponding strain
            strain, blob_name, fname = self._strains.pop(0)
            noise, self._noise = np.split(self._noise, [strain.shape[0]])
            t0, _ = fname.replace(".gwf", "").split("-")

            # subtract the noise estimate from the strain
            # and create a gwpy timeseries from it
            cleaned = strain - noise
            timeseries = TimeSeries(
                cleaned,
                t0=int(t0),
                sample_rate=self.sample_rate,
                channel=self.channel_name
            )

            # write the file and upload it to a blob
            # then delete it once we're done with it
            timeseries.write(fname)
            blob = self.bucket.blob(blob_name)
            blob.upload_from_filename(fname)
            os.remove(fname)

            self._children.output.send(blob_name)
