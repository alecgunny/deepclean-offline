import concurrent.futures
import logging
import multiprocessing as mp
import os
import pickle
import queue
import re
import threading
import time
from zlib import adler32
from functools import lru_cache

import attr
import cloud_utils as cloud
import requests
from cloud_utils.utils import wait_for
from google.cloud import container_v1 as container
from google.cloud import storage


_PACKAGE = "deepclean-offline"
_PACKAGE_URL = f"https://github.com/alecgunny/{_PACKAGE}.git"
_RUN = f"./{_PACKAGE}/client/run.sh"
_MODELS = ["snapshotter", "deepclean"]
logging.getLogger("paramiko").setLevel(logging.CRITICAL)


def split_count(total, size):
    return (total - 1) // size + 1


def get_blobs(bucket, total_clients):
    blobs = bucket.list_blobs(prefix="asc_inmon_rds/H1")
    timestamp_re = re.compile("(?<=-)[0-9]{10}(?=-)")

    def sort_key(x):
        fname = x.split("/")[-1]
        return timestamp_re.search(fname).group(0)

    blob_names = sorted([blob.name for blob in blobs], key=sort_key)

    # I don't feel like explaining the code below
    # and there's no way it's optimal but basically
    # we're breaking up the blobs as evenly as possible
    # among all the clients and keeping them in order
    blobs_per_client = split_count(len(blob_names), total_clients)
    remainder = blobs_per_client - len(blob_names) % blobs_per_client
    blobs = []
    for i in range(total_clients - remainder):
        blobs.append(
            blob_names[i * blobs_per_client: (i + 1) * blobs_per_client]
        )

    start_idx = (i + 1) * blobs_per_client
    for i in range(remainder):
        blobs.append(
            blob_names[
                start_idx + i * (blobs_per_client - 1):
                start_idx + (i + 1) * (blobs_per_client - 1)
            ]
        )
    return blobs


def update_model_configs(
    bucket: storage.Bucket,
    streams: int,
    instances: int
) -> None:
    count_re = re.compile("(?<=\n  count: )[0-9]+(?=\n)")

    for blob in bucket.list_blobs():
        # only updating config protobufs
        if not blob.name.endswith("config.pbtxt"):
            continue

        model_name = blob.name.split("/")[0]
        if model_name == "snapshotter":
            count = streams
        elif model_name == "deepclean":
            count = instances
        else:
            continue
        logging.info(
            f"Scaling model {model_name} to count {count}"
        )

        # replace the instance group count
        # in the config protobuf
        config_str = blob.download_as_bytes().decode()
        config_str = count_re.sub(str(count), config_str)

        # delete the existing blob and
        # replace it with the updated config
        blob_name = blob.name
        blob.delete()
        blob = bucket.blob(blob_name)
        blob.upload_from_string(
            config_str, content_type="application/octet-stream"
        )


def _run_in_pool(fn, args, msg, exit_msg, max_workers=None):
    q = queue.Queue()

    def wrapper(*args):
        try:
            result = fn(*args)
            q.put(result)
        except Exception as e:
            q.put(e)

    threads = []
    for arg in args:
        if not isinstance(arg, tuple):
            arg = (arg,)
        t = threading.Thread(target=wrapper, args=arg)
        t.start()
        threads.append(t)

    results = []

    def _callback():
        try:
            result = q.get_nowait()
            if isinstance(result, Exception):
                raise result
            results.append(result)
        except queue.Empty:
            pass
        return not any([t.is_alive() for t in threads])

    wait_for(_callback, msg, exit_msg)
    return results


def configure_vm(vm):
    vm.wait_until_ready(verbose=False)

    cmds = [f"git clone -q {_PACKAGE_URL}", f"{_RUN} install", f"{_RUN} create"]
    for cmd in cmds:
        _, err = vm.run(cmd)
        if err:
            raise RuntimeError(f"Command {cmd} failed with message {err}")


def configure_vms_parallel(vms):
    _run_in_pool(
        configure_vm,
        vms,
        msg="Waiting for VMs to configure",
        exit_msg="Configured all VMs",
        max_workers=min(32, len(vms))
    )


class ExperimentCluster:
    def __init__(
        self,
        service_account_key_file: str,
        project: str,
        zone: str,
        cluster_name: str,
        num_nodes: int,
        gpus_per_node: int,
        vcpus_per_gpu: int,
        gpu_type: str = "t4"
    ):
        self._manager = cloud.GKEClusterManager(
            project=project, zone=zone, credentials=service_account_key_file
        )
        self._cluster_config = container.Cluster(
            name=cluster_name,
            node_pools=[container.NodePool(
                name="default-pool",
                initial_node_count=2,
                config=container.NodeConfig()
            )]
        )
        vcpus_per_node = vcpus_per_gpu * gpus_per_node
        node_pool_config = cloud.create_gpu_node_pool_config(
            vcpus=vcpus_per_node,
            gpus=gpus_per_node,
            gpu_type=gpu_type,
        )
        self._node_pool_config = container.NodePool(
            name=f"tritonserver-{gpu_type}-pool",
            initial_node_count=num_nodes,
            config=node_pool_config
        )
        self.gpu_type = gpu_type

        self._cluster = None
        self._node_pool = None

    @property
    def credentials(self):
        return self._manager.client.credentials

    @property
    def k8s_client(self):
        if self._cluster is None:
            raise ValueError("No active cluster!")
        return self._cluster.k8s_client

    def __enter__(self):
        self._cluster = self._manager.create_resource(self._cluster_config)
        self._cluster.deploy_gpu_drivers()
        self._node_pool = self._cluster.create_resource(self._node_pool_config)
        return self

    def __exit__(self, *exc_args):
        self._manager.delete_resource(self._cluster)

    def deploy_servers(self, bucket_name):
        if self._cluster is None:
            raise ValueError("No active cluster!")

        node_config = self._node_pool.get()
        vcpus = int(node_config.config.machine_type.split("-")[-1]) - 1
        gpus = node_config.config.accelerators[0].accelerator_count

        for i in range(node_config.initial_node_count):
            self._cluster.deploy(
                os.path.join("apps", "tritonserver.yaml"),
                gpus=gpus,
                tag="20.11",
                vcpus=vcpus,
                bucket=bucket_name,
                name=f"tritonserver-{i}",
                gpu=self.gpu_type
            )

    def get_ips(self):
        if self._cluster is None:
            raise ValueError("No active cluster!")

        node_config = self._node_pool.get()
        ips = []
        for i in range(node_config.initial_node_count):
            self.k8s_client.wait_for_deployment(name=f"tritonserver-{i}")
            ip = self.k8s_client.wait_for_service(name=f"tritonserver-{i}")
            ips.append(ip)
        return ips


class ExperimentClusterCPU:
    def __init__(
        self,
        service_account_key_file: str,
        project: str,
        zone: str,
        cluster_name: str,
        num_nodes: int,
        vcpus: int
    ):
        self._manager = cloud.GKEClusterManager(
            project=project, zone=zone, credentials=service_account_key_file
        )
        self._cluster_config = container.Cluster(
            name=cluster_name,
            node_pools=[container.NodePool(
                name="default-pool",
                initial_node_count=2,
                config=container.NodeConfig()
            )]
        )
        node_pool_config = cloud.create_gpu_node_pool_config(
            vcpus=vcpus,
            gpus=4,
            gpu_type="t4"
        )

        self._node_pool_config = container.NodePool(
            name="tritonserver-cpu-pool",
            initial_node_count=num_nodes,
            config=node_pool_config
        )
        self._cluster = None
        self._node_pool = None

    @property
    def credentials(self):
        return self._manager.client.credentials

    @property
    def k8s_client(self):
        if self._cluster is None:
            raise ValueError("No active cluster!")
        return self._cluster.k8s_client

    def __enter__(self):
        self._cluster = self._manager.create_resource(self._cluster_config)
        self._cluster.deploy_gpu_drivers()
        self._node_pool = self._cluster.create_resource(self._node_pool_config)
        return self

    def __exit__(self, *exc_args):
        self._manager.delete_resource(self._cluster)

    def deploy_servers(self, bucket_name):
        if self._cluster is None:
            raise ValueError("No active cluster!")

        node_config = self._node_pool.get()
        for i in range(node_config.initial_node_count):
            self._cluster.deploy(
                os.path.join("apps", "tritonserver-cpu.yaml"),
                tag="20.11",
                bucket=bucket_name,
                name=f"tritonserver-{i}",
            )

    def get_ips(self):
        if self._cluster is None:
            raise ValueError("No active cluster!")

        node_config = self._node_pool.get()
        ips = []
        for i in range(node_config.initial_node_count):
            self.k8s_client.wait_for_deployment(name=f"tritonserver-{i}")
            ip = self.k8s_client.wait_for_service(name=f"tritonserver-{i}")
            ips.append(ip)
        return ips


@attr.s(auto_attribs=True)
class RunParallel:
    model_name: str
    model_version: int
    generation_rate: float
    sequence_id: int
    bucket_name: str
    output_bucket_name: str
    kernel_stride: float
    sample_rate: float = 4096
    chunk_size: float = 1024

    def __attrs_post_init__(self):
        self._start_time = None

    @property
    def command(self):
        command = f"{_RUN} run"
        for a in self.__attrs_attrs__:
            if a.name != "sequence_id":
                command += " --{} {}".format(
                    a.name.replace("_", "-"), self.__dict__[a.name]
                )
        if self._start_time is not None:
            command += f" --start-time {self._start_time}"

        return command + (
            " --url {ip}:8001 --sequence-id {sequence_id} --fnames {fnames}"
        )

    def run_on_vm(self, vm, fnames, ip, sequence_id):
        fnames = " ".join(fnames)
        command = self.command.format(
            ip=ip, sequence_id=sequence_id, fnames=fnames
        )
        out, err = vm.run(command)
        if err:
            raise RuntimeError(f"Encountered error on client: {err}")

        # parse out framecpp stderr info
        # err_re = re.compile("^Loading: Fr.+$", re.MULTILINE)
        # err = err_re.sub("", err).strip()

    def __call__(self, vms, fnames, ips):
        seq_ids = [self.sequence_id + i for i in range(len(vms))]
        args = zip(vms, fnames, ips, seq_ids)

        self._start_time = time.time() + 60
        _run_in_pool(
            self.run_on_vm,
            args,
            "Waiting for tasks to complete",
            "All tasks completed",
            max_workers=min(32, len(vms))
        )


_hexes = "[0-9a-f]"
_gpu_id_pattern = "-".join([_hexes + f"{{{i}}}" for i in [8, 4, 4, 4, 12]])
_res = [
    re.compile('(?<=nv_inference_)[a-z_]+(?=_duration_us)'),
    re.compile(f'(?<=gpu_uuid="GPU-){_gpu_id_pattern}(?=")'),
    re.compile(f'(?<=model=")[a-z_-]+(?=",version=)'),
    re.compile("(?<=} )[0-9.]+$")
]


class ServerMonitor(mp.Process):
    def __init__(self, ips, filename):
        self.ips = ips
        self.filename = filename

        self.header = (
            "ip,step,gpu_id,model,process,time (us),interval,count,utilization"
        )
        self._last_times = {}
        self._counts = {}
        self._times = {}

        self._stop_event = mp.Event()
        self._error_q = mp.Queue()
        super().__init__()

    def _get_data_for_ip(self, ip, step):
        response = requests.get(f"http://{ip}:8002/metrics")
        response.raise_for_status()

        request_time = time.time()
        try:
            last_time = self._last_times[ip]
            interval = request_time - last_time
        except KeyError:
            pass
        finally:
            self._last_times[ip] = request_time

        data, counts, utilizations = "", {}, {}
        models_to_update = []
        rows = response.content.decode().split("\n")

        # start by collecting the number of new inference
        # counts and the GPU utilization
        for row in rows:
            if row.startswith("nv_inference_exec_count"):
                try:
                    gpu_id, model, value = [
                        r.search(row).group(0) for r in _res[1:]
                    ]
                except AttributeError:
                    continue

                if model in _MODELS:
                    value = int(float(value))
                    try:
                        count = value - self._counts[(ip, gpu_id, model)]
                        if count > 0:
                            # if no new inferences were registered, we
                            # won't need to update this model below
                            models_to_update.append((gpu_id, model))
                        counts[(ip, gpu_id, model)] = count
                    except KeyError:
                        # we haven't recorded this model before, so
                        # there's no need to update it
                        pass
                    finally:
                        # add or update the number of inference counts
                        # for this model on this GPU
                        self._counts[(ip, gpu_id, model)] = value

            elif row.startswith("nv_gpu_utilization"):
                try:
                    gpu_id, value = [r.search(row).group(0) for r in _res[1::2]]
                except AttributeError:
                    continue
                utilizations[gpu_id] = value

        for row in rows:
            try:
                process, gpu_id, model, value = [
                    r.search(row).group(0) for r in _res
                ]
            except AttributeError:
                continue

            value = float(value)
            index = (ip, process, gpu_id, model)

            if (gpu_id, model) not in models_to_update:
                # we don't need to record a row of data for this
                # GPU/model combination, either because this is
                # the first loop or because we didn't record any
                # new inferences during this interval
                if model in _MODELS and index not in self._times:
                    # we don't have a duration for this process
                    # on this node/GPU/model combo, so create one
                    self._times[index] = value
                continue

            delta = value - self._times[index]
            self._times[index] = value
            utilization = utilizations[gpu_id]
            count = counts[(ip, gpu_id, model)]

            data += "\n" + ",".join([
                ip,
                str(step),
                gpu_id,
                model,
                process,
                str(delta),
                str(interval),
                str(count),
                utilization
            ])
        return data

    @property
    def stopped(self) -> bool:
        return self._stop_event.is_set()

    def stop(self) -> None:
        self._stop_event.set()

    def check(self):
        try:
            e = self._error_q.get_nowait()
        except queue.Empty:
            return
        raise RuntimeError("Error in monitor: " + e)

    def run(self):
        f = open(self.filename, "w")
        f.write(self.header)

        lock = threading.Lock()

        def target(ip):
            step = 0
            try:
                while not self.stopped:
                    data = self._get_data_for_ip(ip, step)
                    if data:
                        with lock:
                            f.write(data)
                        step += 1
            except Exception:
                self.stop()
                raise

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(target, ip) for ip in self.ips]

        try:
            while len(futures) > 0:
                done, futures = concurrent.futures.wait(futures, timeout=1e-2)
                for future in done:
                    exc = future.exception()
                    if exc is not None:
                        raise exc
        except Exception as e:
            self._error_q.put(str(e))
        finally:
            f.close()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc_args):
        self.stop()
        self.join(1)
        try:
            self.close()
        except ValueError:
            self.terminate()


class ServerMonitorCPU(mp.Process):
    def __init__(self, ips, filename):
        self.ips = ips
        self.filename = filename

        self.header = (
            "ip,step,model,process,time (us),interval,count"
        )
        self._last_times = {}
        self._counts = {}
        self._times = {}

        self._stop_event = mp.Event()
        self._error_q = mp.Queue()
        self._models = ["deepclean-cpu", "snapshotter-cpu"]
        super().__init__()

    def _get_data_for_ip(self, ip, step):
        response = requests.get(f"http://{ip}:8002/metrics")
        response.raise_for_status()

        request_time = time.time()
        try:
            last_time = self._last_times[ip]
            interval = request_time - last_time
        except KeyError:
            pass
        finally:
            self._last_times[ip] = request_time

        data, counts = "", {}
        models_to_update = []
        rows = response.content.decode().split("\n")

        # start by collecting the number of new inference
        # counts and the GPU utilization
        for row in rows:
            if row.startswith("nv_inference_exec_count"):
                try:
                    model, value = [
                        r.search(row).group(0) for r in _res[2:]
                    ]
                except AttributeError:
                    continue

                if model in self._models:
                    value = int(float(value))
                    try:
                        count = value - self._counts[(ip, model)]
                        if count > 0:
                            # if no new inferences were registered, we
                            # won't need to update this model below
                            models_to_update.append(model)
                        counts[(ip, model)] = count
                    except KeyError:
                        # we haven't recorded this model before, so
                        # there's no need to update it
                        pass
                    finally:
                        # add or update the number of inference counts
                        # for this model on this GPU
                        self._counts[(ip, model)] = value

        for row in rows:
            try:
                process, model, value = [
                    r.search(row).group(0) for r in _res[:1] + _res[2:]
                ]
            except AttributeError:
                continue

            value = float(value)
            index = (ip, process, model)

            if model not in models_to_update:
                # we don't need to record a row of data for this
                # GPU/model combination, either because this is
                # the first loop or because we didn't record any
                # new inferences during this interval
                if model in self._models and index not in self._times:
                    # we don't have a duration for this process
                    # on this node/GPU/model combo, so create one
                    self._times[index] = value
                continue

            delta = value - self._times[index]
            self._times[index] = value
            count = counts[(ip, model)]

            data += "\n" + ",".join([
                ip,
                str(step),
                model,
                process,
                str(delta),
                str(interval),
                str(count)
            ])
        return data

    @property
    def stopped(self) -> bool:
        return self._stop_event.is_set()

    def stop(self) -> None:
        self._stop_event.set()

    def check(self):
        try:
            e = self._error_q.get_nowait()
        except queue.Empty:
            return
        raise RuntimeError("Error in monitor: " + e)

    def run(self):
        f = open(self.filename, "w")
        f.write(self.header)

        lock = threading.Lock()

        def target(ip):
            step = 0
            try:
                while not self.stopped:
                    data = self._get_data_for_ip(ip, step)
                    if data:
                        with lock:
                            f.write(data)
                        step += 1
            except Exception:
                self.stop()
                raise

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(target, ip) for ip in self.ips]

        try:
            while len(futures) > 0:
                done, futures = concurrent.futures.wait(futures, timeout=1e-2)
                for future in done:
                    exc = future.exception()
                    if exc is not None:
                        raise exc
        except Exception as e:
            self._error_q.put(str(e))
        finally:
            f.close()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc_args):
        self.stop()
        self.join(1)
        try:
            self.close()
        except ValueError:
            self.terminate()


@attr.s(auto_attribs=True, frozen=True)
class RunConfig:
    num_nodes: int
    gpus_per_node: int
    clients_per_node: int
    instances: int
    vcpus_per_gpu: int
    kernel_stride: float
    generation_rate: float

    def __attrs_post_init__(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    @property
    def output_dir(self):
        return os.path.join("analysis", "data", self.id)

    def save(self):
        with open(os.path.join(self.output_dir, "config.pkl"), "wb") as f:
            pickle.dump(self, f)

    @lru_cache(None)
    def _make_string(self):
        string = "{"
        for a in self.__attrs_attrs__:
            value = self.__dict__[a.name]
            if a.type is float:
                value = float(value)
            value = str(value)

            if "\n" in value:
                lines = value.split("\n")
                lines = lines[:1] + ["\t" + line for line in lines[1:]]
                value = "\n".join(lines)
            string += f"\n\t{a.name}: {value},"
        string = string[:-1]
        return string + "\n}"

    @property
    def total_clients(self):
        return self.clients_per_node * self.num_nodes

    def __str__(self):
        return f"RunConfig {self.id} " + self._make_string()

    @property
    def id(self):
        string = self._make_string()
        return hex(adler32(string.encode())).split("x")[1]
