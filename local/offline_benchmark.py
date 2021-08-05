import logging
import os
import sys
import time
from itertools import cycle

import typeo
from google.cloud import storage

import utils
from utils import gcp


def main(
    service_account_key_file: str,
    ssh_key_file: str,
    username: str,
    project: str,
    zone: str,
    cluster_name: str,
    data_bucket_name: str,
    output_data_bucket_name: str,
    model_repo_bucket_name: str,
    num_nodes: int,
    gpus_per_node: int,
    clients_per_node: int,
    instances_per_gpu: int,
    vcpus_per_gpu: int,
    kernel_stride: float,
    generation_rate: float,
):
    run_config = utils.RunConfig(
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,
        clients_per_node=clients_per_node,
        instances=instances_per_gpu,
        vcpus_per_gpu=vcpus_per_gpu,
        kernel_stride=kernel_stride,
        generation_rate=generation_rate,
    )
    run_config.save()

    cluster = utils.ExperimentCluster(
        service_account_key_file=service_account_key_file,
        project=project,
        zone=zone,
        cluster_name=cluster_name,
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,
        vcpus_per_gpu=vcpus_per_gpu,
        gpu_type="v100",
    )

    storage_client = storage.Client(credentials=cluster.credentials)

    # divide our blobs up equally among all the clients
    data_bucket = storage_client.get_bucket(data_bucket_name)
    blobs = utils.get_blobs(data_bucket, run_config.total_clients)
    blobs = [blob[: len(blob) // 2] for blob in blobs]

    # set up the Triton snapshotter config so
    # that the appropriate number of snapshot
    # instances are available on each node
    model_repo_bucket = storage_client.get_bucket(model_repo_bucket_name)
    streams_per_gpu = utils.split_count(clients_per_node, gpus_per_node)
    utils.update_model_configs(
        model_repo_bucket, streams_per_gpu, instances_per_gpu
    )

    # set up a VM manager with a connection
    # we can use to execute commands over
    # SSH and copy output from the run via SCP
    client_manager = gcp.ClientVMManager(
        project=project,
        zone=zone,
        prefix="o2-client",
        service_account_key_file=service_account_key_file,
        connection=gcp.VMConnection(username, ssh_key_file),
    )
    with cluster:
        cluster.deploy_servers(model_repo_bucket_name)

        client_manager.create_instances(run_config.total_clients, 8)
        with client_manager:
            utils.configure_vms_parallel(client_manager.instances)
            ips = cluster.get_ips()

            runner = utils.RunParallel(
                model_name="deepclean_stream",
                model_version=1,
                generation_rate=500,
                sequence_id=1001,
                bucket_name=data_bucket_name,
                output_bucket_name=output_data_bucket_name,
                kernel_stride=kernel_stride,
                sample_rate=4096,
            )

            # run the clients while monitoring the server,
            # keep track of how much time it takes us
            stats_fname = os.path.join(
                run_config.output_dir, "server-stats.csv"
            )
            start_time = time.time()
            with utils.ServerMonitor(ips, stats_fname) as monitor:
                runner(client_manager.instances, blobs, cycle(ips))
                end_time = time.time()

            # check to make sure the monitor didn't encounter
            # any errors and return the time delta for the run
            monitor.check()
            logging.info("Completed in {} s".format(end_time - start_time))


if __name__ == "__main__":
    parser = typeo.make_parser(main)
    flags = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d - %(levelname)-8s %(message)s",
        stream=sys.stdout,
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    main(**vars(flags))
