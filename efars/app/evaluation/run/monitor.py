from multiprocessing import Pool
import concurrent
from functools import partial
import time
import datetime
import docker
import shutil
import pathlib
import os
import collections


class Monitor():

    def __init__(self, container_names, docker_sock, run_index):
        self.docker_sock = docker_sock
        self.low_level_docker_client = docker.APIClient(base_url=docker_sock)
        self.docker_client = docker.from_env()
        self.container_ids = list(
            map(lambda id: self.get_dockerid(id), container_names))
        self.target_folder = os.path.join(
            "data", "runs", "{0}".format(run_index), "monitor")
        self.clean_up()
        self.worker_pool = concurrent.futures.ProcessPoolExecutor(max_workers=len(self.container_ids))

    def get_dockerid(self, container_name):
        return self.docker_client.containers.get(container_name).id

    def clean_up(self):
        if os.path.isdir(self.target_folder):
            shutil.rmtree(self.target_folder)
        pathlib.Path(self.target_folder).mkdir(parents=True, exist_ok=True)

    def tear_down(self):
        self.worker_pool.shutdown()

    def measure(self):
        for cid in self.container_ids:
            self.worker_pool.submit(take_measurement, cid, self.docker_sock, self.target_folder)

def take_measurement(id, docker_sock, target_folder):
    docker_client = docker.APIClient(base_url=docker_sock)
    stats = docker_client.stats(id, stream=False)
    container_name = docker_client.inspect_container(id)['Name'][1:]
    log_stats(stats, container_name, target_folder)


def log_stats(stats, file_name, target_folder):
    read_at = stats["read"]
    pids_stats = stats["pids_stats"]
    pids_path = os.path.join(
        target_folder, "pids_{0}.csv".format(file_name))
    cpu_path = os.path.join(
        target_folder, "cpu_{0}.csv".format(file_name))
    memory_path = os.path.join(
        target_folder, "memory_{0}.csv".format(file_name))
    io_path = os.path.join(
        target_folder, "blkio_{0}.csv".format(file_name))
    cpu_stats = {"cpu_stats":  stats["cpu_stats"],
                 "precpu_stats": stats["precpu_stats"]}
    memory_stats = stats["memory_stats"]
    blkio_stats = stats['blkio_stats']
    collect_and_write(pids_stats, read_at, pids_path)
    collect_and_write(cpu_stats, read_at, cpu_path)
    collect_and_write(memory_stats, read_at, memory_path)
    collect_and_write(blkio_stats, read_at, io_path)


def collect_and_write(data, read_at, target_path):
    # Write all the data from memory
    flattened_data = flatten(data)
    loggable_data = [read_at] + list(map(str, flattened_data.values()))
    # check if file empty:
    if not os.path.exists(target_path):
        write_log(target_path,  ["read_at"] + list(flattened_data.keys()))
    write_log(target_path,  loggable_data)


def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            # another hash map that can be mapped
            items.extend(flatten(v, new_key, sep=sep).items())
        elif type(v) == list:
            # another list will be converted in a indexed dict
            items.extend(flatten({str(i): key for i, key in enumerate(v)} , new_key, sep=sep).items())
        else:            
            items.append((new_key, v))
    return dict(items)


def write_log(filePath, tupel):
    if filePath is None:
        raise ValueError()
    if tupel is None:
        raise ValueError()
    fw = open(filePath, "a")
    fw.write(','.join(["\"{0}\"".format(value) for value in tupel]) + "\r\n")
    fw.close()
