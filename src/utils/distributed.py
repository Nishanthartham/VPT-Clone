#!/usr/bin/env python3
#this code is from detectron2
"""Distributed helpers."""

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
_LOCAL_PROCESS_GROUP = None
import os

def get_world_size() -> int:
    print(f"dist.is_available={dist.is_available}")
    print(f"dist.is_initialized={dist.is_initialized}")
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def create_local_process_group(num_workers_per_machine: int) -> None:
    """
    Create a process group that contains ranks within the same machine.

    Detectron2's launch() in engine/launch.py will call this function. If you start
    workers without launch(), you'll have to also call this. Otherwise utilities
    like `get_local_rank()` will not work.

    This function contains a barrier. All processes must call it together.

    Args:
        num_workers_per_machine: the number of worker processes per machine. Typically
          the number of GPUs.
    """
    global _LOCAL_PROCESS_GROUP
    assert _LOCAL_PROCESS_GROUP is None
    assert get_world_size() % num_workers_per_machine == 0 #me Checking if mentioned no of gpus are all active
    num_machines = get_world_size() // num_workers_per_machine
    machine_rank = get_rank() // num_workers_per_machine
    for i in range(num_machines):
        ranks_on_i = list(range(i * num_workers_per_machine, (i + 1) * num_workers_per_machine))
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            _LOCAL_PROCESS_GROUP = pg

def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_master_process(num_gpus=8):
    """
    Determines if the current process is the master process.
    """
    if torch.distributed.is_initialized():
        return dist.get_rank() % num_gpus == 0
    else:
        return True

def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def setup_process(rank, master_addr, master_port, world_size, backend='gloo'):
    print(f'setting up rank = {rank} world_size={world_size} backend={backend}')

    # set up the master's ip address so this child process can coordinate
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    print(f"master_addr={master_addr} master_port={master_port}")

    # Initializes the default distributed process group, and this will also initialize the distributed package.
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    print(f"true = {dist.is_initialized()}")

    print(f"rank = {rank} init complete")
    print(f"rank = {rank} destroy complete")
    dist.destroy_process_group()

        

def run(
    rank,
    master_addr,
    master_port,
    world_size,
    backend,
    # init_method,
    # shard_id,
    # num_shards,
    # backend,
    cfg,
    *args,
):
    """
    Runs a function from a child process.
    Args:
        local_rank (int): rank of the current process on the current machine.
        num_proc (int): number of processes per machine.
        func (function): function to execute on each of the process.
        init_method (string): method to initialize the distributed training.
            TCP initialization: equiring a network address reachable from all
            processes followed by the port.
            Shared file-system initialization: makes use of a file system that
            is shared and visible from all machines. The URL should start with
            file:// and contain a path to a non-existent file on a shared file
            system.
        shard_id (int): the rank of the current machine.
        num_shards (int): number of overall machines for the distributed
            training job.
        backend (string): three distributed backends ('nccl', 'gloo', 'mpi') are
            supports, each with different capabilities. Details can be found
            here:
            https://pytorch.org/docs/stable/distributed.html
        cfg (CfgNode): configs. Details can be found in
            loco/config/defaults.py
    """
    # Initialize the process group.
    # create_local_process_group(cfg.NUM_GPUS)
    # shard_id = get_rank()
    # num_shards=1
    # backend=cfg.DIST_BACKEND
    # world_size = cfg.NUM_GPUS
    # world_size = num_proc * num_shards

    # world_size = 3

    # print(f"world_size = {world_size}, rank = {rank}")
    # init_method = f'{cfg.DIST_INIT_PATH}:{port}'
    # print(f'setting up rank = {rank} world_size={world_size} backend={backend}')
    print(f'params => master_addr = {master_addr}, master_port={master_port},backend={backend},*args = {args} setting up rank = {rank} world_size={world_size} backend={backend}')
    # os.environ['MASTER_ADDR'] = master_addr
    # os.environ['MASTER_PORT'] = master_port

    # try:
    #     torch.distributed.init_process_group(
    #         backend=backend,
    #         world_size=world_size,
    #         rank=rank,
    #     )
    #     print(f"dist.is_initialized = {dist.is_initialized()}")

    # except Exception as e:
    #     raise e

    # torch.cuda.set_device(rank)
    # # func(cfg, args)
    sampler = DistributedSampler(*args)
    # loader = torch.utils.data.DataLoader(
    #     args[0],
    #     batch_size=args[1],
    #     shuffle=(False if sampler else shuffle),
    #     sampler=sampler,
    #     num_workers=cfg.DATA.NUM_WORKERS,
    #     pin_memory=cfg.DATA.PIN_MEMORY,
    #     drop_last=args[2],
    # )
    # return loader
    # sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=cfg.NUM_GPUS, rank=rank)

    # Create DataLoader with distributed sampler
    loader = torch.utils.data.DataLoader(
        args[0],
        batch_size=args[1],
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATA.NUM_WORKERS,
        pin_memory=cfg.DATA.PIN_MEMORY,
        drop_last=args[2],
    )

    # Put the DataLoader into the queue for communication with _construct_loader
    args[3].put(loader)

    

def destroy_process_group():
    """Destroys the default process group."""
    torch.distributed.destroy_process_group()


def scaled_all_reduce(cfg, tensors):
    """Performs the scaled all_reduce operation on the provided tensors.

    The input tensors are modified in-place. Currently supports only the sum
    reduction operator. The reduced values are scaled by the inverse size of
    the process group (equivalent to cfg.NUM_GPUS).
    """
    # Queue the reductions
    reductions = []
    for tensor in tensors:
        reduction = torch.distributed.all_reduce(tensor, async_op=True)
        reductions.append(reduction)
    # Wait for reductions to finish
    for reduction in reductions:
        reduction.wait()
    # Scale the results
    for tensor in tensors:
        tensor.mul_(1.0 / cfg.NUM_GPUS / cfg.NUM_SHARDS)
    return tensors


def cat_all_gather(tensors):
    """Performs the concatenated all_gather operation on the provided tensors.
    """
    tensors_gather = [
        torch.ones_like(tensors)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensors, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def local_cat_all_gather(tensors):
    """Performs the concatenated all_gather operation on the provided tensors.
    """
    tensors_gather = [
        torch.ones_like(tensors)
        for _ in range(get_local_size())
    ]
    torch.distributed.all_gather(
        tensors_gather,
        tensors,
        async_op=False,
        group=_LOCAL_PROCESS_GROUP,
    )
    output = torch.cat(tensors_gather, dim=0)
    return output


def get_local_size():
    """
    Returns:
        The size of the per-machine process group,
        i.e. the number of processes per machine.
    """
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size(group=_LOCAL_PROCESS_GROUP)


def get_local_rank():
    """
    Returns:
        The rank of the current process within the local (per-machine) process group.
    """
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    assert _LOCAL_PROCESS_GROUP is not None
    return dist.get_rank(group=_LOCAL_PROCESS_GROUP)
