# # loader.py with distributed

#!/usr/bin/env python3
 
"""Data loader.""" 
import torch
import torch.distributed as dist
import os
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
import torch.multiprocessing as mp
from ..utils.distributed import run
from ..utils import logging
from .datasets.json_dataset import (
    CUB200Dataset, CarsDataset, DogsDataset, FlowersDataset, NabirdsDataset
)

logger = logging.get_logger("visual_prompt")
_DATASET_CATALOG = {
    "CUB": CUB200Dataset,
    'OxfordFlowers': FlowersDataset,
    'StanfordCars': CarsDataset,
    'StanfordDogs': DogsDataset,
    "nabirds": NabirdsDataset,
}

def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])

def _construct_loader(cfg, split, batch_size, shuffle, drop_last):
    """Constructs the data loader for the given dataset."""
    dataset_name = cfg.DATA.NAME

    # Construct the dataset
    if dataset_name.startswith("vtab-"):
        # import the tensorflow here only if needed
        from .datasets.tf_dataset import TFDataset
        dataset = TFDataset(cfg, split)
    else:
        assert (
            dataset_name in _DATASET_CATALOG.keys()
        ), "Dataset '{}' not supported".format(dataset_name)
        dataset = _DATASET_CATALOG[dataset_name](cfg, split)
    # Create a sampler for multi-process training
    # print("before creating sampler ",sampler)
#############################Change################
    # sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None
    sampler =  None

    # sampler = run(DistributedSampler,dataset) if cfg.NUM_GPUS > 1 else None
    # sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None
    # Create a loader
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(False if sampler else shuffle),
        sampler=sampler,
        num_workers=cfg.DATA.NUM_WORKERS,
        pin_memory=cfg.DATA.PIN_MEMORY,
        drop_last=drop_last,
    )
    return loader
    # master_port=find_free_port()
    # master_addr=cfg.DIST_INIT_PATH
    # backend="nccl"
    # fun_params = {"num_workers":cfg.DATA.NUM_WORKERS,"pin_memory":cfg.DATA.PIN_MEMORY}
    # loader = mp.spawn(run,nprocs=cfg.NUM_GPUS,args=(master_addr,master_port,cfg.NUM_GPUS,backend,cfg,dataset,batch_size,drop_last))
    """
#new code
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    world_size = 3
    try:
        print("starting")
        torch.distributed.init_process_group(
            backend=backend,
            world_size=world_size,
            rank=0,
        )
        print(f"dist.is_initialized = {dist.is_initialized()}")

    except Exception as e:
        raise e

    queue = mp.Queue()

    # Spawn processes for data loading
    processes = []
    for i in range(cfg.NUM_GPUS):
        p = mp.Process(target=a.run, args=(i, cfg, dataset, batch_size, drop_last, queue))
        p.start()
        processes.append(p)

    # Collect DataLoader objects from the queue
    loaders = []
    for _ in range(cfg.NUM_GPUS):
        loader = loader_queue.get()
        loaders.append(loader)

    # Wait for all processes to finish
    for p in processes:
        p.join()

    return loaders
    print(f"loader",loaders)
    # return loader

 """

def construct_train_loader(cfg):
    print("Inside loader.py constructing gpus number")
    """Train loader wrapper."""
    logger.info(f"number of gpus {cfg.NUM_GPUS}")

    if cfg.NUM_GPUS > 1:
        print("GPus are greater than 1")
        drop_last = True
    else:
        drop_last = False
    print("drop_last value ",drop_last)
    return _construct_loader(
        cfg=cfg,
        split="train",
        # batch_size=int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS),#currently we have 3 gpus but we didnt implement multiprocessing 
        batch_size=cfg.DATA.BATCH_SIZE,
        shuffle=True,
        drop_last=drop_last,
    )


def construct_trainval_loader(cfg):
    """Train loader wrapper."""
    if cfg.NUM_GPUS > 1:
        drop_last = True
    else:
        drop_last = False
    print("drop_last value ",drop_last)
    return _construct_loader(
        cfg=cfg,
        split="trainval",
        batch_size=int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=True,
        drop_last=drop_last,
    )


def construct_test_loader(cfg):
    """Test loader wrapper."""
    return _construct_loader(
        cfg=cfg,
        split="test",
        batch_size=int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS),
        shuffle=False,
        drop_last=False,
    )


def construct_val_loader(cfg, batch_size=None):
    if batch_size is None:
        bs = int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS)
    else:
        bs = batch_size
    """Validation loader wrapper."""
    return _construct_loader(
        cfg=cfg,
        split="val",
        batch_size=bs,
        shuffle=False,
        drop_last=False,
    )


def shuffle(loader, cur_epoch):
    """"Shuffles the data."""
    assert isinstance(
        loader.sampler, (RandomSampler, DistributedSampler)
    ), "Sampler type '{}' not supported".format(type(loader.sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(loader.sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        loader.sampler.set_epoch(cur_epoch)






#without dist


# #!/usr/bin/env python3
 
# """Data loader.""" 
# import torch
# from torch.utils.data.distributed import DistributedSampler
# from torch.utils.data.sampler import RandomSampler

# from ..utils import logging
# from .datasets.json_dataset import (
#     CUB200Dataset, CarsDataset, DogsDataset, FlowersDataset, NabirdsDataset
# )

# logger = logging.get_logger("visual_prompt")
# _DATASET_CATALOG = {
#     "CUB": CUB200Dataset,
#     'OxfordFlowers': FlowersDataset,
#     'StanfordCars': CarsDataset,
#     'StanfordDogs': DogsDataset,
#     "nabirds": NabirdsDataset,
# }


# def _construct_loader(cfg, split, batch_size, shuffle, drop_last):
#     """Constructs the data loader for the given dataset."""
#     dataset_name = cfg.DATA.NAME

#     # Construct the dataset
#     if dataset_name.startswith("vtab-"):
#         # import the tensorflow here only if needed
#         from .datasets.tf_dataset import TFDataset
#         dataset = TFDataset(cfg, split)
#     else:
#         assert (
#             dataset_name in _DATASET_CATALOG.keys()
#         ), "Dataset '{}' not supported".format(dataset_name)
#         dataset = _DATASET_CATALOG[dataset_name](cfg, split)
#     # Create a sampler for multi-process training
#     # print("before creating sampler ",sampler)
# #############################Change################
#     # sampler = DistributedSampler(dataset) if cfg.NUM_GPUS > 1 else None
#     sampler =  None
    
#     # Create a loader
#     print("creating sampler ",sampler)
#     loader = torch.utils.data.DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=(False if sampler else shuffle),
#         sampler=sampler,
#         num_workers=cfg.DATA.NUM_WORKERS,
#         pin_memory=cfg.DATA.PIN_MEMORY,
#         drop_last=drop_last,
#     )
#     return loader
 

# def construct_train_loader(cfg):
#     print("Inside loader.py constructing gpus number")
#     """Train loader wrapper."""
#     logger.info(f"number of gpus {cfg.NUM_GPUS}")

#     if cfg.NUM_GPUS > 1:
#         print("GPus are greater than 1")
#         drop_last = True
#     else:
#         drop_last = False
#     print("drop_last value ",drop_last)
#     return _construct_loader(
#         cfg=cfg,
#         split="train",
#         batch_size=int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS),#currently we have 3 gpus but we didnt implement multiprocessing 
#         # batch_size=cfg.DATA.BATCH_SIZE,
#         shuffle=True,
#         drop_last=drop_last,
#     )


# def construct_trainval_loader(cfg):
#     """Train loader wrapper."""
#     if cfg.NUM_GPUS > 1:
#         drop_last = True
#     else:
#         drop_last = False
#     print("drop_last value ",drop_last)
#     return _construct_loader(
#         cfg=cfg,
#         split="trainval",
#         batch_size=int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS),
#         shuffle=True,
#         drop_last=drop_last,
#     )


# def construct_test_loader(cfg):
#     """Test loader wrapper."""
#     return _construct_loader(
#         cfg=cfg,
#         split="test",
#         batch_size=int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS),
#         shuffle=False,
#         drop_last=False,
#     )


# def construct_val_loader(cfg, batch_size=None):
#     if batch_size is None:
#         bs = int(cfg.DATA.BATCH_SIZE / cfg.NUM_GPUS)
#     else:
#         bs = batch_size
#     """Validation loader wrapper."""
#     return _construct_loader(
#         cfg=cfg,
#         split="val",
#         batch_size=bs,
#         shuffle=False,
#         drop_last=False,
#     )


# def shuffle(loader, cur_epoch):
#     """"Shuffles the data."""
#     assert isinstance(
#         loader.sampler, (RandomSampler, DistributedSampler)
#     ), "Sampler type '{}' not supported".format(type(loader.sampler))
#     # RandomSampler handles shuffling automatically
#     if isinstance(loader.sampler, DistributedSampler):
#         # DistributedSampler shuffles data based on epoch
#         loader.sampler.set_epoch(cur_epoch)

