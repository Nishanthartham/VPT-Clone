#!/usr/bin/env python3
"""
Model construction functions.
"""
from tabnanny import verbose
import torch

from .resnet import ResNet
from .convnext import ConvNeXt
from .vit_models import ViT, Swin, SSLViT
from ..utils import logging
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.distributed as dist
from ..utils.distributed import get_rank, get_world_size
# from detectron2.engine.launch import launch


logger = logging.get_logger("visual_prompt")
# Supported model types
_MODEL_TYPES = {
    "resnet": ResNet,
    "convnext": ConvNeXt,
    "vit": ViT,
    "swin": Swin,
    "ssl-vit": SSLViT,
}


def build_model(cfg): 
    """
    build model here
    """
    assert (
        cfg.MODEL.TYPE in _MODEL_TYPES.keys()
    ), "Model type '{}' not supported".format(cfg.MODEL.TYPE)
    assert (
        cfg.NUM_GPUS <= torch.cuda.device_count()
    ), "Cannot use more GPU devices than available"

    # Construct the model
    train_type = cfg.MODEL.TYPE
    model = _MODEL_TYPES[train_type](cfg)

    log_model_info(model, verbose=cfg.DBG)
    model, device = load_model_to_device(model, cfg)
    logger.info(f"Device used for model: {device}")
    return model, device


def log_model_info(model, verbose=False):
    """Logs model info"""
    if verbose:
        logger.info(f"Classification Model:\n{model}")
    model_total_params = sum(p.numel() for p in model.parameters())
    model_grad_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Total Parameters: {0}\t Gradient Parameters: {1}".format(
        model_total_params, model_grad_params))
    logger.info("tuned percent:%.3f"%(model_grad_params/model_total_params*100))


def get_current_device():
    if torch.cuda.is_available():
        # Determine the GPU used by the current process
        cur_device = torch.cuda.current_device()
    else:
        cur_device = torch.device('cpu')
    return cur_device

def main_func(process_index,model,cfg):
    # torch.cuda.set_device(device_id)
    # cur_device = get_current_device()
            
    dist.init_process_group(backend='nccl',rank=get_rank(),world_size=get_world_size() ,init_method="tcp://{}:12400".format("xulab-gpu1.pc.cc.cmu.edu"))
    device_id = process_index
    model = model.cuda(device=device_id)
    torch.cuda.set_device(device_id)

                # Make model replica operate on the current device
    model = torch.nn.parallel.DistributedDataParallel(
                module=model, device_ids=[device_id], output_device=device_id,
                find_unused_parameters=True,
            )
def load_model_to_device(model, cfg):
    cur_device = get_current_device()
    print("Curr device ",cur_device)
    if torch.cuda.is_available():
        model = model.cuda(device=cur_device)
        ## Use multi-process data parallel model in the multi-gpu setting
        # if cfg.NUM_GPUS > 1:
        #     # def main_func(process_index, cfg):
        #     #     device_id = process_index
        #     #     model = model.cuda(device=process_index)

        #     #     # Make model replica operate on the current device

        #     #     model = torch.nn.parallel.DistributedDataParallel(
        #     #     module=model, device_ids=[device_id], output_device=device_id,
        #     #     find_unused_parameters=True,
        #     # ) 
        #     mp.spawn(
        #         main_func,
        #         args=(model,cfg),
        #         nprocs=cfg.NUM_GPUS,
        #         join=True
        #     )
        #     # launch(
        #     #     main_func,
        #     #     num_gpus_per_machine=cfg.NUM_GPUS,
        #     #     num_machines=1,  # Adjust as needed
        #     #     machine_rank=0,  # Adjust as needed
        #     #     # dist_url="auto",
        #     #     args=(cfg),  # Pass any additional arguments needed for training
        #     # ) 
        # logger.info("done")  
    else:
        model = model.to(cur_device)
    return model, cur_device
