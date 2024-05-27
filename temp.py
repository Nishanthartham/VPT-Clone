# from fvcore.common.config import CfgNode
# import argparse
import torch
# def temp(cfg,args):
#     print(args.config_file)
#     print("&&&&&&&&&&")
#     print(cfg)
#     # print(cfg.RUN_N_TIMES)
    
# def default_argument_parser():
#     parser = argparse.ArgumentParser(description="visual-prompt")
#     parser.add_argument(
#         "--config-file", default="", metavar="FILE", help="path to config file")
#     parser.add_argument(
#         "opts",
#         help="Modify config options using the command-line",
#         default=None,
#         nargs=argparse.REMAINDER,
#     )
#     return parser


# if __name__ == '__main__':

#     args = default_argument_parser().parse_args()
#     cfg = CfgNode()
#     cfg.NUM_GPUS = 1
#     cfg.NUM_SHARDS = 1
#     cfg.DBG = False
#     cfg.OUTPUT_DIR = "./output"
#     cfg.RUN_N_TIMES = 5
# # Note that non-determinism may still be present due to non-deterministic
# # operator implementations in GPU operator libraries
#     cfg.SEED = None
#     cfg.merge_from_file(args.config_file)
#     temp(cfg,args)


a = torch.rand(196,1)
print(a)
#  a.values

b = a.expand(196,196)
print(b)