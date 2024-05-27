#!/usr/bin/env python3
"""
vit with prompt: a clean version with the default settings of VPT
""" 
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv

from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair
from torch.nn import Conv2d, Dropout,Linear
from scipy import ndimage

from ..vit_backbones.vit import CONFIGS, Transformer, VisionTransformer, np2th
from ...utils import logging

logger = logging.get_logger("visual_prompt")


class PromptedTransformer(Transformer):
    def __init__(self, prompt_config, config, img_size, vis):
        assert prompt_config.LOCATION == "prepend"
        assert prompt_config.INITIATION == "random"
        assert prompt_config.NUM_DEEP_LAYERS is None
        assert not prompt_config.DEEP_SHARED
        super(PromptedTransformer, self).__init__(
            config, img_size, vis)
        
        self.prompt_config = prompt_config #added newly not present in parent class
        self.vit_config = config
        
        img_size = _pair(img_size)
        n_patches = (img_size[0] // 16) * (img_size[1] // 16)
        patch_size = _pair(config.patches["size"])
        num_tokens = self.prompt_config.NUM_TOKENS
        self.num_tokens = num_tokens  # number of prompted tokens
        logger.info(f"Number of prompt tokens  is {num_tokens}")
        # logger.info(f"Patch size is {patch_size}")
        # logger.info(f"self.prompt_config.DROPOUT {self.prompt_config.DROPOUT}")
        # logger.info(f"Projection of prompt {self.prompt_config.PROJECT} output size {config.hidden_size}")
        self.prompt_dropout = Dropout(self.prompt_config.DROPOUT)

##########Config.hidden_size = 768 (16*16*3) each patch size**RGB
        # if project the prompt embeddings
        if self.prompt_config.PROJECT > -1:
            # only for prepend / add
            prompt_dim = self.prompt_config.PROJECT 
            self.prompt_proj = nn.Linear(
                prompt_dim, config.hidden_size)
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')#activation fun like relu
        else:
            prompt_dim = config.hidden_size #prompt_dim = -1 means no change req in prompt_dim
            self.prompt_proj = nn.Identity()

        # initiate prompt:
        if self.prompt_config.INITIATION == "random":
            #prompt_dim = 768 float(3 * reduce(mul, patch_size, 1) + prompt_dim) = 3(channels)*(16*16(patch_size))+768
            # logger.info(f"denominator of val is {3 * reduce(mul, patch_size, 1) + prompt_dim}")
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa
            # logger.info(f"val = {val}")

            self.prompt_embeddings = nn.Parameter(torch.zeros( #nn.parameter is applied to initiate that it is a learnable param
                1, num_tokens, prompt_dim)) #(1,100,768)
            # logger.info(f"prompt_embeddings dim before xavier_uniform initialization are (1,{num_tokens},{prompt_dim})")
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)
            # logger.info(f"initialised prompt embeddings size are {self.prompt_embeddings.data.size()}")#debugging
            if self.prompt_config.DEEP:  # noqa

                total_d_layer = config.transformer["num_layers"]-1
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                    total_d_layer, num_tokens, prompt_dim))
                # xavier_uniform initialization
                nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

        else:
            raise ValueError("Other initiation scheme is not supported")

    def incorporate_prompt(self, x):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0]

        # after CLS token, all before image patches
        x = self.embeddings(x)  # (batch_size, 1 + n_patches, hidden_dim)
        # logger.info(f"size of patches without prompt = {x.size()}") # torch.Size([19, 197, 768])
        x = torch.cat((
                x[:, :1, :],#cls
                self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),#prompts
                x[:, 1:, :]#patches
            ), dim=1)
        # logger.info(f"Final embedding size {x.size()}") #(batch_size,cls+patches+Num_Tokens,hidden_dim) = (21,1+196+100,768)
        # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)

        return x

    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training:
            self.encoder.eval()
            self.embeddings.eval()
            self.prompt_proj.train()
            self.prompt_dropout.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    def forward_deep_prompt(self, embedding_output):
        attn_weights = []
        hidden_states = None
        weights = None
        B = embedding_output.shape[0]
        num_layers = self.vit_config.transformer["num_layers"]

        for i in range(num_layers):
            if i == 0:
                hidden_states, weights = self.encoder.layer[i](embedding_output)
            else:
                if i <= self.deep_prompt_embeddings.shape[0]:#only for deep
                    deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                        self.deep_prompt_embeddings[i-1]).expand(B, -1, -1))

                    hidden_states = torch.cat((
                        hidden_states[:, :1, :],
                        deep_prompt_emb,
                        hidden_states[:, (1+self.num_tokens):, :]
                    ), dim=1)


                hidden_states, weights = self.encoder.layer[i](hidden_states)

            if self.encoder.vis:
                attn_weights.append(weights)

        encoded = self.encoder.encoder_norm(hidden_states)
        return encoded, attn_weights

    def forward(self, x):
        # this is the default version:
        embedding_output = self.incorporate_prompt(x)
        if self.prompt_config.DEEP:
            encoded, attn_weights = self.forward_deep_prompt(
                embedding_output)
        else:
            encoded, attn_weights = self.encoder(embedding_output)

        return encoded, attn_weights
 

class PromptedVisionTransformer(VisionTransformer):
    def __init__(
        self, prompt_cfg, model_type,
        img_size=224, num_classes=21843, vis=False):
        assert prompt_cfg.VIT_POOL_TYPE == "original"
        super(PromptedVisionTransformer, self).__init__(model_type, img_size, num_classes, vis)
        if prompt_cfg is None:
            raise ValueError("prompt_cfg cannot be None if using PromptedVisionTransformer")
        self.prompt_cfg = prompt_cfg
        vit_cfg = CONFIGS[model_type]
        self.transformer = PromptedTransformer(
            prompt_cfg, vit_cfg, img_size, vis)
        img_size = _pair(img_size)
        n_patches = (img_size[0] // 16) * (img_size[1] // 16)#hardcoded
        self.attention_weights = nn.Parameter(torch.zeros( #nn.parameter is applied to initiate that it is a learnable param
                1, vit_cfg.hidden_size, n_patches+prompt_cfg.NUM_TOKENS,device='cuda'))
        # self.attention_weights = nn.Parameter(torch.zeros( #nn.parameter is applied to initiate that it is a learnable param
        #         1, vit_cfg.hidden_size, n_patches,device='cuda'))
        logger.info(f"Num classes vit.py ={num_classes}")#######make it dynamic
        logger.info(f"include_attention_token = {self.prompt_cfg.INCLUDE_ATTENTION_TOKEN}")
        self.prompt_head = Linear(vit_cfg.hidden_size,200)
    def initialize_attention_weights(self):
        # logger.info(f"initalise self.attention_weights = {self.attention_weights.data.shape}")

        patch_size = _pair(16)#hard code 
        prompt_dim = 768

        # combine prompt embeddings with image-patch embeddings
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

         #only patch tokens no prompts
        # logger.info(f"prompt_embeddings dim before xavier_uniform initialization are (1,{num_tokens},{prompt_dim})")
        # xavier_uniform initialization
        nn.init.uniform_(self.attention_weights.data, -val, val)
        # return x  
    def compute_embedding_attention(self,embeddings):
        self.initialize_attention_weights()
        raw_cls = embeddings[:,0]#cls [21,768]
        raw_cls = raw_cls.unsqueeze(1)
        # if (self.prompt_cfg.DBG):
        # logger.info(f"mine self.attention_weights = {self.attention_weights.data.shape}")
        # logger.info(f"clss = {raw_cls.data.shape}")
        # logger.info(f"clss = {raw_cls.data.shape}")
        self.attention_embeddings = torch.matmul(raw_cls,self.attention_weights)#(b_s,1,196)
        # self.attention_embeddings = torch.zeros(raw_cls.data.shape[0],1,196,device=torch.device('cuda'))
        # Second step perform E(att)
        #embeddings size = ([21, 297, 768]) leave top 101 columns 
        # #new changes
        # self.attention_embeddings = torch.transpose(self.attention_embeddings,1,2)
        # logger.info(f"self.attention_embeddings = {self.attention_embeddings.data.size()}")
        # self.attention_embeddings = self.attention_embeddings.expand(raw_cls[2],self.attention_embeddings[1])
        # logger.info(f"self.attention_embeddings = {self.attention_embeddings.data.size()}")
        # final_attention_embeddings = torch.matmul(self.attention_embeddings,embeddings[:,101:]) Excluding prompts
        # final_attention_embeddings = torch.matmul(self.attention_embeddings,embeddings[:,1:])#Excluding only cls
        final_attention_embeddings = torch.matmul(self.attention_embeddings,embeddings[:,1:])#Excluding only cls
        # logger.info(f"final_attention_embeddings = {final_attention_embeddings.data.size()}")
        return final_attention_embeddings
          
    def forward(self, x, vis=False):
        x, attn_weights = self.transformer(x)
        x_cls = x[:, 0]#######################Cls extraction ##########################
#####################################################################################################
        # logger.info(f"self.head from prompt = {self.head}")
        if (self.prompt_cfg.INCLUDE_ATTENTION_TOKEN):
            final_attention_embeddings = self.compute_embedding_attention(x)    
            final_attention_embeddings = final_attention_embeddings.squeeze()#

            # final_attention_embeddings = torch.zeros(768)

            outputs = torch.cat((x_cls,final_attention_embeddings),dim=1)
            # logits = self.cls_head(outputs)
            logits = self.head(outputs)
        else:
            logits = self.head(x_cls)
        logger.debug(f"The patch attention embeddings are {self.attention_weights.data[0,0:6,0:6]}")  
        # logger.info(f"x = {x.shape} and att_emb = {final_attention_embeddings.shape}")#x = torch.Size([21, 768]) and att_emb = torch.Size([21, 768])
        # logger.info(f"output dim  is {outputs.shape}")
        #output dim  is torch.Size([21, 1536])
        #x are the embeddings its size is = torch.Size([21, 297, 768])
        # logger.info(f"I guess cls extraction from vit_prompt {x} and {x.shape}")
        #self.head = Linear(config.hidden_size, num_classes) if num_classes > 0 else nn.Identity()
        # logits = self.head(outputs)
        # patch_attention_logits = self.head(final_attention_embeddings)
        # patch_attention_logits = patch_attention_logits.squeeze()
        # logger.info(f"logits value probabilites ig {logits} and {logits.shape}")
        # logger.info(f"logits value patch_attention_logits {patch_attention_logits} and {patch_attention_logits.shape}")
        if not vis:
            return logits
        # return patch_attention_logits, logits, attn_weights ## send final_attention_embeddings
        return  logits, attn_weights ## send final_attention_embeddings
