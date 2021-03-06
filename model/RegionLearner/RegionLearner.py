import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from model.RegionLearner.Quantizer import VectorQuantizer
from model.helper import Attention

class Aggregation(nn.Module):
    """
    The second step of RegionLearner.
    It is designed to aggregate quantized tokens into several semantic regions
    """
    def __init__(self, token_dim, num_region=8):
        super(Aggregation, self).__init__()
        self.num_region = num_region
        self.token_dim= token_dim
        self.spatial_pooling = nn.AdaptiveAvgPool2d((1,1))
        self.spatial_att = nn.Conv2d(in_channels=token_dim,
                        out_channels=self.num_region, # each channel used as att map for capturing one region
                        kernel_size=3,
                        padding=1
                        )

        print("Performing Aggregation to learn %d regions ...." %(num_region))

    def forward(self, x):
        # x [B, C, H, W]
        B = x.size(0)
        region_mask = self.spatial_att(x) # [B, S, H, W]
        learned_region_list = []
        for s in range(self.num_region):
            # print(x.size(), att_mask[:,s,...].unsqueeze(1).size())
            learned_region = x * region_mask[:,s,...].unsqueeze(1) # [B, C, H, W] * [B, 1, H, W] --> [B, C, H, W]
            learned_region_list.append(self.spatial_pooling(learned_region).reshape(B, self.token_dim)) # [B, C, H, W] --> [B, C, 1, 1]

        #  learned_region_list [B, C, 1]
        # print(learned_region_list[0].size())
        learned_regions = torch.stack(learned_region_list, dim=-1) # [B, C, S]
        return learned_regions, region_mask # [B, C, S]


class Motion_Excitation(nn.Module):
    """
    It is designed to xxx
    """
    def __init__(self, token_dim):
        super(Motion_Excitation, self).__init__()
        self.ex_proj = nn.Conv2d(in_channels=token_dim,
                        out_channels=1, # each channel used as att map for capturing one region
                        kernel_size=3,
                        padding=1
                        )
        print("Performing Motion Excitation ...")

    def ex(self, x, h, w):
        # X: [B, T, L, C]
        _, T, L, C = x.size()
        x = x.reshape(-1, L, C) #[B*T, L, C]
        x = x.transpose(1, 2) #[B*T, C, L]
        x = x.reshape(-1, C, h, w)
        motion_map = self.ex_proj(x) # [B*T, 1, H, W]
        motion_map = motion_map.reshape(-1, T, L)
        return motion_map

    def forward(self, x, h, w):
        # X: [B, T, L, C]
        # shift X:
        T = x.size(1)
        x_ = x.clone()
        x_[:,1:,...] = x[:,:T-1,...]
        motion_map = self.ex(x - x_, h, w) # [B, T, L]
        motion_map = F.softmax(motion_map, dim=-1) # [B, T, L]
        # motion_map = F.softmax(fc(fc(x_) - x), dim=-1)
        # print('motion_map:', x.size(), motion_map.size())
        motaion_feas = x * motion_map.unsqueeze(-1) # [B, T, L, C]
        return motaion_feas


class RegionLearner(nn.Module):
    """
    Learning implicit regions without supervision from video feature map.
    """
    def __init__(self, VQ_num_tokens=None, VQ_token_dim=768, AGG_region_num=None, Interaction_depth=None, ME=None, dist=False):
        super(RegionLearner, self).__init__()
        self.Quantization = None
        self.Aggregation = None
        self.Interaction = None
        self.Motion_Excitation = None
        

        if VQ_num_tokens and VQ_token_dim:
            self.Quantization = VectorQuantizer(VQ_num_tokens, VQ_token_dim, dist=dist)
            if ME:
                self.Motion_Excitation = Motion_Excitation(VQ_token_dim)
            

        if AGG_region_num:
            self.Aggregation = Aggregation(token_dim=VQ_token_dim, num_region=AGG_region_num)
            if Interaction_depth:
                print("Performing Interaction among regions with %d layers ...." %(Interaction_depth))
                if Interaction_depth>1:
                    self.Interaction = nn.ModuleList([Attention(VQ_token_dim)
                        for i in range(Interaction_depth)])
                else:
                    # TODO Delete it, now is kept for old models.
                    self.Interaction = Attention(VQ_token_dim)

    def forward(self, in_feas, cur_f=1, epoch=0):
        if self.Quantization:
            B, L, C = in_feas.size()
            vd_inputs = in_feas.reshape(-1, C) # [BL, C]
            vd_outputs, encoding_indices = self.Quantization(vd_inputs)  # [BL, C], [BL, 1]
            h = int(math.sqrt(L))
            w = int(L//h)
            encoding_indices = encoding_indices.reshape(-1, h, w)

            # Perform Motion Excitation
            if cur_f>1 and self.Motion_Excitation:
                in_feas = in_feas.view(-1, cur_f, L, C)
                motion_feas = self.Motion_Excitation(in_feas, h, w) # [B, T, L, C]
                motion_feas = motion_feas.reshape(-1, C)
                # print('size:', motion_feas.size(), vd_outputs.size())
                vd_outputs = vd_outputs + motion_feas
                # vd_outputs = torch.cat([vd_outputs,motion_feas], dim=-1)

            

        
        if self.Aggregation:
            #  [B*L, C]
            vd_outputs = vd_outputs.reshape(B, L, C)
            vd_outputs = vd_outputs.transpose(1, 2) #[B, C, L]
            vd_outputs = vd_outputs.reshape(B, C, h, w)
            #  [B, C, H, W]
            #  print('learn regions, x size:\t', x.size())
            vd_outputs, region_mask = self.Aggregation(vd_outputs) # [B, C, S]
            vd_outputs = vd_outputs.transpose(1, 2) # [B, S, C]
            if self.Interaction:
                # print('do joint att')
                _, S, C = vd_outputs.size()
                # print('vd_outputs:\t', vd_outputs.size())
                T = cur_f
                # TODO we can do spatial-temporal on regions
                vd_outputs = vd_outputs.reshape(-1, T, S, C)
                vd_outputs = vd_outputs.reshape(-1, T*S, C)
                #  [b, T*S, C]
                # print('att inputs:\t', vd_outputs.size())
                vd_outputs = self.Interaction(vd_outputs)
                # print('att outputs:\t', vd_outputs.size())
                vd_outputs = vd_outputs.reshape(-1, T, S, C)
                vd_outputs = vd_outputs.reshape(-1, S, C) # [B, S, C]
                
            return vd_outputs, encoding_indices, region_mask
        else:
            vd_outputs = vd_outputs.reshape(B, L, C)
            return vd_outputs, encoding_indices, None





if __name__ == "__main__":
    B, T, L = 2, 2, 196
    token_dim = 768
    num_tokens = 2048
    RL = RegionLearner(VQ_num_tokens=num_tokens, VQ_token_dim=token_dim, AGG_region_num=8, Interaction_depth=1, dist=False, ME=True)
    inputs = torch.randn(B*T, L, token_dim)
    print('Input of RegionLearner:\t', inputs.size())
    outputs, encoding_indices, region_mask = RL(inputs, T)
    print('Output of RegionLearner:\t', outputs.size())
    print('Encoding Indices of Quantization:\t', encoding_indices.size())
    if region_mask is not None:
        print('Region Mask of Aggregation:\t', region_mask.size())
    
