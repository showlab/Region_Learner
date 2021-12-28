"""
This script defines some useful function for model building.
"""

import numpy as np
import torch
from torch import nn
from einops import rearrange, repeat
import os
import json
from datetime import datetime

def get_random_sample_indices(
        seq_len, sample_ratio=0.9, num_samples=100, device=torch.device("cpu")):
    """
    Args:
        seq_len: int, the sampled indices will be in the range [0, seq_len-1]
        num_samples: sample size
        device: torch.device
    Returns:
        1D torch.LongTensor consisting of sorted sample indices
        (sort should not affect the results as we use transformers)
    """
    if sample_ratio:
        num_samples = int(seq_len*sample_ratio)
        
    if num_samples >= seq_len:
        # return all indices
        sample_indices = np.arange(seq_len)
    else:
        sample_indices = np.random.choice(
            seq_len, size=num_samples, replace=False)
        sample_indices = np.sort(sample_indices)
    return torch.from_numpy(sample_indices).long().to(device)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        if self.with_qkv:
           qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
           q, k, v = qkv[0], qkv[1], qkv[2]
        else:
           qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
           q, k, v  = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
           x = self.proj(x)
           x = self.proj_drop(x)
        return x




############ Following functions are used to visualize results from RegionLearner #####################

def file_check(file_name):
    path = os.path.dirname(file_name)
    if not os.path.exists(path):
        os.makedirs(path)

def json_write(data, file_name):
    try:
        file_check(file_name)
        with open(file_name, 'w+') as outfile:
            json.dump(data, outfile)
        print('json file saved at %s'%(file_name))
    except:
        import traceback
        traceback.print_exc()
        print('cannot write %s'%(file_name))
        
def save_vis_re(data, vis_re, save_pth=None, timestamp=True):
    # meta_arr = {'raw_captions': caption, 'paths': rel_fp, 'dataset': self.dataset_name}
    # data = {'video': final, 'text': caption, 'meta': meta_arr, 'frame_idxs': idxs}
    # print('save_vis_re:\t', data.keys())
    indices, region_mask = vis_re
    vids = data['meta']['paths']
    raw_caps= data['meta']['raw_captions']
    # TODO support multiple frames
    frame_idxs= data['frame_idxs'][0].tolist()

    re = {}
    timestamp = datetime.now().strftime(r'%m%d_%H%M%S') if timestamp else ''
    print(len(vids), indices.size(), region_mask.size())
    for i in range(len(vids)):
        k = str(vids[i])
        v = indices[i].cpu().detach().tolist()
        v1 = region_mask[i].cpu().detach().tolist()
        # print(k,v)
        re[k] = {'cluster_id':v, 'region_mask':v1, 'raw_caption':raw_caps[i], 'frame_idxs':frame_idxs[i]}
        # re[k] = {'cluster_id':v, 'region_mask':v1, 'raw_caption':raw_caps[i]}
        
    # print(re)
    save_pth = os.path.join(save_pth, timestamp)
    try:
        json_write(re, '%s/vis.json'%(save_pth))
    except:
        print("failed to save results!!!")
        import traceback
        traceback.print_exc()