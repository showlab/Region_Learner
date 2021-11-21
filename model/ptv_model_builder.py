import torch.nn as nn
from pytorchvideo.models.vision_transformers import create_multiscale_vision_transformers




def get_head_act(act_func):
    """
    Return the actual head activation function given the activation fucntion name.
    Args:
        act_func (string): activation function to use. 'softmax': applies
        softmax on the output. 'sigmoid': applies sigmoid on the output.
    Returns:
        nn.Module: the activation layer.
    """
    if act_func == "softmax":
        return nn.Softmax(dim=1)
    elif act_func == "sigmoid":
        return nn.Sigmoid()
    else:
        raise NotImplementedError(
            "{} is not supported as a head activation "
            "function.".format(act_func)
        )


        
# @MODEL_REGISTRY.register()
class PTVMViT(nn.Module):
    """
    MViT models using PyTorchVideo model builder.
    """

    def __init__(self, cls_embed_on=True, seq_pool_type='cls', temporal_size=4):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(PTVMViT, self).__init__()

        # assert (
        #     cfg.DETECTION.ENABLE is False
        # ), "Detection model is not supported for PTVMViT yet."
        self.cls_embed_on = cls_embed_on
        self.seq_pool_type = seq_pool_type # ["cls", "meam", "none"]
        self.temporal_size = temporal_size
        self._construct_network()

    def _construct_network(self):
        """
        Builds a MViT model.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """

        spatial_size = 224
        temporal_size = self.temporal_size
        embed_dim_mul = [[1, 2.0], [3, 2.0], [14, 2.0]]
        atten_head_mul = [[1, 2.0], [3, 2.0], [14, 2.0]]
        pool_q_stride_size = [[1, 1, 2, 2], [3, 1, 2, 2], [14, 1, 2, 2]]
        pool_kv_stride_adaptive = [1, 8, 8]
        pool_kvq_kernel = [3, 3, 3]
        head_num_classes = 400
        MViT_B = create_multiscale_vision_transformers(
            spatial_size=spatial_size,
            temporal_size=temporal_size,
            embed_dim_mul=embed_dim_mul,
            atten_head_mul=atten_head_mul,
            pool_q_stride_size=pool_q_stride_size,
            pool_kv_stride_adaptive=pool_kv_stride_adaptive,
            pool_kvq_kernel=pool_kvq_kernel,
            head_num_classes=head_num_classes,
        )
        self.model=MViT_B

        # self.model = create_multiscale_vision_transformers(
        #     spatial_size=cfg.DATA.TRAIN_CROP_SIZE,
        #     temporal_size=cfg.DATA.NUM_FRAMES,
        #     # cls_embed_on=cfg.MVIT.CLS_EMBED_ON,
        #     cls_embed_on=self.cls_embed_on,
        #     # seq_pool_type=self.seq_pool_type,
        #     sep_pos_embed=cfg.MVIT.SEP_POS_EMBED,
        #     depth=cfg.MVIT.DEPTH,
        #     norm=cfg.MVIT.NORM,
        #     # Patch embed config.
        #     input_channels = cfg.DATA.INPUT_CHANNEL_NUM[0],
        #     patch_embed_dim = cfg.MVIT.EMBED_DIM,
        #     conv_patch_embed_kernel = (3, 7, 7), #cfg.MVIT.PATCH_KERNEL,
        #     conv_patch_embed_stride = (2, 4, 4), #cfg.MVIT.PATCH_STRIDE,
        #     conv_patch_embed_padding = (1, 3, 3), #cfg.MVIT.PATCH_PADDING,
        #     # enable_patch_embed_norm = cfg.MVIT.NORM_STEM,
        #     # use_2d_patch=cfg.MVIT.PATCH_2D,
        #     # Attention block config.
        #     num_heads = cfg.MVIT.NUM_HEADS,
        #     mlp_ratio = cfg.MVIT.MLP_RATIO,
        #     qkv_bias = cfg.MVIT.QKV_BIAS,
        #     # dropout_rate_block = cfg.MVIT.DROPOUT_RATE,
        #     droppath_rate_block = cfg.MVIT.DROPPATH_RATE,
        #     pooling_mode = cfg.MVIT.MODE,
        #     # pool_first = cfg.MVIT.POOL_FIRST,
        #     embed_dim_mul = cfg.MVIT.DIM_MUL,
        #     atten_head_mul = cfg.MVIT.HEAD_MUL,
        #     pool_q_stride_size = cfg.MVIT.POOL_Q_STRIDE,
        #     # pool_kv_stride_size = cfg.MVIT.POOL_KV_STRIDE,
        #     pool_kv_stride_adaptive = cfg.MVIT.POOL_KV_STRIDE_ADAPTIVE,
        #     # pool_kvq_kernel = cfg.MVIT.POOL_KVQ_KERNEL,
        #     # Head config.
        #     head_dropout_rate = cfg.MODEL.DROPOUT_RATE,
        #     head_num_classes = cfg.MODEL.NUM_CLASSES,
        # )

        # self.post_act = get_head_act(cfg.MODEL.HEAD_ACT)


    def forward(self, x, bboxes=None):
        B, C, T, H, W = x.size()
        T_o, H_o, W_o = T//2, H//32, W//32
        # x = x[0]
        x = self.model(x) # [B, thw+1, C]

        # if not self.training:
            # x = self.post_act(x)

        if self.seq_pool_type=='cls':
            # print('cls')
            out = x[:, 1, :]
            return out, None
        elif self.seq_pool_type=='mean':
            # print('mean')
            return torch.mean(x[:, 1:, :], 1, ), None
        elif self.seq_pool_type=='none':
            # print('none')
            # reshape
            return x[:, 1:, :].reshape(B, T_o, H_o, W_o, -1), None

        


# class DictAsMember(dict):
#     def __getattr__(self, name):
#         value = self[name]
#         if isinstance(value, dict):
#             value = DictAsMember(value)
#         return value

import yaml
import torch
import time

def load_part_state(model, pretrained_dict):
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict) and 'pos_embed_temporal' not in k}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    return model_dict
    
    

def build_MViT_32(cls_embed_on=True, seq_pool_type='cls', temporal_size=4):
    # with open("cfg/MVIT_B_32x3_CONV.yaml", "r") as stream:
    #     try:
    #         cfg = yaml.safe_load(stream)
    #         cfg = DictAsMember(cfg)
    #         # print(cfg)
    #     except yaml.YAMLError as exc:
    #         print(exc)

    net = PTVMViT(seq_pool_type=seq_pool_type, temporal_size=temporal_size).cuda() # try by Mr. Yan
    model_pth = 'model/weights/MVIT_B_32x3_f294077834.pyth'
    state = torch.load(model_pth)
    # print(state['model_state'].keys())
    # net.model.load_state_dict(state['model_state'], strict=False)

    pretrained_dict = load_part_state(net.model, state['model_state'])
    # 3. load the new state dict
    net.model.load_state_dict(pretrained_dict)



    net.model.head = nn.Identity()
    return net

if __name__ == "__main__":
    B, T, H, W = 32, 4, 224, 224
    net = build_MViT_32(seq_pool_type='mean', temporal_size=T)
    print(net)
    # net.pre_logits = nn.Identity()
    
    inputs = torch.rand(B, 3, T, H, W).cuda() # b, 
    # T_o, H_o, W_o = T//2, H//32, W//32
    print('inputs:\t', inputs.size())
    since = time.time()
    # for i in range(1):
    outputs = net(inputs)
    print('MViT takes %f'%(time.time()-since))
    print('outputs:\t', outputs.size())
    # outputs = outputs.reshape(B, T_o, H_o, W_o, -1)
    # print('reshape outputs:\t', outputs.size())