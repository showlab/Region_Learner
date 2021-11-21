import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

try:
    from model.Visual_Dict import SOHO_Pre_VD
    from model.Region_Learner import RegionLearner
    from model.Tube_Builder import Tube_Builder
except:
    from Visual_Dict import SOHO_Pre_VD
    from Region_Learner import RegionLearner
    from Tube_Builder import Tube_Builder
# from model.Visual_Dict import SOHO_Pre_VD
# from model.Region_Learner import RegionLearner
# from model.Tube_Builder import Tube_Builder

def selection(x):
    """
        select values that appears with high frequence (more than 'avg_cnt' times).
    """
    cnt=np.bincount(x)
    avg_cnt = np.sum(cnt)/len(cnt)
#     print('keep_th(if cnt is larger than thia value, will be keep):', keep_th)
    keep_values = np.where(cnt>(avg_cnt))
    mask = np.isin(x, keep_values)
    return mask


class SoftRegion(nn.Module):
    """
    Our new model that aims at finding the soft region with supervision from video feature map.
    """
    def __init__(self, num_frames, num_tokens, token_dim, att=False, not_plus=False, dist=False, selecting=False, selecting_start=-1, learn_region=False, build_tube=False):
        super(SoftRegion, self).__init__()
        print("Using Clustering Moudle...")
        # create a visual dictionary to learn a limited latent embeddings (N=2048 or others) to abstract/cluster meta-semantics?
        # It works like clustering
        # self.visual_dict = SOHO_Pre_VD(num_tokens=2048, token_dim=token_dim, dist=True)
        self.visual_dict = SOHO_Pre_VD(num_tokens, token_dim, dist=dist)
        self.att = att
        self.not_plus = not_plus
        self.selecting = selecting
        self.selecting_start = selecting_start
        self.learn_region = learn_region
        self.learn_region_num = 8
        self.build_tube = build_tube
        
        self.num_frames = num_frames

        if self.build_tube:
            assert num_frames>1
            self.tube_builder = Tube_Builder(token_dim, temp_stride=2)

        assert (self.att and self.selecting) == False

        # print('selecting:', self.selecting)

        if self.att:
            self.atttion_layer = nn.Sequential(
                nn.Conv1d(in_channels=token_dim,
                        out_channels=1,
                        kernel_size=3,
                        padding=1,
                        bias=False), nn.ReLU())

        if self.learn_region:
            # pass
            self.region_learner = RegionLearner(token_dim=token_dim, num_region=self.learn_region_num)


    def forward(self, in_feas, epoch=0):
        B, L, C = in_feas.size()
        print('in_feas size:\t', in_feas.size())
#         exit(0)
        if self.num_frames>1 and self.build_tube:
            in_feas = in_feas.reshape(-1, self.num_frames, L, C)
            # print('in_feas size:\t', in_feas.size())
            in_feas = self.tube_builder(in_feas) # [B, T//2, L, C]
            in_feas = in_feas.reshape(-1, L, C)
            # print('in_feas size:\t', in_feas.size())
            B, L, C = in_feas.size()
                

        # print('in_feas:\t', in_feas.size())
        vd_inputs = in_feas.reshape(-1, C) # [BL, C]
        vd_outputs, encoding_indices = self.visual_dict(vd_inputs)  # [BL, C], [BL, 1]
        h = int(math.sqrt(L))
        w = int(L//h)
        # print('encoding_indices:\t', encoding_indices.size())
        # print('h, w:\t', h, w)

        encoding_indices = encoding_indices.reshape(-1, h, w)

        

        # print('epoch', epoch)
        if self.att:
            # build soft regions by attention
            # it is hard for us to get dynamic regions via the clustering result (encoding_indices), it is time-comsuming
            # compute a attention mask with the guide of vd_outputs (in which similar semantics are represented by same "latent embeddings(called words?)")
            # thus, attention output same weights given same inputs
            vd_outputs = vd_outputs.reshape(-1, C, 1)
            # print('ATT vd_outputs', vd_outputs.size())
            att_mask = self.atttion_layer(vd_outputs)  # [BL, C, 1] -> [BL, 1, 1]
            # print('att_mask:\t', att_mask.size())
            att_mask = F.softmax(att_mask.reshape(-1, L, 1), 1) # [B, L, 1]
            # compared with classic attention mask, our mask will cluster
            # print('in_feas:\t', in_feas.size(), 'att_mask:\t', att_mask.size())
            # print(att_mask)
            

            if self.not_plus:
                out_feas = in_feas * att_mask
            else:
                vd_outputs = vd_outputs.reshape(-1, L, C)
                out_feas = vd_outputs + in_feas * att_mask # [B, L, C]

            return out_feas, encoding_indices
        elif self.selecting and self.training and epoch>self.selecting_start:
            # print('in epoch :', epoch)
            # data = torch.randint(num_tokens, (batch_size*num_frames, seq_size)).cuda()
            # since = time.time()
            mask = torch.ones_like(encoding_indices).detach().cpu().numpy()
            data = encoding_indices.detach().cpu().numpy()
            data = data.reshape(B, -1) # [B, h, w] --> [B, L]
            mask = mask.reshape(B, -1)
            for i, item in enumerate(data):
                mask[i] = selection(item)

            vd_outputs = vd_outputs.reshape(B, L, C)
            mask = torch.from_numpy(mask).reshape(B, L, 1).cuda()
            vd_outputs = vd_outputs*mask # [B, L, C]
            # print(mask)
            return vd_outputs, encoding_indices
        elif self.learn_region:
            #  [B*L, C]
            vd_outputs = vd_outputs.reshape(B, L, C)
            vd_outputs = vd_outputs.transpose(1, 2) #[B, C, L]
            vd_outputs = vd_outputs.reshape(B, C, h, w)
            #  [B, C, H, W]
            #  print('learn regions, x size:\t', x.size())
            vd_outputs = self.region_learner(vd_outputs) # [B, C, S]
            vd_outputs = vd_outputs.transpose(1, 2) # [B, S, C]
            return vd_outputs, encoding_indices
        else:
            vd_outputs = vd_outputs.reshape(B, L, C)
            return vd_outputs, encoding_indices




if __name__ == "__main__":
    from Visual_Dict import SOHO_Pre_VD
    B, T, L = 2, 4, 196
    token_dim = 768
    num_tokens = 2048
    SR = SoftRegion(T, num_tokens=num_tokens, token_dim=token_dim, dist=False, build_tube=True)
    inputs = torch.randn(B*T, L, token_dim)
    print('inputs:\t', inputs.size())
    outputs, encoding_indices = SR(inputs)
    print('outputs:\t', outputs.size())
    # print(outputs)
    print('encoding_indices:\t', encoding_indices.size())