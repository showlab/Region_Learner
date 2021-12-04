import torch
import torch.nn as nn


class RegionLearner(nn.Module):
    def __init__(self, token_dim, num_region=8):
        super(RegionLearner, self).__init__()

        self.num_region = num_region
        self.token_dim= token_dim
        print('Region Learner learning ', num_region, ' regions...')

        self.spatial_pooling = nn.AdaptiveAvgPool2d((1,1))
        self.spatial_att = nn.Conv2d(in_channels=token_dim,
                        out_channels=self.num_region, # each channel used as att map for capturing one region
                        kernel_size=3,
                        padding=1
                        )

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

if __name__ == "__main__":
    import time
    B, H, W = 64, 14, 14
    token_dim = 768
    num_region = 8
    RL = RegionLearner(token_dim=token_dim, num_region=num_region).cuda()

    since = time.time()
    for i in range(100):
        inputs = torch.randn(B, token_dim, H, W).cuda()
        # print('inputs:\t', inputs.size())
        outputs = RL(inputs)
        # print('outputs:\t', outputs.size())
    print('time:\t', time.time()-since)