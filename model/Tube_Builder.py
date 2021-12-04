import torch
from torch import nn
import time

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Tube_Builder(nn.Module):
    def __init__(self, in_features, temp_stride=2):
        super(Tube_Builder, self).__init__()
        C = in_features
        self.temp_stride = temp_stride
        self.MLP = Mlp(in_features=temp_stride*C, hidden_features=(temp_stride//2)*C, out_features=C)
        # print('Tube is built!')
        
    def forward(self, X):
        """
        Given a feature map, this function will build temporal tubes for them.
        X: [B, T, L, C], L=HW
        """
    #     [B, T//2, L, C] [B, T//2, L, C]
        tube_feas = torch.cat((X[:,::self.temp_stride,...], X[:,1::self.temp_stride,...]), dim=-1) # X:[B, T, L, C] --> [B, T//2, L, 2C]
        tube_feas = self.MLP(tube_feas) # [B, T//2, L, 2C] --> [B, T//2, L, C]

        return tube_feas # [B, T//2, L, C]
    