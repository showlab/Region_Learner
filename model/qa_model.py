import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm




class FCNet(nn.Module):
    """
    Simple class for multi-layer non-linear fully connect network
    Activate function: ReLU()
    """
    def __init__(self, dims, dropout=0.0, norm=True):
        super(FCNet, self).__init__()
        self.num_layers = len(dims) -1
        self.drop = dropout
        self.norm = norm
        self.main = nn.Sequential(*self._init_layers(dims))

    def _init_layers(self, dims):
        layers = []
        for i in range(self.num_layers):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            # layers.append(nn.Dropout(self.drop))
            if self.norm:
                layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            else:
                layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        return layers

    def forward(self, x):
        return self.main(x)


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout=0.0):
        super(SimpleClassifier, self).__init__()
        self.q_net = FCNet([in_dim[0], hid_dim[0]], dropout)
        self.v_net = FCNet([in_dim[1], hid_dim[0]], dropout)
        self.main = nn.Sequential(
            nn.Linear(hid_dim[0], hid_dim[1]),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            nn.Linear(hid_dim[1], out_dim)
        )

    def forward(self, q_emb, v_emb):
        joint_repr = self.q_net(q_emb) * self.v_net(v_emb)
        logits = self.main(joint_repr)
        return logits

class BUTDQAHead(nn.Module):
    def __init__(self, v_dim, q_dim, hid_dim, num_ans):
        super(BUTDQAHead, self).__init__()
        self.classifier = SimpleClassifier([q_dim, v_dim], [hid_dim, hid_dim*2], num_ans)

    def forward(self, video_embed, question_embed):
        logits = self.classifier(question_embed, video_embed)
        return logits
