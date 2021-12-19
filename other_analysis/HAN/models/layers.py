import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv


class SemanticAttentionLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super(SemanticAttentionLayer, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, z):
        w = self.layers(z).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)

        return (beta * z).sum(1)


class HANLayer(nn.Module):
    def __init__(self, num_metapaths, in_dim, hidden_dim, num_heads, dropout):
        super(HANLayer, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.node_attn_layer = nn.ModuleList()
        for i in range(num_metapaths):
            self.node_attn_layer.append(GATConv(in_dim, hidden_dim, num_heads, dropout, dropout, activation=F.elu))

    def forward(self, g_list, feature_list, target_idx_list):
        h = [GATLayer(g, feature).view(-1, self.num_heads * self.hidden_dim)[target_idx] for GATLayer, g, feature, target_idx in
             zip(self.node_attn_layer, g_list, feature_list, target_idx_list)]

        return h
