import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GATLayer, MultiHeadGATLayer
import numpy as np


class BaseGAT(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super(BaseGAT, self).__init__()
        self.layer_src = MultiHeadGATLayer(in_dim, out_dim, num_heads)
        self.layer_dst = MultiHeadGATLayer(in_dim, out_dim, num_heads)

    def forward(self, inputs):
        if self.use_minibatch:
            g, features_list, type_mask, idx_batch = inputs
        else:
            g, features_list, type_mask = inputs

        # ntype-specific transformation
        transformed_features = torch.zeros(type_mask.shape[0], self.hidden_dim, device=features_list[0].device)
        for i, fc in enumerate(self.fc_list):
            node_indices = np.where(type_mask == i)[0]
            transformed_features[node_indices] = fc(features_list[i])
        transformed_features = self.feat_drop(transformed_features)

        h = self.layer(inputs)
        return h
