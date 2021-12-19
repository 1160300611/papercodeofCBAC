import torch.nn as nn
import torch
import numpy as np
from models.layers import SemanticAttentionLayer, HANLayer


class HAN(nn.Module):
    def __init__(self, num_metapaths, in_dim, hidden_dim, out_dim, num_heads, dropout):
        super(HAN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(num_metapaths, in_dim, hidden_dim, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(num_metapaths, hidden_dim * num_heads[l - 1],
                                        hidden_dim, num_heads[l], dropout))
        self.semantic_attn_layer = SemanticAttentionLayer(num_heads[0] * hidden_dim)
        self.fc = nn.Linear(hidden_dim * num_heads[-1], out_dim)
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)

    def forward(self, g_list, feature_list, target_idx_list):
        for layer in self.layers:
            feature_list = layer(g_list, feature_list, target_idx_list)
        semantic_embedding = torch.stack(feature_list, dim=1)
        outputs = self.semantic_attn_layer(semantic_embedding)
        outputs = self.fc(outputs)

        return outputs


class HAN_lp(nn.Module):
    def __init__(self, num_metapath_list, feats_dim_list, in_dim, hidden_dim, out_dim, num_heads, dropout):
        super(HAN_lp, self).__init__()
        self.num_metapaths = num_metapath_list
        self.hidden_dim = hidden_dim
        self.layer1 = HAN(num_metapath_list[0], in_dim, hidden_dim, out_dim, num_heads, dropout)
        self.layer2 = HAN(num_metapath_list[1], in_dim, hidden_dim, out_dim, num_heads, dropout)
        # ntype-specific transformation
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True) for feats_dim in feats_dim_list])
        # feature dropout after trainsformation
        if dropout > 0:
            self.feat_drop = nn.Dropout(dropout)
        else:
            self.feat_drop = lambda x: x
        # initialization of fc layers
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

    def forward(self, inputs):
        g_lists, features_list, type_mask, target_idx_lists, idx_node_lists = inputs

        # ntype-specific transformation
        transformed_features = torch.zeros(type_mask.shape[0], self.hidden_dim, device=features_list[0].device)
        for i, fc in enumerate(self.fc_list):
            node_indices = np.where(type_mask == i)[0]
            transformed_features[node_indices] = fc(features_list[i])
        transformed_features = self.feat_drop(transformed_features)

        feat_list = [[], []]
        for i, nlist in enumerate(idx_node_lists):
            for nodes in nlist:
                feat_list[i].append(transformed_features[nodes])

        gene_h = self.layer1(g_lists[0], feat_list[0], target_idx_lists[0])
        dis_h = self.layer2(g_lists[1], feat_list[1], target_idx_lists[1])

        return [gene_h, dis_h]

