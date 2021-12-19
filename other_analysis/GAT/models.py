import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from layers import MultiHeadGATLayer


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(in_dim, hidden_dim, num_heads)
        # 注意输入的维度是 hidden_dim * num_heads 因为多头的结果都被拼接在了
        # 一起。 此外输出层只有一个头。
        # self.layer2 = MultiHeadGATLayer(hidden_dim * num_heads, out_dim, 1)

    def forward(self, inputs):
        logits = self.layer1(inputs)
        # logits = F.elu(logits)
        # logits = self.layer2(logits)
        return logits

class GAT_lp(nn.Module):
    def __init__(self, feats_dim_list, in_dim, hidden_dim, out_dim, num_heads, dropout_rate=0.5):
        super(GAT_lp, self).__init__()
        self.hidden_dim = hidden_dim

        # ntype-specific transformation
        self.fc_list = nn.ModuleList([nn.Linear(feats_dim, hidden_dim, bias=True) for feats_dim in feats_dim_list])
        # feature dropout after trainsformation
        if dropout_rate > 0:
            self.feat_drop = nn.Dropout(dropout_rate)
        else:
            self.feat_drop = lambda x: x
        # initialization of fc layers
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        self.gene_layer = GAT(in_dim, hidden_dim, out_dim, num_heads)
        self.disease_layer = GAT(in_dim, hidden_dim, out_dim, num_heads)
        self.gene_fc = nn.Linear(out_dim, out_dim)
        self.disease_fc = nn.Linear(out_dim, out_dim)
        nn.init.xavier_normal_(self.gene_fc.weight, gain=1.414)
        nn.init.xavier_normal_(self.disease_fc.weight, gain=1.414)

    def forward(self, inputs):
        g, features, type_mask, target_idx, idx_node = inputs

        # ntype-specific transformation
        transformed_features = torch.zeros(type_mask.shape[0], self.hidden_dim, device=features[0].device)
        for i, fc in enumerate(self.fc_list):
            node_indices = np.where(type_mask == i)[0]
            transformed_features[node_indices] = fc(features[i])
        transformed_features = self.feat_drop(transformed_features)
        logits_gene = self.gene_layer((g[0][0], transformed_features, target_idx[0][0], idx_node[0][0]))
        logits_disease = self.disease_layer((g[1][0], transformed_features, target_idx[1][0], idx_node[1][0]))

        return logits_gene, logits_disease

