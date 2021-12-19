import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, use_minibatch=True):
        super(GATLayer, self).__init__()
        self.use_minibatch = use_minibatch
        self.g = None
        # 公式 (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # 公式 (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)

    def edge_attention(self, edges):
        # 公式 (2) 所需，边上的用户定义函数
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # 公式 (3), (4)所需，传递消息用的用户定义函数
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # 公式 (3), (4)所需, 归约用的用户定义函数
        # 公式 (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # 公式 (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, inputs):
        if self.use_minibatch:
            g, features, target_idx, idx_node = inputs
        else:
            g, features, idx_node = inputs
        self.g = g
        # 公式 (1)
        self.g.ndata['z'] = F.embedding(idx_node, features)
        # 公式 (2)
        self.g.apply_edges(self.edge_attention)
        # 公式 (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)
        if self.use_minibatch:
            ret = self.g.ndata.pop('h')[target_idx]
        else:
            ret = self.g.ndata.pop('h')
        return ret


class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, merge='mean'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(in_dim, out_dim))
        self.merge = merge

    def forward(self, inputs):
        head_outs = [attn_head(inputs) for attn_head in self.heads]
        if self.merge == 'cat':
            # 对输出特征维度（第1维）做拼接
            return torch.cat(head_outs, dim=1)
        else:
            # 用求平均整合多头结果
            return torch.mean(torch.stack(head_outs), dim=0)