import torch
import torch.nn as nn
from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.pool import global_mean_pool
import torch.nn.functional as func
import numpy as np


class GeneralizedCELoss(nn.Module):
    def __init__(self, q=0.7):
        super(GeneralizedCELoss, self).__init__()
        self.q = q

    def forward(self, logits, targets):
        p = func.softmax(logits, dim=1)
        Yg = torch.gather(p, 1, torch.unsqueeze(targets, 1))
        # modify gradient of cross entropy
        loss_weight = (Yg.squeeze().detach() ** self.q) * self.q
        loss = func.cross_entropy(logits, targets, reduction='none') * loss_weight
        return loss


class GCNNet(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(2)])
        self.act_fn = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index, batch, node_mask=None, edge_mask=None):
        if node_mask is not None:
            x = x * node_mask
        x = self.dropout(self.fc(x))
        for conv in self.layers:
            x = conv(x, edge_index, edge_weight=edge_mask)
            x = self.dropout(self.act_fn(x))
        return global_mean_pool(x, batch)


class GCNMasker(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(2)])
        self.act_fn = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.node_score_fn = nn.Linear(hidden_dim, 1)
        self.edge_score_fn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, edge_index):
        x = self.dropout(self.fc(x))

        for conv in self.layers:
            x = conv(x, edge_index)
            x = self.dropout(self.act_fn(x))
        node_score = self.node_score_fn(x)
        node_score = func.sigmoid(node_score)
        row, col = edge_index
        edge_score = torch.cat([x[row], x[col]], dim=-1)
        edge_score = self.edge_score_fn(edge_score)
        edge_score = func.sigmoid(edge_score)
        return edge_score, node_score


class MyModel(nn.Module):
    def __init__(self, in_dim=1024, hidden_dim=256):
        super().__init__()
        self.masker = GCNMasker(in_dim=in_dim, hidden_dim=hidden_dim)
        self.gnn_c = GCNNet(in_dim=in_dim, hidden_dim=hidden_dim)
        self.gnn_b = GCNNet(in_dim=in_dim, hidden_dim=hidden_dim)

        self.mlp_c = nn.Linear(hidden_dim * 2, 2)
        self.mlp_b = nn.Linear(hidden_dim * 2, 2)

        self.classifier_loss_fn = nn.CrossEntropyLoss(reduction='none')
        self.bias_loss_fn = GeneralizedCELoss()

    def forward(self, data, swap=False):
        batch_size = data.y.shape[0]
        edge_score_c, node_score_c = self.masker(data.x, data.edge_index)
        edge_score_b = 1 - edge_score_c
        node_score_b = 1 - node_score_c

        score_c = self.gnn_c(data.x, data.edge_index, data.batch, node_score_c, edge_score_c)
        score_b = self.gnn_b(data.x, data.edge_index, data.batch, node_score_b, edge_score_b)

        z_c = torch.cat((score_c, score_b.detach()), dim=1)
        z_b = torch.cat((score_c.detach(), score_b), dim=1)

        pred_c = self.mlp_c(z_c)
        pred_b = self.mlp_b(z_b)

        loss_c = self.classifier_loss_fn(pred_c, data.y).detach()
        loss_b = self.classifier_loss_fn(pred_b, data.y).detach()

        loss_weight = loss_b / (loss_b + loss_c + 1e-8)

        loss_dis_conflict = self.classifier_loss_fn(pred_c, data.y)
        loss_dis_align = self.bias_loss_fn(pred_b, data.y)

        indices = np.random.permutation(batch_size)
        z_b_swap = score_b[indices]
        label_swap = data.y[indices]
        z_mix_conflict = torch.cat((score_c, z_b_swap.detach()), dim=1)
        z_mix_align = torch.cat((score_c.detach(), z_b_swap), dim=1)
        pred_mix_conflict = self.mlp_c(z_mix_conflict)
        pred_mix_align = self.mlp_b(z_mix_align)
        loss_swap_conflict = self.classifier_loss_fn(pred_mix_conflict, data.y)
        loss_swap_align = self.bias_loss_fn(pred_mix_align, label_swap)
        lambda_swap = 15

        loss_swap = loss_swap_conflict.mean() + 1 * loss_swap_align.mean()
        loss_dis = loss_dis_conflict.mean() + 1 * loss_dis_align.mean()
        loss = loss_dis + lambda_swap * loss_swap

        return self.mlp_c(z_c), loss, data.y, batch_size


