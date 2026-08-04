import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FFN(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output, alpha):
        super().__init__()
        self.dense1 = nn.Linear(dim_input, dim_hidden)
        self.elu = torch.nn.ELU(alpha)
        self.dense2 = nn.Linear(dim_hidden, dim_output)

    def forward(self, X):
        return self.dense2(self.elu(self.dense1(X)))


class AddNorm(nn.Module):
    def __init__(self, dropout, layer_dim):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(layer_dim)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.4, use_bias=True):
        super().__init__()

        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)

    def forward(self, h):
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)

        Q_reshape = Q_h.view(self.num_heads, -1, self.out_dim)
        K_reshape = K_h.view(self.num_heads, -1, self.out_dim)
        V_reshape = V_h.view(self.num_heads, -1, self.out_dim)
        sim_mat = F.softmax(torch.matmul(Q_reshape, K_reshape.permute(0, 2, 1)), -1)
        sim_mat = self.dropout(sim_mat)
        head_out = torch.matmul(sim_mat, V_reshape).view(-1, self.num_heads * self.out_dim)
        return head_out


class TransformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, attn_drop=0.6, add_drop=0.2, alpha=1.0):
        super().__init__()
        self.addnorm1 = AddNorm(add_drop, in_dim)
        self.addnorm2 = AddNorm(add_drop, in_dim)
        self.ffn = FFN(in_dim, in_dim * 2, in_dim, alpha)
        self.attention = MultiHeadAttentionLayer(in_dim, out_dim, num_heads, dropout=attn_drop)
        self.activate = torch.nn.ELU()
        self.O = nn.Linear(out_dim * num_heads, in_dim)

    def forward(self, h):
        h_in1 = h
        attn_out = self.attention(h)
        a_trans = self.O(attn_out)
        h = self.addnorm1(h_in1, a_trans)
        h_in2 = h
        h = self.addnorm2(h_in2, self.ffn(h))
        return h


class Encoder(nn.Module):
    def __init__(self, xtype, dims, net_params):
        super().__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        num_heads = net_params['num_heads']
        attn_drop = net_params['attn_drop']
        num_layers = net_params['num_layers']
        add_drop = net_params['add_drop']
        final_embed = net_params['final_embed']

        self.embedding_h1 = nn.Linear(in_dim, hidden_dim)
        self.xtype = xtype

        self.layers = nn.ModuleList([TransformerLayer(hidden_dim, out_dim, num_heads, attn_drop, add_drop)
                                     for _ in range(num_layers)])
        self.final_projector = nn.Linear(hidden_dim, final_embed)

        self.contrast_proj = ProjectionHead(final_embed)

    def forward(self, h1, h2d=None):
        h1 = self.embedding_h1(h1)
        for layer in self.layers:
            h1 = layer(h1)

        rep = self.final_projector(h1)
        rep_p = self.contrast_proj(rep)

        return rep, rep_p


class ProjectionHead(nn.Module):
    def __init__(
            self,
            embedding_dim,
            projection_dim=128
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.layer_norm = nn.LayerNorm(projection_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        projected = self.projection(x)
        x = self.layer_norm(projected)
        x = self.gelu(x)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        return x
