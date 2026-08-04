import torch
import torch.nn as nn
import math
import einops
from timm.models.vision_transformer import Mlp


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0, proj_drop=0):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        C, G = x.shape
        qkv = self.qkv(x).reshape(C, 3, self.num_heads, G // self.num_heads)
        qkv = einops.rearrange(qkv, 'c n h fph -> n h c fph')
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(0, 1)
        x = einops.rearrange(x, 'c h fph -> c (h fph)')
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, cond_dim, num_heads=8, qkv_bias=False, attn_drop=0, proj_drop=0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(cond_dim, 2 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, cond):
        B, D = x.shape

        q = self.q(x).reshape(B, self.num_heads, self.head_dim) * self.scale

        kv = self.kv(cond).reshape(B, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(1)

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).reshape(B, D)
        x = self.proj_drop(self.proj(x))
        return x


def modulate(x, shift, scale):
    res = x * (1 + scale) + shift
    return res


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiTBlock(nn.Module):
    def __init__(self, feature_dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(feature_dim)
        self.self_attn = SelfAttention(feature_dim, num_heads)

        self.norm_cross = nn.LayerNorm(feature_dim)
        self.cross_attn = CrossAttention(feature_dim, feature_dim, num_heads)

        self.norm2 = nn.LayerNorm(feature_dim)
        self.mlp = Mlp(feature_dim, hidden_features=int(feature_dim*mlp_ratio))

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(feature_dim, 9 * feature_dim, bias=True)
        )

    def forward(self, x, cond):
        (shift_msa, scale_msa, gate_msa,
         shift_cross, scale_cross, gate_cross,
         shift_mlp, scale_mlp, gate_mlp) = self.adaLN_modulation(cond).chunk(9, dim=1)

        x = x + gate_msa * self.self_attn(modulate(self.norm1(x), shift_msa, scale_msa))

        x = x + gate_cross * self.cross_attn(
            modulate(self.norm_cross(x), shift_cross, scale_cross),
            cond
        )

        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


BaseBlock = {'dit': DiTBlock}


class Decoder(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 cond_size,
                 depth,
                 dit_type,
                 num_heads,
                 mlp_ratio=4.0,
                 **kwargs) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cond_size= cond_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.dit_type = dit_type
        self.in_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size)
        )
        self.cond_layer = nn.Sequential(nn.Linear(cond_size, hidden_size))
        self.cond_proj = nn.ModuleList([
            nn.Linear(cond_size, hidden_size) for _ in range(depth)
        ])

        self.time_emb = TimestepEmbedder(hidden_size=self.hidden_size)

        self.blks = nn.ModuleList([
            BaseBlock[dit_type](self.hidden_size, mlp_ratio=self.mlp_ratio, num_heads=self.num_heads) for _ in
            range(self.depth)
        ])

        self.out_layer = FinalLayer(self.hidden_size, self.input_size)

    def forward(self, x, t, y, mask=0.0):
        x = x.float()
        t = self.time_emb(t)
        prob= 1-mask

        mask_vec = torch.bernoulli(torch.zeros(y.shape[0], device=y.device) + prob)[:, None].repeat(1, y.shape[1])

        y = y* mask_vec

        x = self.in_layer(x)

        for i, blk in enumerate(self.blks):
            c = t + self.cond_proj[i](y)
            x = blk(x, c)
        return self.out_layer(x, c)
