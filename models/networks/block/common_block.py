import torch
import torch.nn as nn
import torch.nn.functional as F

def _resolve_group_count(num_channels, max_groups=32):
    upper = min(max_groups, num_channels)
    for groups in range(upper, 0, -1):
        if num_channels % groups == 0:
            return groups
    return 1

def build_group_norm(num_channels, max_groups=32):
    return nn.GroupNorm(_resolve_group_count(num_channels, max_groups=max_groups), num_channels)

class FeaturePerturbation(nn.Module):
    def __init__(self, lam=0.9, kap=0.2, eps=1e-6, use_gpu=True):
        super().__init__()
        self.eps = eps
        self.lam = lam
        self.kap = kap
        self.use_gpu = use_gpu

    def forward(self, x):
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        batch_mu = mu.mean(dim=[0], keepdim=True)
        batch_psi = (mu.var(dim=[0], keepdim=True) + self.eps).sqrt()
        batch_sig = sig.mean(dim=[0], keepdim=True)
        batch_phi = (sig.var(dim=[0], keepdim=True) + self.eps).sqrt()
        epsilon = torch.empty(1, device=x.device).uniform_(-self.kap, self.kap)

        gamma = self.lam * sig + (1 - self.lam) * batch_sig + epsilon * batch_phi
        beta = self.lam * mu + (1 - self.lam) * batch_mu + epsilon * batch_psi
        x_normed = (x - mu) / sig
        return gamma * x_normed + beta

class DepthGuider(nn.Module):
    def __init__(self, in_channels, depth_channels=1, attn_size=8):
        super().__init__()
        mid = max(16, in_channels // 2)
        num_heads = max(1, min(4, mid // 16))
        while mid % num_heads != 0:
            num_heads -= 1
        self.num_heads = num_heads
        self.head_dim = mid // num_heads
        self.scale = self.head_dim ** -0.5
        self.attn_size = attn_size
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(depth_channels, mid, kernel_size=3, padding=1, bias=False),
            build_group_norm(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1, bias=False),
            build_group_norm(mid),
            nn.ReLU(inplace=True)
        )
        self.rgb_proj = nn.Conv2d(in_channels, mid, kernel_size=1, bias=False)
        self.rgb_norm = nn.LayerNorm(mid)
        self.depth_norm = nn.LayerNorm(mid)
        self.ctx_norm = nn.LayerNorm(mid)
        self.rgb_pos = nn.Parameter(torch.zeros(1, self.attn_size * self.attn_size, mid))
        self.depth_pos = nn.Parameter(torch.zeros(1, self.attn_size * self.attn_size, mid))
        self.q_proj = nn.Linear(mid, mid)
        self.k_proj = nn.Linear(mid, mid)
        self.v_proj = nn.Linear(mid, mid)
        self.ctx_proj = nn.Linear(mid, mid)
        self.fusion = nn.Sequential(
            nn.Conv2d(mid * 2, mid, kernel_size=1, bias=False),
            build_group_norm(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, kernel_size=3, padding=1, bias=False),
            build_group_norm(mid),
            nn.ReLU(inplace=True)
        )
        self.delta_proj = nn.Conv2d(mid, in_channels, kernel_size=1)
        nn.init.constant_(self.delta_proj.weight, 0)
        nn.init.constant_(self.delta_proj.bias, 0)

    def _resize_pos_embed(self, pos_embed, pool_h, pool_w):
        pos = pos_embed.view(1, self.attn_size, self.attn_size, -1).permute(0, 3, 1, 2)
        if (pool_h, pool_w) != (self.attn_size, self.attn_size):
            pos = F.interpolate(pos, size=(pool_h, pool_w), mode='bilinear', align_corners=False)
        return pos.flatten(2).transpose(1, 2)

    def compute_delta(self, rgb_feat, depth):
        b, _, h, w = rgb_feat.shape
        if depth.shape[2:] != rgb_feat.shape[2:]:
            depth = F.interpolate(depth, size=rgb_feat.shape[2:], mode='bilinear', align_corners=False)
        pool_h = min(h, self.attn_size)
        pool_w = min(w, self.attn_size)
        rgb_map = F.adaptive_avg_pool2d(self.rgb_proj(rgb_feat), (pool_h, pool_w))
        depth_map = F.adaptive_avg_pool2d(self.depth_encoder(depth), (pool_h, pool_w))
        rgb_tokens = rgb_map.flatten(2).transpose(1, 2)
        depth_tokens = depth_map.flatten(2).transpose(1, 2)
        rgb_tokens = self.rgb_norm(rgb_tokens + self._resize_pos_embed(self.rgb_pos, pool_h, pool_w))
        depth_tokens = self.depth_norm(depth_tokens + self._resize_pos_embed(self.depth_pos, pool_h, pool_w))
        q = self.q_proj(rgb_tokens).view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(depth_tokens).view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(depth_tokens).view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        ctx = torch.matmul(attn, v).transpose(1, 2).reshape(b, -1, self.num_heads * self.head_dim)
        ctx = self.ctx_norm(self.ctx_proj(ctx))
        ctx = ctx.transpose(1, 2).reshape(b, -1, pool_h, pool_w)
        fused = rgb_map + self.fusion(torch.cat([depth_map, ctx], dim=1))
        if fused.shape[2:] != (h, w):
            fused = F.interpolate(fused, size=(h, w), mode='bilinear', align_corners=False)
        return self.delta_proj(fused)

    def compute_gamma_beta(self, rgb_feat, depth):
        delta = self.compute_delta(rgb_feat, depth)
        gamma = torch.zeros_like(delta)
        beta = delta
        return gamma, beta

    def forward(self, rgb_feat, depth):
        return rgb_feat + self.compute_delta(rgb_feat, depth)
