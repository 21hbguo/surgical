import torch
import torch.nn as nn

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
