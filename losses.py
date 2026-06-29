import torch
import torch.nn as nn
import torch.nn.functional as F


def spectral_angle_loss(pred, target):
    p = pred.flatten(2)
    t = target.flatten(2)
    dot = (p * t).sum(dim=1)
    pn = torch.linalg.norm(p, dim=1)
    tn = torch.linalg.norm(t, dim=1)
    cos = dot / (pn * tn + 1e-6)
    return torch.acos(torch.clamp(cos, -1.0 + 1e-6, 1.0 - 1e-6)).mean()


def masked_l1(pred, target, mask=None):
    e = (pred - target).abs()
    if mask is None:
        return e.mean()
    return (e * mask).sum() / (mask.sum() * pred.shape[1] + 1e-6)


def gaussian_nll(pred, target, log_sigma, mask=None):
    sigma2 = torch.exp(2.0 * log_sigma)
    e = ((pred - target).pow(2).mean(dim=1, keepdim=True) / (2.0 * sigma2)) + log_sigma
    if mask is None:
        return e.mean()
    return (e * mask).sum() / (mask.sum() + 1e-6)


def physical_penalty(pred):
    return F.relu(-pred).mean() + F.relu(pred - 1.0).mean()


class TerraLoss(nn.Module):
    def __init__(self, recon_weight=1.0, sam_weight=0.1, change_weight=0.3, phys_weight=0.05, nll_weight=0.05):
        super().__init__()
        self.recon_weight = recon_weight
        self.sam_weight = sam_weight
        self.change_weight = change_weight
        self.phys_weight = phys_weight
        self.nll_weight = nll_weight

    def forward(self, out, batch):
        pred = out['pred']
        target = batch['target']
        mask = batch['mask']
        rec = masked_l1(pred, target, mask)
        sam = spectral_angle_loss(pred, target)
        ch = F.binary_cross_entropy_with_logits(out['change_logits'], batch['change'])
        phys = physical_penalty(pred)
        nll = gaussian_nll(pred, target, out['log_sigma'], mask)
        loss = self.recon_weight * rec + self.sam_weight * sam + self.change_weight * ch + self.phys_weight * phys + self.nll_weight * nll
        return {'loss': loss, 'recon': rec.detach(), 'sam': sam.detach(), 'change': ch.detach(), 'phys': phys.detach(), 'nll': nll.detach()}
