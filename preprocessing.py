import math
import numpy as np
import torch
import torch.nn.functional as F


def percentile_clip_np(x, low=1.0, high=99.0):
    x = x.astype(np.float32)
    axes = tuple(range(1, x.ndim)) if x.ndim == 3 else None
    lo = np.percentile(x, low, axis=axes, keepdims=True)
    hi = np.percentile(x, high, axis=axes, keepdims=True)
    return np.clip((x - lo) / (hi - lo + 1e-6), 0.0, 1.0).astype(np.float32)


def to_chw(x):
    x = np.asarray(x)
    if x.ndim == 2:
        x = x[None]
    if x.ndim == 3 and x.shape[-1] in {1, 3, 4, 8, 10, 12, 13}:
        x = np.transpose(x, (2, 0, 1))
    return x.astype(np.float32)


def crop_or_resize_tensor(x, size):
    if x.ndim == 3:
        x = x.unsqueeze(0)
        y = F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)[0]
        return y
    if x.ndim == 4:
        return F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)
    raise ValueError('tensor must have shape C,H,W or B,C,H,W')


def coordinate_grid(batch, height, width, device):
    yy = torch.linspace(-1.0, 1.0, height, device=device)
    xx = torch.linspace(-1.0, 1.0, width, device=device)
    y, x = torch.meshgrid(yy, xx, indexing='ij')
    g = torch.stack([x, y, torch.sin(math.pi * x), torch.cos(math.pi * y)], dim=0)
    return g.unsqueeze(0).repeat(batch, 1, 1, 1)


def random_rect_mask(batch, height, width, min_frac=0.18, max_frac=0.42, device='cpu'):
    m = torch.zeros(batch, 1, height, width, device=device)
    for b in range(batch):
        rh = int(height * float(torch.empty(1).uniform_(min_frac, max_frac)))
        rw = int(width * float(torch.empty(1).uniform_(min_frac, max_frac)))
        y0 = int(torch.randint(0, max(1, height - rh + 1), (1,)))
        x0 = int(torch.randint(0, max(1, width - rw + 1), (1,)))
        m[b, :, y0:y0 + rh, x0:x0 + rw] = 1.0
    return m


def gaussian_kernel(size, sigma, device):
    ax = torch.arange(size, device=device) - size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    k = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    return k.view(1, 1, size, size)


def cloud_like_mask(batch, height, width, device='cpu'):
    z = torch.rand(batch, 1, height, width, device=device)
    k = gaussian_kernel(23, 5.0, device)
    s = F.conv2d(F.pad(z, (11, 11, 11, 11), mode='reflect'), k)
    q = torch.quantile(s.flatten(1), 0.72, dim=1).view(batch, 1, 1, 1)
    return (s > q).float()


def apply_cloud(x, mask, strength=0.85):
    return torch.clamp(x * (1.0 - mask * 0.65) + mask * strength, 0.0, 1.0)


def change_from_pair(a, b, threshold=0.12):
    d = (a - b).abs().mean(dim=1, keepdim=True)
    return (d > threshold).float()


def save_rgb(path, x):
    from PIL import Image
    if torch.is_tensor(x):
        x = x.detach().cpu().float().numpy()
    if x.ndim == 3 and x.shape[0] >= 3:
        x = np.transpose(x[:3], (1, 2, 0))
    if x.ndim == 3 and x.shape[0] == 1:
        x = np.repeat(np.transpose(x, (1, 2, 0)), 3, axis=2)
    x = np.clip(x, 0, 1)
    Image.fromarray((x * 255).astype(np.uint8)).save(path)
