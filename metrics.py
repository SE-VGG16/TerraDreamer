import math
import torch
import torch.nn.functional as F


def psnr(pred, target):
    mse = F.mse_loss(pred, target).item()
    return 99.0 if mse <= 1e-12 else 20.0 * math.log10(1.0 / math.sqrt(mse))


def ssim(pred, target):
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = pred.mean(dim=(-1, -2), keepdim=True)
    mu_y = target.mean(dim=(-1, -2), keepdim=True)
    sig_x = ((pred - mu_x) ** 2).mean(dim=(-1, -2), keepdim=True)
    sig_y = ((target - mu_y) ** 2).mean(dim=(-1, -2), keepdim=True)
    sig_xy = ((pred - mu_x) * (target - mu_y)).mean(dim=(-1, -2), keepdim=True)
    v = ((2 * mu_x * mu_y + c1) * (2 * sig_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sig_x + sig_y + c2) + 1e-8)
    return v.mean().item()


def sam(pred, target):
    p = pred.flatten(2)
    t = target.flatten(2)
    dot = (p * t).sum(dim=1)
    pn = torch.linalg.norm(p, dim=1)
    tn = torch.linalg.norm(t, dim=1)
    angle = torch.acos(torch.clamp(dot / (pn * tn + 1e-6), -1.0 + 1e-6, 1.0 - 1e-6))
    return angle.mean().item()


def change_scores(logits, target, threshold=0.5):
    pred = (torch.sigmoid(logits) > threshold).float()
    target = (target > threshold).float()
    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return f1, iou


def physical_consistency(pred):
    good = ((pred >= 0.0) & (pred <= 1.0)).float().mean().item()
    return good * 100.0


def coverage90(pred, target, log_sigma):
    sigma = torch.exp(log_sigma)
    err = (pred - target).abs().mean(dim=1, keepdim=True)
    return (err <= 1.645 * sigma).float().mean().item()


def compute_metrics(out, batch):
    pred = out['pred'].detach().clamp(0, 1)
    target = batch['target'].detach().clamp(0, 1)
    f1, iou = change_scores(out['change_logits'].detach(), batch['change'].detach())
    return {
        'psnr': psnr(pred, target),
        'ssim': ssim(pred, target),
        'sam': sam(pred, target),
        'change_f1': f1,
        'region_iou': iou,
        'phys_consist': physical_consistency(pred),
        'cov90': coverage90(pred, target, out['log_sigma'].detach())
    }


def average_dicts(rows):
    keys = rows[0].keys()
    return {k: sum(float(r[k]) for r in rows) / len(rows) for k in keys}
