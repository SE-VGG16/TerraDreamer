import math
import torch


def meters_to_pixels(meters, gsd):
    return int(round(float(meters) / max(float(gsd), 1e-6)))


def pixels_to_meters(pixels, gsd):
    return float(pixels) * float(gsd)


def day_of_year_phase(day):
    a = 2.0 * math.pi * float(day) / 365.25
    return math.sin(a), math.cos(a)


def normalize_gsd(gsd, reference=10.0):
    return math.log(max(float(gsd), 1e-6) / float(reference))


def batch_scalar(x, device):
    if torch.is_tensor(x):
        return x.float().to(device).view(-1, 1)
    return torch.tensor(x, dtype=torch.float32, device=device).view(-1, 1)
