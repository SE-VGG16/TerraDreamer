import random
import json
from pathlib import Path
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_auto():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def save_json(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))


def save_ckpt(path, model, optimizer, epoch, cfg, score):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'cfg': cfg.__dict__, 'score': score}, p)


def load_ckpt(path, model, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model'])
    return ckpt
