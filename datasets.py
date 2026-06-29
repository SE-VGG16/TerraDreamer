from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .preprocessing import percentile_clip_np, to_chw, crop_or_resize_tensor, random_rect_mask, cloud_like_mask, apply_cloud, change_from_pair


class TerraDreamerNPZDataset(Dataset):
    def __init__(self, root, image_size=128, channels=3, condition_channels=8, normalize=True):
        self.root = Path(root)
        self.files = sorted(self.root.glob('**/*.npz'))
        if len(self.files) == 0:
            raise FileNotFoundError(str(self.root))
        self.image_size = image_size
        self.channels = channels
        self.condition_channels = condition_channels
        self.normalize = normalize

    def __len__(self):
        return len(self.files)

    def _load_array(self, data, key, shape):
        if key in data:
            return data[key].astype(np.float32)
        return np.zeros(shape, dtype=np.float32)

    def __getitem__(self, index):
        data = np.load(self.files[index])
        context = data['context'].astype(np.float32)
        target = data['target'].astype(np.float32)
        if context.ndim == 4:
            t, c, h, w = context.shape
            context = context.reshape(t * c, h, w)
        else:
            context = to_chw(context)
        target = to_chw(target)
        if self.normalize:
            context = percentile_clip_np(context)
            target = percentile_clip_np(target)
        context = torch.from_numpy(context)
        target = torch.from_numpy(target)
        context = crop_or_resize_tensor(context, self.image_size)
        target = crop_or_resize_tensor(target, self.image_size)
        if context.shape[0] < self.channels:
            pad = torch.zeros(self.channels - context.shape[0], self.image_size, self.image_size)
            context = torch.cat([context, pad], dim=0)
        if context.shape[0] > self.channels:
            context = context[:self.channels]
        if target.shape[0] < self.channels:
            pad = torch.zeros(self.channels - target.shape[0], self.image_size, self.image_size)
            target = torch.cat([target, pad], dim=0)
        if target.shape[0] > self.channels:
            target = target[:self.channels]
        if 'condition' in data:
            cond = torch.from_numpy(data['condition'].astype(np.float32))
            cond = crop_or_resize_tensor(cond, self.image_size)
        else:
            cond = torch.zeros(self.condition_channels, self.image_size, self.image_size)
        if cond.shape[0] < self.condition_channels:
            pad = torch.zeros(self.condition_channels - cond.shape[0], self.image_size, self.image_size)
            cond = torch.cat([cond, pad], dim=0)
        if cond.shape[0] > self.condition_channels:
            cond = cond[:self.condition_channels]
        if 'mask' in data:
            mask = torch.from_numpy(data['mask'].astype(np.float32))
            mask = crop_or_resize_tensor(mask, self.image_size)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            mask = mask[:1]
        else:
            mask = torch.zeros(1, self.image_size, self.image_size)
        if 'change' in data:
            change = torch.from_numpy(data['change'].astype(np.float32))
            change = crop_or_resize_tensor(change, self.image_size)
            if change.ndim == 2:
                change = change.unsqueeze(0)
            change = change[:1]
        else:
            change = change_from_pair(context.unsqueeze(0), target.unsqueeze(0))[0]
        item = {
            'context': context.float(),
            'target': target.float(),
            'condition': cond.float(),
            'mask': mask.float().clamp(0, 1),
            'change': change.float().clamp(0, 1),
            'mode': torch.tensor(int(data['mode']) if 'mode' in data else 0, dtype=torch.long),
            'dt': torch.tensor(float(data['dt']) if 'dt' in data else 0.0, dtype=torch.float32),
            'sensor_id': torch.tensor(int(data['sensor_id']) if 'sensor_id' in data else 0, dtype=torch.long),
            'intervention_type': torch.tensor(int(data['intervention_type']) if 'intervention_type' in data else 0, dtype=torch.long),
            'path': str(self.files[index])
        }
        return item


class SyntheticEODataset(Dataset):
    def __init__(self, length=128, image_size=128, channels=3, condition_channels=8):
        self.length = length
        self.image_size = image_size
        self.channels = channels
        self.condition_channels = condition_channels

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        g = torch.Generator().manual_seed(index + 11)
        h = self.image_size
        w = self.image_size
        base = torch.rand(self.channels, h, w, generator=g) * 0.18 + 0.25
        yy = torch.linspace(0, 1, h).view(1, h, 1)
        xx = torch.linspace(0, 1, w).view(1, 1, w)
        base = base + 0.12 * torch.sin(14 * xx + 9 * yy)
        base = base.clamp(0, 1)
        target = base.clone()
        mode = index % 4
        mask = torch.zeros(1, h, w)
        if mode == 0:
            mask = random_rect_mask(1, h, w)[0]
            context = base * (1 - mask) + 0.5 * mask
            target = base
        elif mode == 1:
            target = torch.roll(base, shifts=(2, 3), dims=(1, 2)).clamp(0, 1)
            context = base
            mask = torch.ones(1, h, w)
        elif mode == 2:
            mask = random_rect_mask(1, h, w, 0.25, 0.45)[0]
            target = base * (1 - mask) + torch.tensor([0.55, 0.40, 0.28]).view(3, 1, 1)[:self.channels] * mask
            context = base
        else:
            cm = cloud_like_mask(1, h, w)[0]
            context = apply_cloud(base.unsqueeze(0), cm.unsqueeze(0))[0]
            target = base
            mask = cm
        condition = torch.rand(self.condition_channels, h, w, generator=g)
        change = change_from_pair(context.unsqueeze(0), target.unsqueeze(0))[0]
        return {
            'context': context.float(),
            'target': target.float(),
            'condition': condition.float(),
            'mask': mask.float(),
            'change': change.float(),
            'mode': torch.tensor(mode, dtype=torch.long),
            'dt': torch.tensor(float(mode + 1), dtype=torch.float32),
            'sensor_id': torch.tensor(0, dtype=torch.long),
            'intervention_type': torch.tensor(1 if mode == 2 else 0, dtype=torch.long),
            'path': str(index)
        }
