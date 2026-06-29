import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import torch
from terradreamer.model import TerraDreamer
from terradreamer.datasets import SyntheticEODataset
from terradreamer.losses import TerraLoss
from terradreamer.metrics import compute_metrics


def stack(items, key):
    return torch.stack([x[key] for x in items])


def test_model_shapes():
    ds = SyntheticEODataset(length=4, image_size=24, channels=3, condition_channels=8)
    items = [ds[i] for i in range(2)]
    batch = {
        'context': stack(items, 'context'),
        'target': stack(items, 'target'),
        'condition': stack(items, 'condition'),
        'mask': stack(items, 'mask'),
        'change': stack(items, 'change'),
        'mode': torch.stack([x['mode'] for x in items]),
        'dt': torch.stack([x['dt'] for x in items]),
        'sensor_id': torch.stack([x['sensor_id'] for x in items]),
        'intervention_type': torch.stack([x['intervention_type'] for x in items])
    }
    model = TerraDreamer(3, 3, 8, 8)
    out = model(batch['context'], batch['condition'], batch['mask'], batch['mode'], batch['dt'], batch['sensor_id'], batch['intervention_type'])
    assert out['pred'].shape == batch['target'].shape
    assert out['change_logits'].shape == batch['change'].shape
    assert out['log_sigma'].shape == batch['change'].shape


def test_loss_and_metrics():
    ds = SyntheticEODataset(length=4, image_size=24, channels=3, condition_channels=8)
    items = [ds[i] for i in range(2)]
    batch = {
        'context': stack(items, 'context'),
        'target': stack(items, 'target'),
        'condition': stack(items, 'condition'),
        'mask': stack(items, 'mask'),
        'change': stack(items, 'change'),
        'mode': torch.stack([x['mode'] for x in items]),
        'dt': torch.stack([x['dt'] for x in items]),
        'sensor_id': torch.stack([x['sensor_id'] for x in items]),
        'intervention_type': torch.stack([x['intervention_type'] for x in items])
    }
    model = TerraDreamer(3, 3, 8, 8)
    out = model(batch['context'], batch['condition'], batch['mask'], batch['mode'], batch['dt'], batch['sensor_id'], batch['intervention_type'])
    losses = TerraLoss()(out, batch)
    metrics = compute_metrics(out, batch)
    assert torch.isfinite(losses['loss'])
    assert 'psnr' in metrics
