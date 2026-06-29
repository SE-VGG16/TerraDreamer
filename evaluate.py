import argparse
import torch
from torch.utils.data import DataLoader
from terradreamer.config import TerraConfig
from terradreamer.datasets import TerraDreamerNPZDataset, SyntheticEODataset
from terradreamer.model import TerraDreamer
from terradreamer.metrics import compute_metrics, average_dicts
from terradreamer.utils import device_auto, to_device, load_ckpt, save_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', default='data/synthetic')
    p.add_argument('--ckpt', required=True)
    p.add_argument('--out', default='eval.json')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--channels', type=int, default=3)
    p.add_argument('--condition-channels', type=int, default=8)
    p.add_argument('--image-size', type=int, default=128)
    p.add_argument('--synthetic', action='store_true')
    args = p.parse_args()
    cfg = TerraConfig(input_channels=args.channels, output_channels=args.channels, condition_channels=args.condition_channels, image_size=args.image_size)
    device = device_auto()
    dataset = SyntheticEODataset(64, cfg.image_size, cfg.input_channels, cfg.condition_channels) if args.synthetic else TerraDreamerNPZDataset(args.data_root, cfg.image_size, cfg.input_channels, cfg.condition_channels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = TerraDreamer(cfg.input_channels, cfg.output_channels, cfg.condition_channels, cfg.base_channels, cfg.modes, cfg.sensors, cfg.intervention_types).to(device)
    load_ckpt(args.ckpt, model, device)
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            out = model(batch['context'], batch['condition'], batch['mask'], batch['mode'], batch['dt'], batch['sensor_id'], batch['intervention_type'])
            rows.append(compute_metrics(out, batch))
    result = average_dicts(rows)
    save_json(args.out, result)
    print(result)


if __name__ == '__main__':
    main()
