import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from terradreamer.config import TerraConfig
from terradreamer.datasets import TerraDreamerNPZDataset, SyntheticEODataset
from terradreamer.model import TerraDreamer
from terradreamer.losses import TerraLoss
from terradreamer.metrics import compute_metrics, average_dicts
from terradreamer.utils import set_seed, device_auto, to_device, save_ckpt, save_json


def build_dataset(args, cfg):
    if args.synthetic:
        return SyntheticEODataset(length=args.synthetic_n, image_size=cfg.image_size, channels=cfg.input_channels, condition_channels=cfg.condition_channels)
    return TerraDreamerNPZDataset(args.data_root, cfg.image_size, cfg.input_channels, cfg.condition_channels)


def run_eval(model, loader, device):
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            out = model(batch['context'], batch['condition'], batch['mask'], batch['mode'], batch['dt'], batch['sensor_id'], batch['intervention_type'])
            rows.append(compute_metrics(out, batch))
    return average_dicts(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', default='data/synthetic')
    p.add_argument('--work-dir', default='runs/terradreamer')
    p.add_argument('--config', default='')
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--channels', type=int, default=3)
    p.add_argument('--condition-channels', type=int, default=8)
    p.add_argument('--image-size', type=int, default=128)
    p.add_argument('--synthetic', action='store_true')
    p.add_argument('--synthetic-n', type=int, default=128)
    args = p.parse_args()
    cfg = TerraConfig.load(args.config) if args.config else TerraConfig()
    cfg.input_channels = args.channels
    cfg.output_channels = args.channels
    cfg.condition_channels = args.condition_channels
    cfg.image_size = args.image_size
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    set_seed(cfg.seed)
    device = device_auto()
    dataset = build_dataset(args, cfg)
    val_n = max(1, int(0.15 * len(dataset)))
    train_n = len(dataset) - val_n
    train_ds, val_ds = random_split(dataset, [train_n, val_n], generator=torch.Generator().manual_seed(cfg.seed))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, drop_last=False)
    model = TerraDreamer(cfg.input_channels, cfg.output_channels, cfg.condition_channels, cfg.base_channels, cfg.modes, cfg.sensors, cfg.intervention_types).to(device)
    loss_fn = TerraLoss(cfg.recon_weight, cfg.sam_weight, cfg.change_weight, cfg.phys_weight, cfg.nll_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    cfg.save(work / 'config.yaml')
    best = -1e9
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        bar = tqdm(train_loader, desc=f'epoch {epoch}/{cfg.epochs}')
        for batch in bar:
            batch = to_device(batch, device)
            out = model(batch['context'], batch['condition'], batch['mask'], batch['mode'], batch['dt'], batch['sensor_id'], batch['intervention_type'])
            losses = loss_fn(out, batch)
            optimizer.zero_grad(set_to_none=True)
            losses['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(losses['loss'].detach())
            bar.set_postfix(loss=f'{float(losses["loss"].detach()):.4f}')
        metrics = run_eval(model, val_loader, device)
        score = metrics['ssim'] + metrics['change_f1'] + metrics['cov90']
        row = {'epoch': epoch, 'train_loss': total / max(1, len(train_loader)), **metrics}
        history.append(row)
        save_json(work / 'history.json', history)
        save_ckpt(work / 'last.pt', model, optimizer, epoch, cfg, score)
        if score > best:
            best = score
            save_ckpt(work / 'best.pt', model, optimizer, epoch, cfg, score)
        print(row)


if __name__ == '__main__':
    main()
