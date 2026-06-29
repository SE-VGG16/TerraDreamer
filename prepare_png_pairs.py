import argparse
from pathlib import Path
import numpy as np
from PIL import Image


def load_png(path, size):
    im = Image.open(path).convert('RGB').resize((size, size))
    x = np.asarray(im).astype(np.float32) / 255.0
    return np.transpose(x, (2, 0, 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', required=True)
    p.add_argument('--target-dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--size', type=int, default=128)
    p.add_argument('--mode', type=int, default=1)
    p.add_argument('--condition-channels', type=int, default=8)
    args = p.parse_args()
    in_dir = Path(args.input_dir)
    tg_dir = Path(args.target_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}])
    for i, f in enumerate(files):
        t = tg_dir / f.name
        if not t.exists():
            continue
        context = load_png(f, args.size)
        target = load_png(t, args.size)
        change = ((np.abs(context - target).mean(axis=0, keepdims=True)) > 0.12).astype(np.float32)
        mask = np.ones((1, args.size, args.size), dtype=np.float32)
        condition = np.zeros((args.condition_channels, args.size, args.size), dtype=np.float32)
        np.savez_compressed(out / f'{i:05d}.npz', context=context, target=target, mask=mask, condition=condition, change=change, mode=args.mode, dt=1.0, sensor_id=0, intervention_type=0)


if __name__ == '__main__':
    main()
