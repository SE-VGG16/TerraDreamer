import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def smooth_noise(size, seed):
    rng = np.random.default_rng(seed)
    x = rng.random((size, size, 3), dtype=np.float32)
    im = Image.fromarray((x * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=5))
    return np.asarray(im).astype(np.float32) / 255.0


def draw_roads(img, seed):
    rng = np.random.default_rng(seed)
    im = Image.fromarray((img * 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    for _ in range(5):
        pts = []
        y = int(rng.integers(0, img.shape[0]))
        for x in range(0, img.shape[1], 16):
            y = int(np.clip(y + rng.integers(-12, 13), 0, img.shape[0] - 1))
            pts.append((x, y))
        d.line(pts, fill=(120, 115, 95), width=int(rng.integers(1, 4)))
    return np.asarray(im).astype(np.float32) / 255.0


def rect_mask(size, rng):
    m = np.zeros((1, size, size), dtype=np.float32)
    h = int(rng.integers(size // 5, size // 2))
    w = int(rng.integers(size // 5, size // 2))
    y = int(rng.integers(0, size - h))
    x = int(rng.integers(0, size - w))
    m[:, y:y + h, x:x + w] = 1.0
    return m


def sample(i, size, channels, condition_channels):
    rng = np.random.default_rng(i)
    img = smooth_noise(size, i)
    img = img * np.array([0.35, 0.58, 0.40], dtype=np.float32)
    img = draw_roads(img, i + 100)
    context = np.transpose(img, (2, 0, 1))[:channels]
    target = context.copy()
    mode = i % 4
    mask = rect_mask(size, rng)
    if mode == 0:
        context = context * (1.0 - mask) + 0.5 * mask
    elif mode == 1:
        target = np.roll(target, shift=(2, 3), axis=(1, 2))
        mask[:] = 1.0
    elif mode == 2:
        color = np.array([0.55, 0.42, 0.30], dtype=np.float32)[:channels, None, None]
        target = target * (1.0 - mask) + color * mask
    else:
        cloud = smooth_noise(size, i + 400).mean(axis=2)
        cloud = (cloud > np.quantile(cloud, 0.72)).astype(np.float32)[None]
        mask = cloud
        context = np.clip(context * (1.0 - 0.65 * cloud) + 0.9 * cloud, 0, 1)
    change = ((np.abs(context - target).mean(axis=0, keepdims=True)) > 0.12).astype(np.float32)
    condition = rng.random((condition_channels, size, size), dtype=np.float32)
    return context.astype(np.float32), target.astype(np.float32), mask.astype(np.float32), condition.astype(np.float32), change.astype(np.float32), mode


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='data/synthetic')
    p.add_argument('--n', type=int, default=128)
    p.add_argument('--size', type=int, default=128)
    p.add_argument('--channels', type=int, default=3)
    p.add_argument('--condition-channels', type=int, default=8)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(args.n):
        context, target, mask, condition, change, mode = sample(i, args.size, args.channels, args.condition_channels)
        np.savez_compressed(out / f'{i:05d}.npz', context=context, target=target, mask=mask, condition=condition, change=change, mode=mode, dt=float(mode + 1), sensor_id=0, intervention_type=1 if mode == 2 else 0)


if __name__ == '__main__':
    main()
