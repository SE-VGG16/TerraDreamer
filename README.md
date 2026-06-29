# TerraDreamer 

This is a compact PyTorch implementation template for the proposed geo-addressable EO world-model idea. It includes dataset loading, preprocessing, model, training, testing, evaluation, utility units, and unit tests.

## Data format

Each sample is an `.npz` file with:

- `context`: float32, shape `[C,H,W]` or `[T,C,H,W]`
- `target`: float32, shape `[C,H,W]`
- `mask`: optional float32, shape `[1,H,W]`
- `condition`: optional float32, shape `[K,H,W]`
- `change`: optional float32, shape `[1,H,W]`
- `mode`: optional int, 0 spatial, 1 temporal, 2 intervention, 3 cloud
- `dt`: optional float
- `sensor_id`: optional int
- `intervention_type`: optional int

## Run quick synthetic training

```bash
cd terradreamer_code
python scripts/create_synthetic_data.py --out data/synthetic --n 80 --size 128 --channels 3 --condition-channels 8
python train.py --data-root data/synthetic --work-dir runs/demo --epochs 2 --batch-size 4 --channels 3 --condition-channels 8
python test.py --data-root data/synthetic --ckpt runs/demo/best.pt --out-dir runs/demo/test_outputs --channels 3 --condition-channels 8
python evaluate.py --data-root data/synthetic --ckpt runs/demo/best.pt --out runs/demo/eval.json --channels 3 --condition-channels 8
pytest tests
```
