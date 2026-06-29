from dataclasses import dataclass, asdict
from pathlib import Path
import json
import yaml

@dataclass
class TerraConfig:
    input_channels: int = 3
    output_channels: int = 3
    condition_channels: int = 8
    base_channels: int = 64
    depth: int = 3
    modes: int = 4
    sensors: int = 8
    intervention_types: int = 8
    image_size: int = 128
    lr: float = 0.0002
    weight_decay: float = 0.0001
    batch_size: int = 4
    epochs: int = 50
    num_workers: int = 2
    recon_weight: float = 1.0
    sam_weight: float = 0.1
    change_weight: float = 0.3
    phys_weight: float = 0.05
    nll_weight: float = 0.05
    seed: int = 7

    @classmethod
    def load(cls, path):
        p = Path(path)
        if not p.exists():
            return cls()
        data = yaml.safe_load(p.read_text()) if p.suffix in {'.yaml', '.yml'} else json.loads(p.read_text())
        return cls(**data)

    def save(self, path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix in {'.yaml', '.yml'}:
            p.write_text(yaml.safe_dump(asdict(self)))
        else:
            p.write_text(json.dumps(asdict(self), indent=2))
