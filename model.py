import torch
import torch.nn as nn
import torch.nn.functional as F
from .preprocessing import coordinate_grid


class ResidualBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.n1 = nn.GroupNorm(min(8, cout), cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.n2 = nn.GroupNorm(min(8, cout), cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        y = F.silu(self.n1(self.c1(x)))
        y = self.n2(self.c2(y))
        return F.silu(y + self.skip(x))


class Down(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = ResidualBlock(cin, cout)
        self.down = nn.Conv2d(cout, cout, 4, stride=2, padding=1)

    def forward(self, x):
        s = self.block(x)
        return self.down(s), s


class Up(nn.Module):
    def __init__(self, cin, skip, cout):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, 4, stride=2, padding=1)
        self.block = ResidualBlock(cout + skip, cout)

    def forward(self, x, s):
        x = self.up(x)
        if x.shape[-2:] != s.shape[-2:]:
            x = F.interpolate(x, size=s.shape[-2:], mode='bilinear', align_corners=False)
        return self.block(torch.cat([x, s], dim=1))


class TerraDreamer(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, condition_channels=8, base_channels=64, modes=4, sensors=8, intervention_types=8):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.condition_channels = condition_channels
        self.mode_emb = nn.Embedding(modes, base_channels)
        self.sensor_emb = nn.Embedding(sensors, base_channels)
        self.intervention_emb = nn.Embedding(intervention_types, base_channels)
        self.dt_mlp = nn.Sequential(nn.Linear(1, base_channels), nn.SiLU(), nn.Linear(base_channels, base_channels))
        self.film = nn.Sequential(nn.Linear(base_channels * 4, base_channels * 4), nn.SiLU(), nn.Linear(base_channels * 4, base_channels * 8))
        cin = input_channels + condition_channels + 1 + 4
        self.stem = ResidualBlock(cin, base_channels)
        self.d1 = Down(base_channels, base_channels * 2)
        self.d2 = Down(base_channels * 2, base_channels * 4)
        self.mid = ResidualBlock(base_channels * 4, base_channels * 4)
        self.u2 = Up(base_channels * 4, base_channels * 4, base_channels * 2)
        self.u1 = Up(base_channels * 2, base_channels * 2, base_channels)
        self.refine = ResidualBlock(base_channels, base_channels)
        self.pred = nn.Conv2d(base_channels, output_channels, 1)
        self.change = nn.Conv2d(base_channels, 1, 1)
        self.log_sigma = nn.Conv2d(base_channels, 1, 1)

    def control(self, mode, sensor_id, intervention_type, dt):
        if dt.ndim == 1:
            dt = dt[:, None]
        z = torch.cat([
            self.mode_emb(mode),
            self.sensor_emb(sensor_id),
            self.intervention_emb(intervention_type),
            self.dt_mlp(dt.float())
        ], dim=1)
        return self.film(z)

    def forward(self, context, condition, mask, mode, dt, sensor_id, intervention_type):
        b, c, h, w = context.shape
        coords = coordinate_grid(b, h, w, context.device)
        x = torch.cat([context, condition, mask, coords], dim=1)
        x = self.stem(x)
        x, s1 = self.d1(x)
        x, s2 = self.d2(x)
        x = self.mid(x)
        g = self.control(mode, sensor_id, intervention_type, dt).view(b, -1, 1, 1)
        gamma, beta = g.chunk(2, dim=1)
        x = x * (1.0 + gamma) + beta
        x = self.u2(x, s2)
        x = self.u1(x, s1)
        x = self.refine(x)
        pred = torch.sigmoid(self.pred(x))
        change_logits = self.change(x)
        log_sigma = torch.clamp(self.log_sigma(x), -5.0, 2.0)
        return {'pred': pred, 'change_logits': change_logits, 'log_sigma': log_sigma}
