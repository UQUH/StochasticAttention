"""Stochastic attention end to end on a small transformer regressor. Runs on a CPU in under a minute and
downloads nothing.

1. Train a deterministic transformer on synthetic tabular data with input-dependent noise (each feature is a
   token, as in the FT-Transformer the paper uses for UCI).
2. Calibrate nu on a held-out calibration split (Section 4 and Appendix B.1 of the paper).
3. Draw S = 19 stochastic forward passes on the test split and score them: W1 of the PIT histogram (0 is
   calibrated), coverage of the [min, max] range (exactly 90% for a calibrated 19-member ensemble), its mean
   width, the ensemble spread, and the fair CRPS.

    python example.py
"""

import time

import numpy as np
import torch
import torch.nn as nn

import stochastic_attention as sa

N_FEATURES = 6
S = 19  # samples per prediction, as on ClimaX


def make_data(n, rng):
    x = rng.uniform(-1.0, 1.0, size=(n, N_FEATURES)).astype(np.float32)
    mean = np.sin(3 * x[:, 0]) + x[:, 1] * x[:, 2] + 0.5 * x[:, 3] ** 2
    noise_sd = 0.05 + 0.25 * np.abs(x[:, 4])  # the noise depends on the input
    y = mean + noise_sd * rng.standard_normal(n)
    return torch.from_numpy(x), torch.from_numpy(y.astype(np.float32))


class TabularTransformer(nn.Module):
    """Each feature becomes a token, a [CLS] token is prepended, and its final state gives the prediction."""

    def __init__(self, n_features=N_FEATURES, dim=32, heads=4, layers=2):
        super().__init__()
        self.weight = nn.Parameter(0.5 * torch.randn(n_features, dim))
        self.bias = nn.Parameter(torch.zeros(n_features, dim))
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim_feedforward=64, dropout=0.0, batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.weight + self.bias
        tokens = torch.cat([self.cls.expand(len(x), -1, -1), tokens], dim=1)
        return self.head(self.encoder(tokens)[:, 0]).squeeze(-1)


def train(model, x, y, epochs=60, batch=128):
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        for idx in torch.randperm(len(x)).split(batch):
            loss = ((model(x[idx]) - y[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    return model.eval()


def main():
    t0 = time.time()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    x_train, y_train = make_data(3000, rng)
    x_cal, y_cal = make_data(1000, rng)    # held out: used only to choose nu
    x_test, y_test = make_data(2000, rng)

    model = train(TabularTransformer(), x_train, y_train)
    with torch.no_grad():
        rmse = float(((model(x_test) - y_test) ** 2).mean().sqrt())
    print(f"deterministic model: test RMSE {rmse:.4f} (trained in {time.time() - t0:.0f} s)")

    def predict(x, nu=None):  # nu=None is the deterministic model
        with sa.use_stochastic_attention(nu):
            return model(x)

    cal_batches = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_cal, y_cal), batch_size=200,
                                              shuffle=True)
    t1 = time.time()
    result = sa.calibrate_nu(predict, cal_batches, nu_range=(1, 200))
    print(f"calibrated in {time.time() - t1:.0f} s: {result}")

    print(f"\ntest set, S = {S} samples, nominal coverage {sa.nominal_coverage(S):.0%}")
    print(f"{'nu':>10} {'W1':>8} {'coverage':>9} {'MPIW':>8} {'spread':>8} {'CRPS':>8}")
    for nu in sorted({1, result.nu, 200}):
        torch.manual_seed(1)
        samples = sa.predictive_samples(predict, x_test, nu, S)
        label = f"{nu}{' (nu*)' if nu == result.nu else ''}"
        print(f"{label:>10} {sa.pit_w1(samples, y_test):8.4f} {sa.coverage(samples, y_test):9.3f} "
              f"{sa.mpiw(samples):8.4f} {sa.spread(samples):8.4f} {sa.crps(samples, y_test):8.4f}")
    print(f"\ntotal {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
