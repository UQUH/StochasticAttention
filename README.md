# Calibrating Scientific Foundation Models with Inference-Time Stochastic Attention

> An inference-time method that gives a pretrained transformer calibrated predictive uncertainty, without retraining.

Transformer-based scientific foundation models give deterministic outputs and provide limited support for
calibrated predictive uncertainty.

This repository provides **Stochastic Attention (SA)**, which replaces the softmax attention weights with
**normalized multinomial samples** controlled by a single **concentration parameter ν**, so that repeated forward
passes form a **predictive ensemble**. ν is set by a **univariate post-hoc calibration**, solved with Bayesian
optimization under uncertainty ([SO-BO-scale](https://github.com/UQUH/SO-BO-scale)). The model's weights are never
changed.

## Repository Layout

```
stochastic_attention.py        — the method: SA weights, switching SA on in a model, calibration of ν, metrics
example.py                     — a small transformer on synthetic data (CPU, under a minute)
test_stochastic_attention.py   — tests
```

The code for the paper's experiments (ClimaX, TimesFM, and FT-Transformer on UCI) will be added to this repository.

## Requirements

- Python 3.8 or later
- `torch` 1.12 or later
- `numpy`
- `pytest` (for the tests only)

## Quick Start

Clone the repository:

```bash
git clone https://github.com/UQUH/StochasticAttention.git
cd StochasticAttention
```

Run the example and the tests:

```bash
python example.py
pytest
```

## Using SA on Your Model

```python
import stochastic_attention as sa

def predict(x, nu=None):  # nu=None gives the deterministic model
    with sa.use_stochastic_attention(nu, model):
        return model(x)

result = sa.calibrate_nu(predict, calibration_loader)            # held-out (x, y) batches
samples = sa.predictive_samples(predict, x_test, result.nu, 19)  # 19 stochastic forward passes
```

This works as it is when the model's attention calls `torch.nn.functional.scaled_dot_product_attention`. If the
model computes the softmax itself, as ClimaX does, first call `sa.install_attention_samplers(model, "attn_drop")`.

## Citation

If you find Stochastic Attention helpful, please cite the following paper:

```bibtex
@inproceedings{yadav2026stochastic,
  title     = {Calibrating Scientific Foundation Models with Inference-Time Stochastic Attention},
  author    = {Yadav, Akash and Adebiyi, Taiwo A. and Zhang, Ruda},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

The calibration of ν follows:

```bibtex
@article{yadav2026bayesian,
  title   = {Bayesian Optimization under Uncertainty for Training a Scale Parameter in Stochastic Models},
  author  = {Yadav, Akash and Zhang, Ruda},
  journal = {ASCE-ASME Journal of Risk and Uncertainty in Engineering Systems, Part A: Civil Engineering},
  volume  = {12},
  number  = {4},
  pages   = {04026050},
  year    = {2026},
  doi     = {10.1061/AJRUA6.RUENG-1854}
}
```

## The Team

Stochastic Attention was developed by the [Uncertainty Quantification Lab](https://uq.uh.edu/group-members) at the
University of Houston.

Primary contributors:
- Akash Yadav\*
- Taiwo A. Adebiyi\*
- Ruda Zhang

\*Equal contribution.
