"""Stochastic attention (SA): calibrated predictive uncertainty from a pretrained transformer, without retraining.

Code for "Calibrating Scientific Foundation Models with Inference-Time Stochastic Attention" (NeurIPS 2026),
by Akash Yadav, Taiwo A. Adebiyi and Ruda Zhang. MIT license.

    import stochastic_attention as sa

    def predict(x, nu=None):                      # nu=None gives the deterministic model
        with sa.use_stochastic_attention(nu, model):
            return model(x)

    result = sa.calibrate_nu(predict, calibration_loader)            # nu* on held-out (x, y) batches
    samples = sa.predictive_samples(predict, x_test, result.nu, 19)  # (19, *prediction shape)
    print(sa.pit_w1(samples, y_test), sa.coverage(samples, y_test))

The file has five parts:
1. the stochastic attention weights and a stochastic scaled dot-product attention (Section 3 of the paper);
2. switching SA on in an existing model;
3. choosing nu on held-out data (Section 4 and Appendix B.1);
4. drawing the predictive ensemble;
5. calibration and sharpness metrics (Section 5.2).

It needs PyTorch 1.12 or later and NumPy. Switching SA on through torch.nn.functional.scaled_dot_product_attention
needs PyTorch 2.0 or later.
"""

from __future__ import annotations

import contextlib
import math
import operator
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterable, Iterator, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__version__ = "0.1.0"


# =====================================================================================================
# 1. Stochastic attention weights
#
# For one attention row, pi is the softmax distribution over the keys and V the value matrix. Deterministic
# attention returns the average pi^T V. Stochastic attention draws nu keys from pi with replacement and
# averages their value vectors, which is the same as multiplying V by the weights counts / nu with
# counts ~ Multinomial(nu, pi) (Definition 3.1). The weights stay on the probability simplex, their mean is
# pi (Proposition 3.1) and their covariance is (diag(pi) - pi pi^T) / nu (Proposition 3.2), so a large nu
# gives back the deterministic model.
# =====================================================================================================

# The nu draws of a row are taken in chunks of this size, so memory grows with rows x _CHUNK, not rows x nu.
_CHUNK = 256


def check_nu(nu) -> int:
    """Return ``nu`` as a positive Python ``int``, or raise ``TypeError`` or ``ValueError``."""
    if isinstance(nu, bool):
        raise TypeError("nu must be a positive integer, got a bool")
    try:
        value = operator.index(nu)
    except TypeError:
        raise TypeError(f"nu must be a positive integer, got {nu!r}") from None
    if value < 1:
        raise ValueError(f"nu must be a positive integer, got {value}")
    return value


def sample_attention_weights(
    probs: torch.Tensor,
    nu: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Replace every attention row by its stochastic attention weights.

    Args:
        probs: Attention probabilities with the keys on the last dimension, for example the
            ``(batch, heads, queries, keys)`` output of a softmax.
        nu: The concentration parameter, a positive integer: the number of keys drawn per row.
        generator: Optional ``torch.Generator`` for the draws.

    Returns:
        A tensor with the shape and dtype of ``probs`` whose rows are ``counts / nu`` with
        ``counts ~ Multinomial(nu, row)``. SA acts after the masked softmax, so a row that is not a
        distribution (all zeros, or NaN for a fully masked query) is returned unchanged.
    """
    nu = check_nu(nu)
    shape = probs.shape
    work_dtype = torch.float64 if probs.dtype == torch.float64 else torch.float32
    rows = probs.detach().reshape(-1, shape[-1]).to(work_dtype)
    clipped = rows.clamp_min(0.0)
    valid = torch.isfinite(rows).all(dim=-1) & (clipped.sum(dim=-1) > 0)
    all_valid = bool(valid.all())
    p = clipped if all_valid else clipped[valid]
    if p.shape[0] == 0:
        return probs.clone()

    counts = torch.zeros_like(p)
    remaining = nu
    while remaining > 0:
        m = min(remaining, _CHUNK)
        idx = torch.multinomial(p, m, replacement=True, generator=generator)
        counts.scatter_add_(1, idx, torch.ones_like(idx, dtype=counts.dtype))
        remaining -= m
    weights = counts / nu

    if all_valid:
        out = weights
    else:
        out = rows.clone()
        out[valid] = weights
    return out.reshape(shape).to(probs.dtype)


def stochastic_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    *,
    nu: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """Scaled dot-product attention with stochastic attention weights.

    Same arguments and result as ``torch.nn.functional.scaled_dot_product_attention`` (it follows the reference
    implementation in the PyTorch documentation), plus ``nu``. With ``nu=None`` the weights are the exact
    softmax, so this is ordinary attention; with an integer ``nu`` every attention row is replaced by
    ``sample_attention_weights(row, nu)`` after the mask and the softmax. A boolean ``attn_mask`` marks with
    ``True`` the keys a query may attend to; a float mask is added to the scores.
    """
    n_q, n_k = query.size(-2), key.size(-2)
    scale_factor = 1.0 / math.sqrt(query.size(-1)) if scale is None else scale
    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), dim=-3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), dim=-3)

    scores = torch.matmul(query, key.transpose(-2, -1)) * scale_factor
    if is_causal:
        if attn_mask is not None:
            raise ValueError("pass either attn_mask or is_causal=True, not both")
        causal = torch.ones(n_q, n_k, dtype=torch.bool, device=query.device).tril(diagonal=0)
        scores = scores.masked_fill(~causal, float("-inf"))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask

    weights = torch.softmax(scores, dim=-1)
    if nu is not None:
        weights = sample_attention_weights(weights, nu, generator=generator)
    if dropout_p > 0.0:
        weights = torch.dropout(weights, dropout_p, train=True)
    return torch.matmul(weights, value)


# =====================================================================================================
# 2. Switching SA on in an existing model
#
# Two mechanisms cover most transformer code, and neither edits the model's source.
# (a) The attention calls torch.nn.functional.scaled_dot_product_attention (PyTorch's TransformerEncoder and
#     MultiheadAttention with need_weights=False, Hugging Face models with attn_implementation="sdpa", timm 0.9+,
#     the PyTorch TimesFM 2.5). Inside `with use_stochastic_attention(nu):` that function is replaced by its
#     stochastic version.
# (b) The model computes the softmax itself and applies a module, usually attention dropout, to the
#     probabilities before they multiply the values (timm's attn_drop, used by ClimaX; the FT-Transformer's
#     attention dropout). install_attention_samplers(model, "attn_drop") replaces that module by an
#     AttentionSampler, which is the identity until use_stochastic_attention(nu, model) sets nu.
# Both are inference-time changes: train the model first. The replacement in (a) is global while the block
# runs, so it is not thread-safe, and it does not reach code compiled with torch.compile or TorchScript.
# =====================================================================================================

# Rows of attention probabilities sum to 1. Half-precision softmax rows can be off by about 1e-3, while a
# tensor that is not attention probabilities is off by far more.
_ROW_SUM_TOL = 1e-2

# Number of attention calls made stochastic so far, by the replaced scaled_dot_product_attention or by a
# sampler. A with-block compares it at entry and exit, which also counts calls made in a nested block.
_STOCHASTIC_CALLS = [0]


class AttentionSampler(nn.Module):
    """Stochastic attention as a module applied to attention probabilities.

    With ``nu=None`` it is the identity, so a model with samplers installed behaves exactly as before; with an
    integer ``nu`` it returns ``sample_attention_weights(probs, nu)``. It first checks that its input holds
    attention probabilities (nonnegative rows summing to one on the last dimension) and raises otherwise,
    because then the module it replaced was not applied to the attention weights.
    """

    def __init__(self, nu: Optional[int] = None, name: str = "") -> None:
        super().__init__()
        self.nu = None if nu is None else check_nu(nu)
        self.name = name

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        if self.nu is None:
            return probs
        _check_probabilities(probs, self.name)
        _STOCHASTIC_CALLS[0] += 1
        return sample_attention_weights(probs, self.nu)

    def extra_repr(self) -> str:
        return f"nu={self.nu}"


def _check_probabilities(probs: torch.Tensor, name: str) -> None:
    sums = probs.detach().float().sum(dim=-1)
    finite = torch.isfinite(sums)
    off = finite & (sums != 0) & ((sums - 1.0).abs() > _ROW_SUM_TOL)
    negative = bool((probs.detach() < -_ROW_SUM_TOL).any())
    if bool(off.any()) or negative:
        lo = float(sums[finite].min()) if bool(finite.any()) else float("nan")
        hi = float(sums[finite].max()) if bool(finite.any()) else float("nan")
        where = f" at '{name}'" if name else ""
        raise RuntimeError(
            f"AttentionSampler{where} received a tensor of shape {tuple(probs.shape)} that does not hold "
            f"attention probabilities (row sums on the last dimension range from {lo:.4g} to {hi:.4g}"
            f"{', and some entries are negative' if negative else ''}). The module it replaced is not "
            "applied to the softmax output. Install the samplers on the attribute that holds the "
            "attention dropout, or restrict them with module_type=."
        )


def install_attention_samplers(
    model: nn.Module,
    attribute: str = "attn_drop",
    module_type: Optional[Union[Type[nn.Module], Tuple[Type[nn.Module], ...]]] = None,
) -> List[str]:
    """Replace the module that each attention layer applies to its probabilities by an ``AttentionSampler``.

    Args:
        model: The model, already trained.
        attribute: Name of the attribute holding the module applied to the attention probabilities right
            before they multiply the values: ``"attn_drop"`` in timm before 0.9 and in ClimaX, ``"dropout"``
            in the FT-Transformer's attention layer (where it may be ``None``).
        module_type: Optional class, or tuple of classes. Only modules of this type are changed; use it when
            ``attribute`` also names modules that are not attention dropout.

    Returns:
        The qualified names of the replaced attributes; every one should be an attention layer. The samplers
        start switched off (``nu=None``); ``use_stochastic_attention(nu, model)`` switches them on.
    """
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, AttentionSampler):
            continue
        if module_type is not None and not isinstance(module, module_type):
            continue
        registered = attribute in module._modules
        none_attribute = attribute in vars(module) and vars(module)[attribute] is None
        if registered or none_attribute:
            targets.append((name, module))
    if not targets:
        kind = "" if module_type is None else f" of type {module_type}"
        raise ValueError(
            f"no submodule{kind} of {type(model).__name__} has an attribute '{attribute}'. Look up, in the "
            "model's attention code, the name of the module applied to the softmax output; if the attention "
            "calls torch.nn.functional.scaled_dot_product_attention instead, use use_stochastic_attention(nu) "
            "without installing samplers."
        )
    names = []
    for name, module in targets:
        qualified = f"{name}.{attribute}" if name else attribute
        setattr(module, attribute, AttentionSampler(name=qualified))
        names.append(qualified)
    return names


def attention_samplers(model: nn.Module) -> List[AttentionSampler]:
    """All ``AttentionSampler`` modules installed in ``model``."""
    return [m for m in model.modules() if isinstance(m, AttentionSampler)]


def _get_mha_fastpath() -> Optional[bool]:
    getter = getattr(getattr(torch.backends, "mha", None), "get_fastpath_enabled", None)
    return None if getter is None else bool(getter())


def _set_mha_fastpath(enabled: Optional[bool]) -> None:
    setter = getattr(getattr(torch.backends, "mha", None), "set_fastpath_enabled", None)
    if enabled is not None and setter is not None:
        setter(enabled)


@contextlib.contextmanager
def use_stochastic_attention(
    nu: Optional[int],
    model: Optional[nn.Module] = None,
    *,
    strict: bool = True,
) -> Iterator[None]:
    """Run attention stochastically, with concentration ``nu``, inside a ``with`` block.

    Inside the block every call to ``torch.nn.functional.scaled_dot_product_attention`` uses stochastic
    attention weights, and so do the ``AttentionSampler`` modules installed in ``model`` if it is given.
    PyTorch's fast path for ``nn.MultiheadAttention`` and ``nn.TransformerEncoderLayer``, which bypasses that
    function, is turned off for the block. Everything is restored on exit, also when the block raises.

    ``nu=None`` does nothing, which makes it easy to write one ``predict(x, nu)`` function for both the
    deterministic and the stochastic model.

    With ``strict=True`` (the default), leaving the block raises ``RuntimeError`` if no attention call inside
    it was made stochastic, because the model then ran deterministically.
    """
    if nu is None:
        yield
        return
    nu = check_nu(nu)
    samplers = attention_samplers(model) if model is not None else []
    saved_nu = [s.nu for s in samplers]
    calls_at_entry = _STOCHASTIC_CALLS[0]
    original_sdpa = getattr(F, "scaled_dot_product_attention", None)
    saved_fastpath = _get_mha_fastpath()

    def _stochastic_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
                         scale=None, enable_gqa=False):
        _STOCHASTIC_CALLS[0] += 1
        return stochastic_scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal,
            scale=scale, nu=nu, enable_gqa=enable_gqa,
        )

    for sampler in samplers:
        sampler.nu = nu
    if original_sdpa is not None:
        F.scaled_dot_product_attention = _stochastic_sdpa
        _set_mha_fastpath(False)
    try:
        yield
    finally:
        if original_sdpa is not None:
            F.scaled_dot_product_attention = original_sdpa
            _set_mha_fastpath(saved_fastpath)
        for sampler, previous in zip(samplers, saved_nu):
            sampler.nu = previous

    if strict and _STOCHASTIC_CALLS[0] == calls_at_entry:
        if original_sdpa is None:
            reason = ("this PyTorch version has no torch.nn.functional.scaled_dot_product_attention (added in "
                      "PyTorch 2.0), so only installed samplers can make attention stochastic")
        else:
            reason = ("the model's attention did not call torch.nn.functional.scaled_dot_product_attention (it "
                      "may compute the softmax itself, use another fused kernel, or be compiled)")
        hint = ("" if samplers else " Install samplers with install_attention_samplers(model, attribute) and "
                "pass the model: use_stochastic_attention(nu, model).")
        raise RuntimeError(
            f"use_stochastic_attention(nu={nu}) made no attention call stochastic, so the model ran "
            f"deterministically: {reason}.{hint}"
        )


# =====================================================================================================
# 3. Choosing nu (Section 4 and Appendix B.1)
#
# For a held-out pair (x, y), let f(x) be the deterministic prediction and f_nu(x) one stochastic pass. nu is
# chosen to match the size of the stochastic deviation, R_nu = ||f_nu(x) - f(x)||, to the size of the
# residual, Z = ||y - f(x)||, by minimizing L(nu) = E[(R_nu - Z)^2] over an integer range. The search is the
# Bayesian optimization of Appendix B.1 (Yadav and Zhang, 2026):
#   - the mean deviation s(nu) of a mini-batch follows a power law, fitted as the Bayesian linear model
#     ln s = a ln nu + ln b + eps z in log-log coordinates;
#   - an initial design, evenly spaced in ln nu, is evaluated on a first mini-batch, which also gives the
#     target size s0 (the mean residual size);
#   - each iteration draws q posterior samples of (a, ln b, eps^2) (Thompson sampling), proposes for each the
#     minimizer of E[(s(nu) - s0)^2], which is (s0 / (b exp(1.5 eps^2)))^(1/a), rounded to the range, and
#     evaluates the proposals on a fresh mini-batch;
#   - the result is the nearest integer, within the range, to that minimizer at the least-squares fit of all
#     evaluations.
# =====================================================================================================

# predict(x, nu) returns a tensor whose first dimension is the batch; nu=None means deterministic.
Predict = Callable[[Any, Optional[int]], torch.Tensor]
Norm = Callable[[torch.Tensor], torch.Tensor]

_MIN_SCALE = 1e-12  # floor for a deviation size before its logarithm is taken


def l2_per_example(diff: torch.Tensor) -> torch.Tensor:
    """Euclidean norm of each example over all non-batch dimensions (the paper's ``||.||_2``)."""
    return torch.linalg.vector_norm(diff.reshape(diff.shape[0], -1).double(), dim=1)


@dataclass
class Evaluation:
    """One evaluation of a candidate nu on one mini-batch (one stochastic pass per example)."""

    nu: int
    scale: float  # mean of ||f_nu(x) - f(x)|| over the batch, the paper's s-hat(nu)
    loss: float   # mean of (||f_nu(x) - f(x)|| - ||y - f(x)||)^2 over the batch, the paper's l-hat(nu)
    batch: int    # 0 for the initial design, k for the k-th iteration


@dataclass
class CalibrationResult:
    """Outcome of ``calibrate_nu``."""

    nu: int                  # the calibrated concentration parameter nu*
    nu_continuous: float     # the surrogate's minimizer before rounding to the range
    target_scale: float      # s0, the mean residual size ||y - f(x)|| on the first batch
    slope: float             # a in ln s = a ln nu + ln b + eps z (close to -1/2 in practice)
    log_intercept: float     # ln b
    noise_var: float         # eps^2
    nu_range: Tuple[int, int]
    evaluations: List[Evaluation] = field(default_factory=list)

    @property
    def at_boundary(self) -> bool:
        """True if the range cut the minimizer off; widen ``nu_range`` and calibrate again."""
        lo, hi = self.nu_range
        return not (lo - 0.5 <= self.nu_continuous <= hi + 0.5)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["at_boundary"] = self.at_boundary
        return out

    def __str__(self) -> str:
        lo, hi = self.nu_range
        visited = sorted({e.nu for e in self.evaluations})
        text = (
            f"nu* = {self.nu} (continuous minimizer {self.nu_continuous:.3f}, range {lo}-{hi}); "
            f"target scale {self.target_scale:.6g}; fit ln s = {self.slope:.4f} ln nu + "
            f"{self.log_intercept:.4f}, eps^2 = {self.noise_var:.3g}; {len(self.evaluations)} evaluations "
            f"at {len(visited)} values of nu"
        )
        if self.at_boundary:
            text += "; the minimizer lies outside the range, so widen nu_range"
        return text


class _BatchStream:
    """Mini-batches from a re-iterable (a DataLoader or a list) or from a function that returns one."""

    def __init__(self, batches: Union[Iterable, Callable[[], Any]]) -> None:
        self._fn = batches if callable(batches) and not hasattr(batches, "__iter__") else None
        self._source = batches
        self._it = None if self._fn is not None else iter(batches)
        self.drawn = 0

    def next(self) -> Tuple[Any, torch.Tensor]:
        if self._fn is not None:
            batch = self._fn()
        else:
            try:
                batch = next(self._it)
            except StopIteration:
                self._it = iter(self._source)
                try:
                    batch = next(self._it)
                except StopIteration:
                    raise RuntimeError(
                        f"the calibration batches ran out after {self.drawn}. Pass a re-iterable such as a "
                        "DataLoader or a list, or a function that returns a batch; a generator can be read "
                        "only once."
                    ) from None
        self.drawn += 1
        if not (isinstance(batch, (tuple, list)) and len(batch) == 2):
            raise TypeError("every calibration batch must be a pair (x, y)")
        return batch[0], batch[1]


def _reference(predict: Predict, norm: Norm, x: Any, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    det = predict(x, None)
    if not torch.is_tensor(y):
        y = torch.as_tensor(y)
    y = y.to(device=det.device, dtype=det.dtype)
    if y.shape != det.shape:
        raise ValueError(f"y has shape {tuple(y.shape)} but predict(x, None) returned {tuple(det.shape)}")
    return det, norm(y - det)


def _evaluate(predict: Predict, norm: Norm, x: Any, det: torch.Tensor, resid: torch.Tensor,
              nu: int, batch: int) -> Evaluation:
    dev = norm(predict(x, nu) - det)
    return Evaluation(nu=int(nu), scale=float(dev.mean()), loss=float(((dev - resid) ** 2).mean()), batch=batch)


def _fit_loglog(evaluations: List[Evaluation]):
    """Least-squares fit of ln s = ln b + a ln nu, with what the posterior under p ~ 1/eps^2 needs."""
    nus = np.array([e.nu for e in evaluations], dtype=float)
    scales = np.array([e.scale for e in evaluations], dtype=float)
    x = np.column_stack([np.ones_like(nus), np.log(nus)])
    y = np.log(np.maximum(scales, _MIN_SCALE))
    xtx = x.T @ x
    theta = np.linalg.solve(xtx, x.T @ y)  # (ln b, a)
    resid = y - x @ theta
    df = len(y) - 2
    s2 = float(resid @ resid / df)
    return theta, s2, df, np.linalg.inv(xtx)


def _minimizer(log_b, a, eps2, s0: float):
    """argmin over nu of E[(b nu^a zeta - s0)^2] with zeta ~ logN(0, eps^2), for any a != 0."""
    log_b, a, eps2 = np.asarray(log_b, float), np.asarray(a, float), np.asarray(eps2, float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        log_nu = (math.log(s0) - log_b - 1.5 * eps2) / a
    return np.exp(np.clip(log_nu, -700.0, 700.0))


def _project(nu, lo: int, hi: int) -> np.ndarray:
    nu = np.nan_to_num(np.asarray(nu, float), nan=float(lo), posinf=float(hi), neginf=float(lo))
    return np.clip(np.rint(nu), lo, hi).astype(int)


def calibrate_nu(
    predict: Predict,
    batches: Union[Iterable, Callable[[], Any]],
    nu_range: Tuple[int, int] = (1, 50),
    *,
    n_initial: int = 40,
    n_candidates: int = 10,
    n_iterations: int = 50,
    norm: Optional[Norm] = None,
    seed: int = 0,
    verbose: bool = False,
) -> CalibrationResult:
    """Calibrate nu on held-out data by the Bayesian optimization of Appendix B.1.

    Args:
        predict: ``predict(x, nu)`` returns the model's prediction for the batch ``x`` as a tensor whose first
            dimension is the batch: deterministic for ``nu=None``, one stochastic forward pass for an integer
            ``nu``. Usually ``with use_stochastic_attention(nu, model): return model(x)``. To calibrate on part
            of the output only (on ClimaX the paper uses the 72-hour Z500 field), return that part.
        batches: Held-out ``(x, y)`` mini-batches not used for training: a DataLoader with ``shuffle=True``, a
            list, or a function that returns a batch. It is read ``n_iterations + 1`` times and restarted when
            exhausted.
        nu_range: The integer search range ``(nu_min, nu_max)``.
        n_initial, n_candidates, n_iterations: The initial design size ``K_0``, the candidates per iteration
            ``q`` and the number of iterations ``K``. The defaults are the paper's ClimaX setting:
            40 + 50 x 10 = 540 evaluations, each one stochastic pass over one mini-batch.
        norm: Maps a ``(batch, ...)`` tensor of differences to per-example sizes; by default the Euclidean norm
            over all non-batch dimensions.
        seed: Seed of the Thompson sampling. The stochastic passes use PyTorch's random number generator, so
            call ``torch.manual_seed`` as well for a reproducible result.
        verbose: Print one line per iteration.

    Returns:
        A ``CalibrationResult``: ``result.nu`` is nu*, ``result.at_boundary`` says whether the range cut the
        minimizer off, and ``result.evaluations`` is the full log, with the loss of every evaluation as a check
        on the chosen value.
    """
    lo, hi = check_nu(nu_range[0]), check_nu(nu_range[1])
    if hi < lo:
        raise ValueError(f"nu_range must be (nu_min, nu_max) with nu_min <= nu_max, got {nu_range}")
    if n_initial < 3:
        raise ValueError("n_initial must be at least 3 so that the log-log fit has a residual variance")
    if n_candidates < 1 or n_iterations < 0:
        raise ValueError("n_candidates must be positive and n_iterations nonnegative")
    norm = norm or l2_per_example
    rng = np.random.default_rng(seed)
    stream = _BatchStream(batches)
    evaluations: List[Evaluation] = []

    with torch.no_grad():
        x0, y0 = stream.next()
        det0, resid0 = _reference(predict, norm, x0, y0)
        s0 = float(resid0.mean())
        if not s0 > 0:
            raise ValueError("the deterministic predictions have no residual on the first calibration batch")
        if lo == hi:
            return CalibrationResult(nu=lo, nu_continuous=float(lo), target_scale=s0, slope=float("nan"),
                                     log_intercept=float("nan"), noise_var=float("nan"), nu_range=(lo, hi))

        design = _project(np.exp(np.linspace(math.log(lo), math.log(hi), n_initial)), lo, hi)
        for i, nu in enumerate(design):
            evaluations.append(_evaluate(predict, norm, x0, det0, resid0, int(nu), batch=0))
            if i == 0 and not evaluations[0].scale > 0:
                raise RuntimeError(
                    f"predict(x, nu={int(nu)}) returned exactly the deterministic prediction, so stochastic "
                    "attention is not active. predict must pass nu to the model, for example "
                    "with use_stochastic_attention(nu, model): return model(x)."
                )
        if verbose:
            print(f"[calibrate_nu] target scale s0 = {s0:.6g}; initial design of {n_initial} values in {lo}-{hi}")

        for k in range(1, n_iterations + 1):
            theta, s2, df, v = _fit_loglog(evaluations)
            sigma2 = s2 * df / rng.chisquare(df, size=n_candidates)
            draws = theta[:, None] + (np.linalg.cholesky(v) @ rng.standard_normal((2, n_candidates))) \
                * np.sqrt(sigma2)[None, :]
            candidates = _project(_minimizer(draws[0], draws[1], sigma2, s0), lo, hi)
            x, y = stream.next()
            det, resid = _reference(predict, norm, x, y)
            for nu in candidates:
                evaluations.append(_evaluate(predict, norm, x, det, resid, int(nu), batch=k))
            if verbose:
                print(f"[calibrate_nu] iteration {k}/{n_iterations}: candidates {candidates.tolist()}, "
                      f"fit minimizer {float(_minimizer(theta[0], theta[1], s2, s0)):.3f}")

    theta, s2, _, _ = _fit_loglog(evaluations)
    log_b, a = float(theta[0]), float(theta[1])
    if not a < 0:
        raise RuntimeError(
            f"the stochastic deviation does not shrink as nu grows (fitted slope a = {a:.4f}; stochastic "
            "attention gives a near -1/2). Check that predict(x, nu) is stochastic for an integer nu and that "
            "the model's attention weights are not one-hot."
        )
    nu_cont = float(_minimizer(log_b, a, s2, s0))
    result = CalibrationResult(nu=int(_project(nu_cont, lo, hi)), nu_continuous=nu_cont, target_scale=s0,
                               slope=a, log_intercept=log_b, noise_var=s2, nu_range=(lo, hi),
                               evaluations=evaluations)
    if verbose:
        print(f"[calibrate_nu] {result}")
    return result


# =====================================================================================================
# 4. The predictive ensemble
# =====================================================================================================

def predictive_samples(predict: Predict, x: Any, nu: int, num_samples: int = 19) -> torch.Tensor:
    """Run ``num_samples`` stochastic forward passes of ``predict(x, nu)`` and stack them along a new first
    dimension, giving shape ``(num_samples, *prediction shape)``. The passes are independent, so for a small
    model one pass over the batch repeated ``num_samples`` times gives the same result faster."""
    nu = check_nu(nu)
    if num_samples < 2:
        raise ValueError("num_samples must be at least 2")
    with torch.no_grad():
        return torch.stack([predict(x, nu) for _ in range(num_samples)], dim=0)


# =====================================================================================================
# 5. Calibration and sharpness metrics (Section 5.2)
#
# Every function takes samples of shape (S, *shape) and, where needed, outcomes y of shape `shape`; the optional
# weights broadcast to `shape` (on ClimaX the paper weights every grid point by the cosine of its latitude).
# Inputs may be NumPy arrays or tensors. For data too large to hold at once, sum rank_histogram over chunks and
# pass the total to w1_from_histogram.
# =====================================================================================================

def _np(a) -> np.ndarray:
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().double().numpy()
    return np.asarray(a, dtype=float)


def _split(samples, y=None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    s = _np(samples)
    if s.ndim < 1 or s.shape[0] < 2:
        raise ValueError("samples must have shape (S, ...) with S >= 2")
    if y is None:
        return s, None
    t = _np(y)
    if t.shape != s.shape[1:]:
        raise ValueError(f"y has shape {t.shape} but samples have shape {s.shape} (expected (S, *y.shape))")
    return s, t


def _weights(weights, shape) -> np.ndarray:
    if weights is None:
        return np.ones(shape)
    w = np.broadcast_to(_np(weights), shape)
    if (w < 0).any() or not w.sum() > 0:
        raise ValueError("weights must be nonnegative with a positive sum")
    return w


def ranks(samples, y, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Rank of each outcome among its S samples, the number of samples below it (0 to S). Ties are broken
    uniformly at random, which matters only for discrete outputs."""
    s, t = _split(samples, y)
    below = (s < t).sum(axis=0)
    ties = (s == t).sum(axis=0)
    if ties.any():
        rng = rng or np.random.default_rng(0)
        below = below + np.floor(rng.random(ties.shape) * (ties + 1)).astype(below.dtype)
    return below


def rank_histogram(samples, y, weights=None, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Weighted count of each rank 0..S, an array of length S + 1 (the PIT histogram of an ensemble)."""
    s, t = _split(samples, y)
    r = ranks(s, t, rng)
    w = _weights(weights, t.shape)
    return np.bincount(r.ravel(), weights=w.ravel(), minlength=s.shape[0] + 1).astype(float)


def w1_from_histogram(hist) -> float:
    """W1 distance between the rank distribution and the uniform one, as in the paper:
    ``W1 = (1/(S+1)) sum_b |G_b - b/(S+1)|`` with ``G_b`` the fraction of ranks below ``b``. Zero is calibrated."""
    c = np.asarray(hist, dtype=float)
    c = c / c.sum()
    n = c.size
    return float(np.abs(np.cumsum(c) - np.arange(1, n + 1) / n).sum() / n)


def pit_w1(samples, y, weights=None, rng: Optional[np.random.Generator] = None) -> float:
    """W1 of the rank (PIT) histogram from uniform. A U shape means too narrow, a hump too wide."""
    return w1_from_histogram(rank_histogram(samples, y, weights, rng))


def nominal_coverage(num_samples: int, j: int = 1) -> float:
    """Exact coverage ``(S + 1 - 2j) / (S + 1)`` of the interval from the j-th smallest to the j-th largest of S
    exchangeable samples: 0.9 for [min, max] of 19 samples, and for the 5th to 95th of 99."""
    if not 1 <= j < (num_samples + 1) / 2:
        raise ValueError(f"j must satisfy 1 <= j < (S + 1) / 2, got j={j} for S={num_samples}")
    return (num_samples + 1 - 2 * j) / (num_samples + 1)


def interval(samples, j: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Lower and upper ends, the j-th smallest and the j-th largest sample at every output."""
    s, _ = _split(samples)
    nominal_coverage(s.shape[0], j)
    srt = np.sort(s, axis=0)
    return srt[j - 1], srt[s.shape[0] - j]


def coverage(samples, y, j: int = 1, weights=None) -> float:
    """Weighted fraction of outcomes inside ``interval(samples, j)``; compare with ``nominal_coverage``."""
    s, t = _split(samples, y)
    lower, upper = interval(s, j)
    w = _weights(weights, t.shape)
    return float((w * ((t >= lower) & (t <= upper))).sum() / w.sum())


def mpiw(samples, j: int = 1, weights=None) -> float:
    """Mean prediction-interval width of ``interval(samples, j)``, weighted."""
    s, _ = _split(samples)
    lower, upper = interval(s, j)
    w = _weights(weights, lower.shape)
    return float((w * (upper - lower)).sum() / w.sum())


def spread(samples, weights=None) -> float:
    """Ensemble spread, the square root of the weighted mean over outputs of the sample variance (ddof=1)."""
    s, _ = _split(samples)
    var = s.var(axis=0, ddof=1)
    w = _weights(weights, var.shape)
    return float(np.sqrt((w * var).sum() / w.sum()))


def crps(samples, y, weights=None, fair: bool = True) -> float:
    """Mean CRPS of the ensemble. ``fair=True`` gives the unbiased estimator
    ``mean|X - y| - sum|X_s - X_t| / (2 S (S - 1))``; the standard estimator, which divides by ``2 S^2``,
    favours ensembles that are too narrow when S is small."""
    s, t = _split(samples, y)
    n = s.shape[0]
    first = np.abs(s - t).mean(axis=0)
    srt = np.sort(s, axis=0)
    coef = (2 * np.arange(1, n + 1) - n - 1).reshape((n,) + (1,) * (s.ndim - 1))
    pair_sum = 2.0 * (coef * srt).sum(axis=0)  # sum over all ordered pairs of |X_s - X_t|
    second = pair_sum / (2 * n * (n - 1)) if fair else pair_sum / (2 * n * n)
    w = _weights(weights, t.shape)
    return float((w * (first - second)).sum() / w.sum())
