"""Tests for stochastic_attention.py; run with `pytest` from this folder.

They check the sampled weights against the paper's Definition 3.1 and Propositions 3.1-3.2, the stochastic
attention against torch.nn.functional.scaled_dot_product_attention, that switching SA on leaves the
deterministic model unchanged and fails loudly when the wiring is wrong, the nu calibration on a problem with a
known answer, and the metrics against closed-form values. The timm and Hugging Face tests run when those
libraries are installed.
"""

import math

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import stochastic_attention as sa

HAS_SDPA = hasattr(F, "scaled_dot_product_attention")
needs_sdpa = pytest.mark.skipif(not HAS_SDPA, reason="needs torch.nn.functional.scaled_dot_product_attention")


# ----------------------------------------------------------------------------------------------------
# The sampled weights and the stochastic attention
# ----------------------------------------------------------------------------------------------------

def _probs(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.softmax(torch.randn(*shape, generator=g) * 2.0, dim=-1)


def test_weights_lie_on_the_simplex_and_on_the_1_over_nu_lattice():
    torch.manual_seed(0)
    nu = 7
    w = sa.sample_attention_weights(_probs(2, 3, 5, 11), nu)
    assert w.shape == (2, 3, 5, 11)
    assert torch.all(w >= 0)
    assert torch.allclose(w.sum(-1), torch.ones(2, 3, 5))
    assert torch.allclose(w * nu, torch.round(w * nu), atol=1e-5)


def test_mean_and_covariance_match_propositions_3_1_and_3_2():
    """E[pi~] = pi, Cov(pi~) = (diag pi - pi pi^T) / nu; and the same pushed through V."""
    torch.manual_seed(1)
    nu, k, n = 5, 6, 400_000
    pi = _probs(k, seed=3).double()
    v = torch.randn(k, 3, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    w = sa.sample_attention_weights(pi.expand(n, k), nu).double()

    cov_theory = (torch.diag(pi) - torch.outer(pi, pi)) / nu
    se_mean = (cov_theory.diagonal() / n).sqrt()
    assert torch.all((w.mean(0) - pi).abs() < 5 * se_mean + 1e-12)
    assert torch.allclose(torch.cov(w.T), cov_theory, atol=2e-4)

    out = w @ v
    out_cov_theory = v.T @ cov_theory @ v
    assert torch.allclose(out.mean(0), pi @ v, atol=5 * float(out_cov_theory.diagonal().max().sqrt()) / n ** 0.5)
    assert torch.allclose(torch.cov(out.T), out_cov_theory, rtol=0.02, atol=1e-4)


def test_shape_term_is_the_covariance_of_one_sampled_value_vector():
    """V^T (diag pi - pi pi^T) V = sum_j pi_j (v_j - o)^T (v_j - o): the paper's 'shape' at nu = 1."""
    pi = _probs(9, seed=5).double()
    v = torch.randn(9, 4, generator=torch.Generator().manual_seed(6), dtype=torch.float64)
    o = pi @ v
    shape = v.T @ (torch.diag(pi) - torch.outer(pi, pi)) @ v
    direct = sum(pi[j] * torch.outer(v[j] - o, v[j] - o) for j in range(9))
    assert torch.allclose(shape, direct)


def test_large_nu_recovers_the_deterministic_weights():
    torch.manual_seed(2)
    p = _probs(4, 8, seed=7)
    w = sa.sample_attention_weights(p, 2_000_000)  # drawn in chunks, memory stays small
    assert (w - p).abs().max() < 2e-3


def test_rows_that_are_not_distributions_pass_through_unchanged():
    torch.manual_seed(3)
    p = _probs(4, 6, seed=8)
    p[1] = float("nan")  # fully masked query with -inf scores
    p[2] = 0.0           # all-zero row
    w = sa.sample_attention_weights(p, 3)
    assert torch.isnan(w[1]).all()
    assert torch.all(w[2] == 0)
    assert torch.allclose(w[[0, 3]].sum(-1), torch.ones(2))


def test_float64_weights_are_exact_multiples_of_1_over_nu():
    w = sa.sample_attention_weights(_probs(50, 9, seed=10).double(), 3)
    assert w.dtype == torch.float64
    assert (w * 3 - torch.round(w * 3)).abs().max() < 1e-12


def test_dtype_and_generator_are_respected():
    p = _probs(3, 5, seed=9).to(torch.float16)
    a = sa.sample_attention_weights(p, 4, generator=torch.Generator().manual_seed(11))
    b = sa.sample_attention_weights(p, 4, generator=torch.Generator().manual_seed(11))
    assert a.dtype == torch.float16
    assert torch.equal(a, b)


@pytest.mark.parametrize("bad", [0, -2, 2.5, True, "3", None])
def test_nu_must_be_a_positive_integer(bad):
    with pytest.raises((TypeError, ValueError)):
        sa.check_nu(bad)


def _sdpa_cases():
    g = torch.Generator().manual_seed(0)

    def r(*s):
        return torch.randn(*s, generator=g, dtype=torch.float64)

    q, k, v = r(2, 3, 5, 8), r(2, 3, 7, 8), r(2, 3, 7, 6)
    bool_mask = torch.rand(2, 1, 5, 7, generator=g) > 0.3
    bool_mask[..., 0] = True  # every query keeps at least one key
    sq = r(2, 3, 6, 8)
    return [
        ("plain", (q, k, v), {}),
        ("bool mask", (q, k, v), {"attn_mask": bool_mask}),
        ("float mask", (q, k, v), {"attn_mask": r(2, 3, 5, 7)}),
        ("causal", (sq, sq, r(2, 3, 6, 6)), {"is_causal": True}),
        ("scale", (q, k, v), {"scale": 0.37}),
        ("no head dim", (q[0], k[0], v[0]), {}),
    ]


@needs_sdpa
@pytest.mark.parametrize("name,args,kwargs", _sdpa_cases(), ids=[c[0] for c in _sdpa_cases()])
def test_nu_none_equals_torch_sdpa(name, args, kwargs):
    expected = F.scaled_dot_product_attention(*args, **kwargs)
    got = sa.stochastic_scaled_dot_product_attention(*args, **kwargs, nu=None)
    assert torch.allclose(got, expected, atol=1e-10, rtol=1e-10), name


def test_grouped_query_attention_matches_repeated_heads():
    g = torch.Generator().manual_seed(1)
    q = torch.randn(2, 6, 5, 8, generator=g, dtype=torch.float64)
    k = torch.randn(2, 2, 7, 8, generator=g, dtype=torch.float64)
    v = torch.randn(2, 2, 7, 4, generator=g, dtype=torch.float64)
    gqa = sa.stochastic_scaled_dot_product_attention(q, k, v, nu=None, enable_gqa=True)
    rep = sa.stochastic_scaled_dot_product_attention(q, k.repeat_interleave(3, 1), v.repeat_interleave(3, 1),
                                                     nu=None)
    assert torch.allclose(gqa, rep)


def test_stochastic_attention_output_averages_to_the_deterministic_output():
    """Proposition 3.1 for one attention layer: E[pi~ V] = pi V."""
    torch.manual_seed(4)
    g = torch.Generator().manual_seed(2)
    q, k, v = (torch.randn(1, 2, 3, 4, generator=g, dtype=torch.float64) for _ in range(3))
    det = sa.stochastic_scaled_dot_product_attention(q, k, v, nu=None)
    n = 20_000
    draws = sa.stochastic_scaled_dot_product_attention(q.expand(n, -1, -1, -1), k.expand(n, -1, -1, -1),
                                                       v.expand(n, -1, -1, -1), nu=3)
    se = draws.std(0) / n ** 0.5
    assert torch.all((draws.mean(0) - det).abs() < 5 * se + 1e-9)
    assert draws.std(0).max() > 0.01  # and it is genuinely random


def test_causal_mask_together_with_attn_mask_is_rejected():
    q = torch.randn(1, 1, 3, 4)
    with pytest.raises(ValueError):
        sa.stochastic_scaled_dot_product_attention(q, q, q, attn_mask=torch.ones(3, 3, dtype=torch.bool),
                                                   is_causal=True)


# ----------------------------------------------------------------------------------------------------
# Switching SA on in models
# ----------------------------------------------------------------------------------------------------

class ExplicitSoftmaxAttention(nn.Module):
    """The ViT attention block of timm 0.6, which ClimaX uses: softmax, then attn_drop, then @ v."""

    def __init__(self, dim=16, heads=4):
        super().__init__()
        self.heads, self.scale = heads, (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.attn_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, x):
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        attn = self.attn_drop(attn)
        return self.proj_drop(self.proj((attn @ v).transpose(1, 2).reshape(b, n, c)))


def _encoder(seed=0):
    torch.manual_seed(seed)
    layer = nn.TransformerEncoderLayer(16, 4, dim_feedforward=32, dropout=0.0, batch_first=True)
    return nn.TransformerEncoder(layer, 2, enable_nested_tensor=False).eval()


@needs_sdpa
def test_context_makes_a_torch_transformer_encoder_stochastic_and_restores_it():
    model = _encoder()
    x = torch.randn(3, 5, 16)
    original = F.scaled_dot_product_attention
    with torch.no_grad():  # eval + no_grad + batch_first is exactly when PyTorch's fast path would run
        det = model(x)
        with sa.use_stochastic_attention(4):
            a, b = model(x), model(x)
        after = model(x)
    assert not torch.allclose(a, b)
    assert not torch.allclose(a, det)
    assert torch.equal(after, det)
    assert F.scaled_dot_product_attention is original


@needs_sdpa
def test_nu_none_leaves_the_model_deterministic():
    model = _encoder()
    x = torch.randn(2, 4, 16)
    with torch.no_grad():
        det = model(x)
        with sa.use_stochastic_attention(None):
            same = model(x)
    assert torch.equal(same, det)


@needs_sdpa
def test_restored_when_the_block_raises():
    original = F.scaled_dot_product_attention
    with pytest.raises(KeyError):
        with sa.use_stochastic_attention(3, strict=False):
            raise KeyError("boom")
    assert F.scaled_dot_product_attention is original


def test_strict_check_fires_when_no_attention_call_was_made_stochastic():
    """Negative test of the guard: a model without attention runs deterministically inside the block."""
    model = nn.Linear(4, 2)
    with pytest.raises(RuntimeError, match="made no attention call stochastic"):
        with sa.use_stochastic_attention(5):
            model(torch.randn(3, 4))
    with sa.use_stochastic_attention(5, strict=False):  # the check can be switched off
        model(torch.randn(3, 4))


def test_samplers_on_explicit_softmax_attention():
    torch.manual_seed(0)
    reference = ExplicitSoftmaxAttention().eval()
    model = ExplicitSoftmaxAttention().eval()
    model.load_state_dict(reference.state_dict())
    assert sa.install_attention_samplers(model, "attn_drop") == ["attn_drop"]
    x = torch.randn(2, 6, 16)
    with torch.no_grad():
        assert torch.equal(model(x), reference(x))  # samplers off: deterministic path unchanged
        with sa.use_stochastic_attention(6, model):
            a, b = model(x), model(x)
        assert torch.equal(model(x), reference(x))  # and off again after the block
    assert not torch.allclose(a, b)
    assert all(s.nu is None for s in sa.attention_samplers(model))


def test_nested_blocks_restore_the_outer_nu_and_count_inner_calls():
    torch.manual_seed(2)
    model = ExplicitSoftmaxAttention().eval()
    sa.install_attention_samplers(model)
    x = torch.randn(2, 6, 16)
    with torch.no_grad(), sa.use_stochastic_attention(4, model):  # all calls happen in the inner block
        with sa.use_stochastic_attention(9, model):
            assert all(s.nu == 9 for s in sa.attention_samplers(model))
            model(x)
        assert all(s.nu == 4 for s in sa.attention_samplers(model))
    assert all(s.nu is None for s in sa.attention_samplers(model))


def test_samplers_average_to_the_deterministic_layer():
    torch.manual_seed(1)
    model = ExplicitSoftmaxAttention().eval().double()
    sa.install_attention_samplers(model)
    x = torch.randn(1, 5, 16, dtype=torch.float64)
    n = 20_000
    with torch.no_grad():
        det = model(x)
        with sa.use_stochastic_attention(4, model):
            draws = model(x.expand(n, -1, -1))
    se = draws.std(0) / n ** 0.5
    assert torch.all((draws.mean(0) - det[0]).abs() < 5 * se + 1e-9)


def test_install_fails_with_a_description_when_the_attribute_is_absent():
    """Negative test of the installer's guard."""
    with pytest.raises(ValueError, match="has an attribute 'attn_drop'"):
        sa.install_attention_samplers(nn.Sequential(nn.Linear(3, 3)), "attn_drop")


def test_sampler_rejects_a_tensor_that_is_not_attention_probabilities():
    """Negative test: a sampler installed on a hidden-state dropout must not silently resample activations."""
    model = ExplicitSoftmaxAttention().eval()
    sa.install_attention_samplers(model, "proj_drop")  # wrong attribute: applied after the projection
    with pytest.raises(RuntimeError, match="does not hold attention probabilities"):
        with torch.no_grad(), sa.use_stochastic_attention(4, model):
            model(torch.randn(2, 6, 16))


def test_install_fills_an_attention_dropout_that_is_none():
    """FT-Transformer style: `self.dropout = nn.Dropout(p) if p else None`, applied only if not None."""

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = None

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = Attn()
            self.head_dropout = nn.Dropout(0.1)

    net = Net()
    assert sa.install_attention_samplers(net, "dropout", module_type=Attn) == ["attention.dropout"]
    assert isinstance(net.attention.dropout, sa.AttentionSampler)
    assert isinstance(net.head_dropout, nn.Dropout)


def _check_model(model, run):
    """Deterministic outside the block, stochastic inside it, unchanged afterwards."""
    with torch.no_grad():
        det = run()
        with sa.use_stochastic_attention(4, model):
            a, b = run(), run()
        after = run()
    assert not torch.allclose(a, b)
    assert torch.equal(after, det)


def test_timm_vision_transformer():
    pytest.importorskip("timm")
    from timm.models.vision_transformer import VisionTransformer

    torch.manual_seed(0)
    model = VisionTransformer(img_size=32, patch_size=8, num_classes=3, embed_dim=32, depth=2, num_heads=4).eval()
    if not any(getattr(m, "fused_attn", False) for m in model.modules()):
        # timm < 0.9 (the version ClimaX uses) computes the softmax itself and applies attn_drop to it
        assert len(sa.install_attention_samplers(model, "attn_drop")) == 2
    x = torch.randn(2, 3, 32, 32)
    _check_model(model, lambda: model(x))


@needs_sdpa
def test_huggingface_model_with_sdpa_attention():
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    torch.manual_seed(0)
    model = transformers.AutoModel.from_config(config, attn_implementation="sdpa").eval()
    ids = torch.randint(0, 64, (2, 7))
    _check_model(model, lambda: model(input_ids=ids).last_hidden_state)


# ----------------------------------------------------------------------------------------------------
# Calibrating nu
# ----------------------------------------------------------------------------------------------------

def _toy(sigma, tau, dim=64, n=2048, batch=128, seed=0):
    """Deterministic prediction 0; one stochastic pass is (sigma / sqrt(nu)) * N(0, I); y ~ tau * N(0, I).

    The mean deviation is sigma E||z|| / sqrt(nu) and the mean residual tau E||z||, so the deviation matches
    the residual at nu = (sigma / tau)^2."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, dim, generator=g)  # x only carries the batch size here
    y = tau * torch.randn(n, dim, generator=g)
    batches = [(x[i:i + batch], y[i:i + batch]) for i in range(0, n, batch)]

    def predict(xb, nu=None):
        if nu is None:
            return torch.zeros_like(xb)
        return sigma / nu ** 0.5 * torch.randn_like(xb)

    return predict, batches


def test_calibration_recovers_the_known_nu():
    torch.manual_seed(0)
    predict, batches = _toy(sigma=4.0, tau=1.0)
    result = sa.calibrate_nu(predict, batches, nu_range=(1, 200), n_initial=20, n_candidates=5, n_iterations=20)
    assert 15 <= result.nu <= 17
    assert abs(result.slope + 0.5) < 0.02
    assert not result.at_boundary
    assert len(result.evaluations) == 20 + 20 * 5


def test_calibration_default_budget_is_the_papers_540_evaluations():
    torch.manual_seed(1)
    predict, batches = _toy(sigma=3.0, tau=1.0)
    result = sa.calibrate_nu(predict, batches)  # nu_range (1, 50), K0 = 40, q = 10, K = 50
    assert len(result.evaluations) == 540
    assert 8 <= result.nu <= 10


def test_calibration_flags_a_minimizer_outside_the_range():
    torch.manual_seed(2)
    predict, batches = _toy(sigma=10.0, tau=1.0)  # wants nu = 100
    result = sa.calibrate_nu(predict, batches, nu_range=(1, 30), n_initial=10, n_candidates=3, n_iterations=5)
    assert result.nu == 30
    assert result.at_boundary
    assert "widen nu_range" in str(result)


def test_calibration_guard_fires_when_predict_ignores_nu():
    """Negative test: predict never turns stochastic attention on."""
    _, batches = _toy(sigma=1.0, tau=1.0)
    with pytest.raises(RuntimeError, match="not active"):
        sa.calibrate_nu(lambda xb, nu=None: torch.zeros_like(xb), batches, n_initial=5, n_iterations=1)


def test_calibration_one_shot_generator_runs_out_with_a_description():
    predict, batches = _toy(sigma=2.0, tau=1.0)
    with pytest.raises(RuntimeError, match="ran out"):
        sa.calibrate_nu(predict, (b for b in batches[:3]), n_initial=5, n_candidates=2, n_iterations=5)


def test_calibration_reports_a_shape_mismatch_between_y_and_the_prediction():
    predict, batches = _toy(sigma=2.0, tau=1.0)
    bad = [(x, y[:, :1]) for x, y in batches]
    with pytest.raises(ValueError, match="shape"):
        sa.calibrate_nu(predict, bad, n_initial=5, n_iterations=1)


def test_calibration_end_to_end_on_a_model_with_stochastic_attention():
    torch.manual_seed(3)

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(4, 12)
            self.attn_drop = nn.Identity()
            self.out = nn.Linear(4 * 6, 1)

        def forward(self, x):  # x: (B, 6, 4) tokens
            q, k, v = self.qkv(x).chunk(3, dim=-1)
            w = (q @ k.transpose(-2, -1) / 2.0).softmax(-1)
            return self.out((self.attn_drop(w) @ v).flatten(1)).squeeze(-1)

    net = Tiny().eval()
    sa.install_attention_samplers(net)
    x = torch.randn(512, 6, 4)
    with torch.no_grad():
        y = net(x) + 0.05 * torch.randn(512)
    batches = [(x[i:i + 64], y[i:i + 64]) for i in range(0, 512, 64)]

    def predict(xb, nu=None):
        with sa.use_stochastic_attention(nu, net):
            return net(xb)

    result = sa.calibrate_nu(predict, batches, nu_range=(1, 500), n_initial=12, n_candidates=4, n_iterations=10)
    assert 1 <= result.nu <= 500
    assert result.slope < -0.3


# ----------------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------------

def _gaussian_case(n=20_000, s=19, width=1.0, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(n)
    samples = width * rng.standard_normal((s, n))
    return samples, y


def test_w1_is_zero_for_a_flat_histogram_and_matches_a_hand_computation():
    assert sa.w1_from_histogram(np.ones(20)) == pytest.approx(0.0, abs=1e-15)
    # all mass in the lowest rank of 3 bins: G = (1, 1, 1) vs (1/3, 2/3, 1) -> (2/3 + 1/3 + 0) / 3
    assert sa.w1_from_histogram([1, 0, 0]) == pytest.approx(1 / 3)


def test_w1_separates_calibrated_from_under_dispersed_ensembles():
    assert sa.pit_w1(*_gaussian_case()) < 0.01
    assert sa.pit_w1(*_gaussian_case(width=0.5)) > 0.05
    hist = sa.rank_histogram(*_gaussian_case(width=0.5))
    assert hist[0] > 2 * hist[10] and hist[-1] > 2 * hist[10]  # U shape


def test_order_statistic_intervals_have_their_exact_coverage():
    assert sa.nominal_coverage(19, 1) == pytest.approx(0.9)
    assert sa.nominal_coverage(99, 5) == pytest.approx(0.9)
    samples, y = _gaussian_case(n=40_000)
    assert sa.coverage(samples, y) == pytest.approx(0.9, abs=0.006)
    samples99, y99 = _gaussian_case(n=20_000, s=99, seed=1)
    assert sa.coverage(samples99, y99, j=5) == pytest.approx(0.9, abs=0.008)
    with pytest.raises(ValueError):
        sa.interval(samples, j=10)


def test_fair_crps_is_unbiased_and_the_standard_estimator_is_not():
    """For y ~ N(0, 1) and a N(0, 1) forecast, E[CRPS] = E|X - y| - E|X - X'| / 2 = 2/sqrt(pi) - 1/sqrt(pi).

    The standard estimator keeps the S zero diagonal pairs, so its spread term is (S - 1)/S of the fair one and
    its expectation is (1 + 1/S)/sqrt(pi): too high for a calibrated ensemble, lower for a narrower one."""
    s = 5
    samples, y = _gaussian_case(n=200_000, s=s, seed=2)
    exact = 1.0 / math.sqrt(math.pi)
    assert sa.crps(samples, y, fair=True) == pytest.approx(exact, rel=0.01)
    assert sa.crps(samples, y, fair=False) == pytest.approx((1 + 1 / s) * exact, rel=0.01)


def test_crps_of_sorted_formula_matches_the_pairwise_definition():
    rng = np.random.default_rng(3)
    s = rng.standard_normal((7, 50))
    y = rng.standard_normal(50)
    pair = np.abs(s[:, None, :] - s[None, :, :]).sum(axis=(0, 1))
    direct = (np.abs(s - y).mean(0) - pair / (2 * 7 * 6)).mean()
    assert sa.crps(s, y) == pytest.approx(direct)


def test_spread_and_mpiw():
    samples, _ = _gaussian_case(n=50_000, width=2.0, seed=4)
    assert sa.spread(samples) == pytest.approx(2.0, rel=0.01)
    assert sa.mpiw(np.array([[0.0, 1.0], [2.0, 5.0], [1.0, 3.0]])) == pytest.approx((2.0 + 4.0) / 2)


def test_weights_select_and_rescale():
    samples, y = _gaussian_case(n=1000, seed=5)
    w = np.zeros(1000)
    w[:500] = 3.0
    assert sa.coverage(samples, y, weights=w) == pytest.approx(sa.coverage(samples[:, :500], y[:500]))
    assert sa.pit_w1(samples, y, weights=np.full(1000, 7.0)) == pytest.approx(sa.pit_w1(samples, y))


def test_metrics_accept_tensors_and_check_shapes():
    samples, y = _gaussian_case(n=100, seed=6)
    assert sa.crps(torch.from_numpy(samples), torch.from_numpy(y)) == pytest.approx(sa.crps(samples, y))
    with pytest.raises(ValueError, match="shape"):
        sa.coverage(samples, y[:50])


def test_ties_are_broken_at_random():
    hist = sa.rank_histogram(np.zeros((19, 10_000)), np.zeros(10_000))
    assert hist.min() > 0.8 * hist.mean()  # all-tied outcomes spread evenly over the 20 ranks
