import numpy as np
import pytest

from prmp_demo.config import load_config
from prmp_demo.gamma import compute_metrics_prefix
from prmp_demo.pipeline.grassmann import grassmann_similarity, normalize_grassmann_score
from prmp_demo.pipeline.koopman_edmd import edmd_overlap_at_end
from prmp_demo.pipeline.preprocess import joint_pca_projection, trajectory_arrays


def test_joint_pca_shapes():
    rng = np.random.RandomState(0)
    Xa = rng.standard_normal((12, 8))
    Xb = rng.standard_normal((12, 8))
    Za, Zb, meta = joint_pca_projection(Xa, Xb, k=4)
    assert Za.shape == (12, 4)
    assert Zb.shape == (12, 4)
    assert meta.get("used_pca") is True


def test_grassmann_similarity_range():
    rng = np.random.RandomState(1)
    Za = rng.standard_normal((10, 6))
    Zb = Za + rng.standard_normal((10, 6)) * 0.05
    score = grassmann_similarity(Za, Zb, window=8, r=3)
    assert not np.isnan(score)
    norm = normalize_grassmann_score(score, 3)
    assert 0.0 <= norm <= 1.0


def test_edmd_overlap_finite():
    rng = np.random.RandomState(3)
    Za = rng.standard_normal((10, 5))
    Zb = Za + rng.standard_normal((10, 5)) * 0.02
    val = edmd_overlap_at_end(Za, Zb, window=8, rank=6)
    assert not np.isnan(val)
    assert 0.0 <= val <= 1.0


def test_gamma_theory_order_and_length():
    cfg = load_config()
    emb = np.random.RandomState(2).standard_normal(16).tolist()
    hist_a = []
    hist_b = []
    for _ in range(8):
        hist_a.append(list(emb))
        hist_b.append(list(emb))
        m = compute_metrics_prefix(hist_a, hist_b, cfg=cfg)
        assert m["rho_text"] > 0.99
        g = m["gamma"]
        assert len(g) == 3
        assert m["gamma_labels"] == ["R_spec_norm", "W_p_alignment_norm", "Grassmann_norm"]


def test_parse_task_protocol_output():
    from prmp_demo.environment.task_protocol import parse_model_output

    text = '{"action_id":"EXPLORE","rationale":"hello"}'
    pa, ok = parse_model_output(text)
    assert ok and pa.action_id == "EXPLORE"


@pytest.mark.optional
def test_tda_wasserstein_when_full_installed():
    pytest.importorskip("ripser")
    pytest.importorskip("persim")
    rng = np.random.RandomState(11)
    Za = rng.standard_normal((12, 4))
    Zb = Za + rng.standard_normal((12, 4)) * 0.15

    from prmp_demo.pipeline.tda_optional import persistence_wasserstein_h0

    dist, reason, _ = persistence_wasserstein_h0(Za, Zb, window=8)
    assert reason is None
    assert dist == dist  # not nan
    assert dist >= 0.0

