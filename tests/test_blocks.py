"""Tests of the isotonic blocks (Venn categories) against their definition."""

import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression

from fast_ivap import IVAP
from fast_ivap._api import _core

BACKENDS = [
    "python",
    pytest.param("cpp", marks=pytest.mark.skipif(_core is None, reason="extension not built")),
]


def blocks_by_definition(scores, labels, test_scores):
    """Boolean masks (2, m, n): calibration items whose fitted value equals the test item's."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    masks = np.zeros((2, len(test_scores), len(scores)), dtype=bool)
    for j, t in enumerate(test_scores):
        for e in (0, 1):
            x = np.r_[scores, t]
            f = IsotonicRegression().fit(x, np.r_[labels, e]).predict(x)
            masks[e, j] = np.isclose(f[:-1], f[-1], rtol=0, atol=1e-12)
    return masks


def assert_blocks_match_definition(backend, scores, labels, test_scores):
    ivap = IVAP(backend=backend).fit(scores, labels)
    lo0, hi0, lo1, hi1 = ivap.predict_blocks(test_scores)
    expected = blocks_by_definition(scores, labels, test_scores)
    for e, (lo, hi) in enumerate(((lo0, hi0), (lo1, hi1))):
        for j in range(len(test_scores)):
            got = np.zeros(len(scores), dtype=bool)
            got[ivap.order_[lo[j] : hi[j]]] = True
            assert np.array_equal(got, expected[e, j]), (e, test_scores[j])

    # The identity p_e = (sum of labels in the block + e) / (block size + 1).
    p0, p1 = ivap.predict_p0_p1(test_scores)
    s0, s1 = ivap.block_sums(test_scores, np.asarray(labels, dtype=float))
    np.testing.assert_allclose(p0, s0 / (hi0 - lo0 + 1), rtol=0, atol=1e-15)
    np.testing.assert_allclose(p1, (s1 + 1) / (hi1 - lo1 + 1), rtol=0, atol=1e-15)


@pytest.mark.parametrize("backend", BACKENDS)
class TestAgainstDefinition:
    def test_continuous_scores(self, backend):
        rng = np.random.default_rng(0)
        scores = rng.normal(size=60)
        labels = rng.random(60) < 1 / (1 + np.exp(-2 * scores))
        test = np.r_[np.linspace(-3, 3, 41), scores[:10], -10.0, 10.0]
        assert_blocks_match_definition(backend, scores, labels, test)

    @pytest.mark.parametrize("seed", range(30))
    def test_heavy_ties(self, backend, seed):
        # About 8 distinct scores: many ties, and test scores equal to them.
        rng = np.random.default_rng(seed)
        n = int(rng.integers(1, 50))
        scores = rng.integers(0, 8, n).astype(float)
        labels = rng.integers(0, 2, n)
        test = np.arange(-1, 9, 0.5)
        assert_blocks_match_definition(backend, scores, labels, test)

    @pytest.mark.parametrize("label", [0, 1])
    def test_constant_labels(self, backend, label):
        scores = [1.0, 2, 2, 3, 4]
        assert_blocks_match_definition(backend, scores, [label] * 5, [0.0, 2.0, 2.5, 4.0, 5.0])

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_tiny(self, backend, n):
        for labels in np.ndindex(*(2,) * n):
            scores = np.arange(n, dtype=float)
            assert_blocks_match_definition(backend, scores, labels, np.arange(-1, n + 0.5, 0.5))
            assert_blocks_match_definition(backend, np.zeros(n), labels, [-1.0, 0.0, 1.0])

    def test_boolean_labels(self, backend):
        scores = [0.3, 0.1, 0.4, 0.1, 0.5, 0.9, 0.2]
        labels = np.array([True, False, True, True, False, True, False])
        assert_blocks_match_definition(backend, scores, labels, [0.0, 0.1, 0.35, 0.5, 1.0])


@pytest.mark.parametrize("backend", BACKENDS)
def test_order_is_stable_argsort(backend):
    scores = [3.0, 1.0, 2.0, 1.0, 3.0, 2.0]
    ivap = IVAP(backend=backend).fit(scores, [0, 1, 0, 1, 1, 0])
    np.testing.assert_array_equal(ivap.order_, np.argsort(scores, kind="stable"))
    assert ivap.order_.dtype == np.int64


@pytest.mark.parametrize("backend", BACKENDS)
def test_nan_test_score_gives_empty_marker(backend):
    ivap = IVAP(backend=backend).fit([1.0, 2.0, 3.0], [0, 1, 1])
    blocks = ivap.predict_blocks([np.nan, 2.0])
    for b in blocks:
        assert b[0] == -1 and b[1] >= 0
        assert b.dtype == np.int64
    s0, s1 = ivap.block_sums([np.nan, 2.0], np.eye(3))
    assert np.isnan(s0[0]).all() and np.isnan(s1[0]).all()
    assert not np.isnan(s0[1]).any() and not np.isnan(s1[1]).any()


@pytest.mark.parametrize("backend", BACKENDS)
def test_block_sums_matrix_against_loop(backend):
    rng = np.random.default_rng(3)
    n, d = 300, 4
    scores = rng.normal(size=n).round(1)
    classes = rng.integers(0, d, n)
    one_hot = np.eye(d)[classes]
    labels = classes == d - 1
    test = np.r_[rng.normal(size=100), scores[:20], -9.0, 9.0]

    ivap = IVAP(backend=backend).fit(scores, labels)
    blocks = ivap.predict_blocks(test)
    sums = ivap.block_sums(test, one_hot)
    for e in (0, 1):
        lo, hi = blocks[2 * e], blocks[2 * e + 1]
        assert sums[e].shape == (len(test), d)
        expected = np.array(
            [one_hot[ivap.order_[a:b]].sum(axis=0) for a, b in zip(lo, hi, strict=True)]
        )
        np.testing.assert_array_equal(sums[e], expected)
        np.testing.assert_array_equal(sums[e].sum(axis=1), hi - lo)


def test_block_sums_rejects_wrong_shape():
    ivap = IVAP().fit([1.0, 2.0, 3.0], [0, 1, 1])
    with pytest.raises(ValueError, match="shape"):
        ivap.block_sums([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="shape"):
        ivap.block_sums([1.0], np.ones((3, 2, 2)))


def test_instance_fitted_before_blocks_existed():
    ivap = IVAP().fit([1.0, 2.0, 3.0], [0, 1, 1])
    del ivap.blocks_, ivap.order_  # as unpickled from fast-ivap 1.0.0
    ivap.predict_p0_p1([1.5])  # still works
    with pytest.raises(RuntimeError, match="fit"):
        ivap.predict_blocks([1.5])


@pytest.mark.skipif(_core is None, reason="extension not built")
def test_backends_agree_exactly():
    rng = np.random.default_rng(4)
    scores = rng.normal(size=50_000).round(2)
    labels = rng.random(50_000) < 1 / (1 + np.exp(-2 * scores))
    test = np.r_[rng.normal(size=5_000) * 1.5, scores[:1000], np.nan]
    py = IVAP(backend="python").fit(scores, labels)
    cpp = IVAP(backend="cpp").fit(scores, labels)
    np.testing.assert_array_equal(py.order_, cpp.order_)
    np.testing.assert_array_equal(py.blocks_, cpp.blocks_)
    np.testing.assert_array_equal(py.predict_blocks(test), cpp.predict_blocks(test))
