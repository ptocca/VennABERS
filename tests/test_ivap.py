"""Tests of the IVAP predictor against its definition, for every backend."""

import pickle

import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression

import fast_ivap
from fast_ivap import IVAP, merge_brier, merge_log_minimax
from fast_ivap._api import _core

BACKENDS = [
    "python",
    pytest.param("cpp", marks=pytest.mark.skipif(_core is None, reason="extension not built")),
]


def ivap_by_definition(scores, labels, test_scores):
    """IVAP straight from the definition: two isotonic fits per test score."""
    p = np.empty((2, len(test_scores)))
    for n, s in enumerate(test_scores):
        for label in (0, 1):
            iso = IsotonicRegression().fit(np.r_[scores, s], np.r_[labels, label])
            p[label, n] = iso.predict([s])[0]
    return p[0], p[1]


def assert_matches_definition(backend, scores, labels, test_scores):
    p0, p1 = IVAP(backend=backend).fit(scores, labels).predict_p0_p1(test_scores)
    d0, d1 = ivap_by_definition(scores, labels, test_scores)
    np.testing.assert_allclose(p0, d0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(p1, d1, rtol=0, atol=1e-12)


@pytest.mark.parametrize("backend", BACKENDS)
class TestAgainstDefinition:
    def test_small_example(self, backend):
        scores = [2.0, 3, 4, 5, 7, 8]
        labels = [0, 1, 0, 0, 1, 0]
        test = [1.0, 2, 2.5, 3, 4.5, 5, 6, 7, 8, 9]
        assert_matches_definition(backend, scores, labels, test)

    @pytest.mark.parametrize("seed", range(40))
    def test_random_with_ties(self, backend, seed):
        # Few distinct scores, so test scores often coincide with calibration
        # scores and calibration scores often repeat.
        rng = np.random.default_rng(seed)
        n = rng.integers(1, 40)
        levels = rng.integers(1, 12)
        scores = rng.integers(0, levels, n).astype(float)
        labels = rng.integers(0, 2, n)
        test = np.arange(-1, levels + 1, 0.5)
        assert_matches_definition(backend, scores, labels, test)

    def test_continuous_scores(self, backend):
        rng = np.random.default_rng(0)
        scores = rng.uniform(-10, 10, 400)
        labels = rng.random(400) < 1 / (1 + np.exp(-scores / 2))
        test = np.r_[np.linspace(-11, 11, 200), scores[:20]]
        assert_matches_definition(backend, scores, labels, test)

    def test_single_example(self, backend):
        assert_matches_definition(backend, [0.5], [1], [0.0, 0.5, 1.0])
        assert_matches_definition(backend, [0.5], [0], [0.0, 0.5, 1.0])

    def test_one_distinct_score(self, backend):
        assert_matches_definition(backend, [1.0, 1.0, 1.0], [0, 1, 1], [0.0, 1.0, 2.0])

    @pytest.mark.parametrize("label", [0, 1])
    def test_constant_labels(self, backend, label):
        assert_matches_definition(backend, [1.0, 2, 3, 4], [label] * 4, [0.0, 2.0, 2.5, 5.0])


@pytest.mark.parametrize("backend", BACKENDS)
def test_extremes(backend):
    ivap = IVAP(backend=backend).fit([1.0, 2.0, 3.0], [1, 0, 1])
    p0, p1 = ivap.predict_p0_p1([-np.inf, np.inf])
    assert p0[0] == 0.0  # label-0 example below everything gets probability 0
    assert p1[1] == 1.0  # label-1 example above everything gets probability 1


@pytest.mark.parametrize("backend", BACKENDS)
def test_nan_test_score_gives_nan(backend):
    ivap = IVAP(backend=backend).fit([1.0, 2.0, 3.0], [0, 1, 1])
    p0, p1 = ivap.predict_p0_p1([np.nan, 2.0])
    assert np.isnan(p0[0]) and np.isnan(p1[0])
    assert not np.isnan(p0[1]) and not np.isnan(p1[1])


@pytest.mark.parametrize("backend", BACKENDS)
def test_p0_never_exceeds_p1(backend):
    rng = np.random.default_rng(1)
    scores = rng.normal(size=2000).round(1)
    labels = rng.random(2000) < 0.3
    p0, p1 = IVAP(backend=backend).fit(scores, labels).predict_p0_p1(rng.normal(size=500))
    assert np.all(p0 <= p1)
    assert np.all((p0 >= 0) & (p1 <= 1))


@pytest.mark.skipif(_core is None, reason="extension not built")
def test_backends_agree_on_large_input():
    rng = np.random.default_rng(2)
    scores = rng.normal(size=50_000).round(3)
    labels = rng.random(50_000) < 1 / (1 + np.exp(-2 * scores))
    test = np.r_[rng.normal(size=5_000) * 1.5, scores[:1000]]
    py = IVAP(backend="python").fit(scores, labels)
    cpp = IVAP(backend="cpp").fit(scores, labels)
    np.testing.assert_array_equal(py.scores_, cpp.scores_)
    np.testing.assert_allclose(py.f0_, cpp.f0_, rtol=0, atol=1e-14)
    np.testing.assert_allclose(py.f1_, cpp.f1_, rtol=0, atol=1e-14)
    np.testing.assert_allclose(py.predict_p0_p1(test), cpp.predict_p0_p1(test), atol=1e-14)


@pytest.mark.parametrize("backend", BACKENDS)
def test_input_forms(backend):
    reference = IVAP(backend=backend).fit([1.0, 2.0, 3.0], [0, 1, 1]).predict_p0_p1([1.5])
    as_column = IVAP(backend=backend).fit([[1.0], [2.0], [3.0]], [False, True, True])
    np.testing.assert_array_equal(as_column.predict_p0_p1([[1.5]]), reference)
    as_ints = IVAP(backend=backend).fit(np.array([1, 2, 3]), np.array([0.0, 1.0, 1.0]))
    np.testing.assert_array_equal(as_ints.predict_p0_p1(np.array([1.5])), reference)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    ("scores", "labels", "message"),
    [
        ([], [], "at least one"),
        ([1.0, 2.0], [0], "labels"),
        ([1.0, 2.0], [0, 2], "0 or 1"),
        ([1.0, np.nan], [0, 1], "finite"),
        ([1.0, np.inf], [0, 1], "finite"),
        ([[1.0, 2.0]], [0], "1-D"),
    ],
)
def test_invalid_input(backend, scores, labels, message):
    with pytest.raises(ValueError, match=message):
        IVAP(backend=backend).fit(scores, labels)


def test_unfitted_and_unknown_backend():
    with pytest.raises(RuntimeError, match="not fitted"):
        IVAP().predict_p0_p1([1.0])
    with pytest.raises(ValueError, match="unknown backend"):
        IVAP(backend="fortran").fit([1.0], [1])


def test_backend_env_override(monkeypatch):
    monkeypatch.setenv("FAST_IVAP_BACKEND", "python")
    assert fast_ivap.default_backend() == "python"


def test_pickle_round_trip():
    ivap = IVAP().fit([1.0, 2.0, 3.0], [0, 1, 1])
    clone = pickle.loads(pickle.dumps(ivap))
    np.testing.assert_array_equal(clone.predict_p0_p1([2.5]), ivap.predict_p0_p1([2.5]))


class TestMerging:
    def test_log_minimax(self):
        p0, p1 = np.array([0.2, 0.0]), np.array([0.4, 1.0])
        np.testing.assert_allclose(merge_log_minimax(p0, p1), p1 / (1 - p0 + p1))

    def test_log_minimax_single_fold_matches_1d(self):
        p0, p1 = np.array([[0.2, 0.5]]), np.array([[0.4, 0.7]])
        np.testing.assert_allclose(
            merge_log_minimax(p0, p1, axis=0), merge_log_minimax(p0[0], p1[0])
        )

    def test_log_minimax_cvap(self):
        p0 = np.array([[0.1], [0.3]])
        p1 = np.array([[0.4], [0.6]])
        gm1 = np.sqrt(0.4 * 0.6)
        gm0 = np.sqrt(0.9 * 0.7)
        np.testing.assert_allclose(merge_log_minimax(p0, p1, axis=0), [gm1 / (gm0 + gm1)])

    def test_brier(self):
        np.testing.assert_allclose(merge_brier(0.2, 0.4), 0.4 + (0.04 - 0.16) / 2)
        np.testing.assert_allclose(merge_brier(0.3, 0.3), 0.3)  # p0 == p1 gives p0

    def test_predict_proba_uses_log_minimax(self):
        ivap = IVAP().fit([1.0, 2.0, 3.0, 4.0], [0, 1, 0, 1])
        test = [0.5, 2.5, 5.0]
        np.testing.assert_allclose(
            ivap.predict_proba(test), merge_log_minimax(*ivap.predict_p0_p1(test))
        )


def test_legacy_function():
    points = [(1.0, 0), (2.0, 1), (3.0, 1)]
    with pytest.warns(DeprecationWarning):
        p0, p1 = fast_ivap.ScoresToMultiProbs(points, [1.5, 3.0])
    np.testing.assert_array_equal(
        (p0, p1), IVAP().fit([1, 2, 3], [0, 1, 1]).predict_p0_p1([1.5, 3])
    )
