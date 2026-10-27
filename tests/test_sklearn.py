"""Tests of the scikit-learn wrapper."""

import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from fast_ivap import IVAP
from fast_ivap.sklearn import VennAbersCalibrator


@pytest.fixture
def calibration_data():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=300)
    labels = rng.random(300) < 1 / (1 + np.exp(-2 * scores))
    return scores, labels


def test_matches_ivap(calibration_data):
    scores, labels = calibration_data
    test = np.linspace(-3, 3, 50)
    cal = VennAbersCalibrator().fit(scores.reshape(-1, 1), labels)
    ivap = IVAP().fit(scores, labels)
    np.testing.assert_array_equal(cal.predict_p0_p1(test.reshape(-1, 1)), ivap.predict_p0_p1(test))
    proba = cal.predict_proba(test)
    assert proba.shape == (50, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    np.testing.assert_allclose(proba[:, 1], ivap.predict_proba(test))


def test_string_labels(calibration_data):
    scores, labels = calibration_data
    y = np.where(labels, "spam", "ham")  # "spam" > "ham", so it is the positive class
    cal = VennAbersCalibrator().fit(scores, y)
    assert list(cal.classes_) == ["ham", "spam"]
    assert set(cal.predict(scores)) <= {"ham", "spam"}
    np.testing.assert_array_equal(
        cal.predict_proba(scores), VennAbersCalibrator().fit(scores, labels).predict_proba(scores)
    )


def test_clone_and_pickle(calibration_data):
    scores, labels = calibration_data
    cal = clone(VennAbersCalibrator(backend="python")).fit(scores, labels)
    assert cal.get_params() == {"backend": "python"}
    restored = pickle.loads(pickle.dumps(cal))
    np.testing.assert_array_equal(restored.predict_proba(scores), cal.predict_proba(scores))


def test_rejects_bad_input(calibration_data):
    scores, labels = calibration_data
    with pytest.raises(ValueError, match="two classes"):
        VennAbersCalibrator().fit(scores, np.arange(len(scores)) % 3)
    with pytest.raises(ValueError, match="single column"):
        VennAbersCalibrator().fit(np.c_[scores, scores], labels)


def test_end_to_end_with_classifier():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 5))
    y = (X @ [1.0, -2.0, 0.5, 0.0, 1.0] + rng.logistic(size=2000)) > 0
    X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.5, random_state=0)
    X_cal, X_test, y_cal, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=0)

    clf = LogisticRegression().fit(X_train, y_train)
    cal = VennAbersCalibrator().fit(clf.decision_function(X_cal), y_cal)
    p = cal.predict_proba(clf.decision_function(X_test))[:, 1]

    # Calibrated probabilities should be sensible: low Brier score, and the
    # average predicted probability close to the observed frequency.
    assert np.mean((p - y_test) ** 2) < 0.2
    assert abs(p.mean() - y_test.mean()) < 0.05
