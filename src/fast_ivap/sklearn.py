"""scikit-learn compatible wrapper around :class:`fast_ivap.IVAP`.

Requires scikit-learn (``pip install fast-ivap[sklearn]``).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, column_or_1d

from ._api import IVAP, Backend


def _scores(X: ArrayLike, allow_nan: bool) -> NDArray[np.float64]:
    """Validate a 1-D array or single-column 2-D array of scores."""
    a: NDArray[np.float64] = check_array(
        X, ensure_2d=False, dtype=np.float64, ensure_all_finite="allow-nan" if allow_nan else True
    )
    if a.ndim == 2:
        if a.shape[1] != 1:
            raise ValueError(f"expected a single column of scores, got {a.shape[1]} columns")
        a = a[:, 0]
    return a


class VennAbersCalibrator(ClassifierMixin, BaseEstimator):
    """Inductive Venn-ABERS calibrator for the scores of a binary classifier.

    ``X`` holds one score per example, as a 1-D array or a single-column 2-D
    array (e.g. ``clf.decision_function(X_cal)`` or
    ``clf.predict_proba(X_cal)[:, 1]``). Higher scores should indicate the
    positive class, which is ``classes_[1]`` (the larger label).

    :meth:`predict_proba` returns the log-minimax probabilities
    ``[1 - p, p]`` with ``p = p1 / (1 - p0 + p1)``; :meth:`predict_p0_p1`
    returns the underlying Venn-ABERS pair.

    Parameters
    ----------
    backend : {"auto", "cpp", "python"}
        Implementation to use (see :class:`fast_ivap.IVAP`).

    Attributes
    ----------
    classes_ : ndarray of shape (2,)
        The two class labels seen in ``fit``.
    ivap_ : IVAP
        The fitted predictor.
    """

    def __init__(self, backend: Backend = "auto"):
        self.backend = backend

    def fit(self, X: ArrayLike, y: ArrayLike) -> VennAbersCalibrator:
        """Fit on calibration scores ``X`` and binary labels ``y``."""
        scores = _scores(X, allow_nan=False)
        labels = column_or_1d(y, warn=True)
        if len(labels) != len(scores):
            raise ValueError(f"got {len(scores)} scores but {len(labels)} labels")
        self.classes_, y_encoded = np.unique(labels, return_inverse=True)
        if len(self.classes_) != 2:
            raise ValueError(f"expected exactly two classes, got {self.classes_}")
        self.ivap_ = IVAP(backend=self.backend).fit(scores, y_encoded)
        return self

    def predict_p0_p1(self, X: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the Venn-ABERS pair ``(p0, p1)`` for the positive class."""
        check_is_fitted(self)
        return self.ivap_.predict_p0_p1(_scores(X, allow_nan=True))

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return class probabilities, shape ``(n_samples, 2)``."""
        check_is_fitted(self)
        p = self.ivap_.predict_proba(_scores(X, allow_nan=True))
        return np.column_stack([1.0 - p, p])

    def predict(self, X: ArrayLike) -> NDArray:
        """Predict the class whose probability is at least 0.5."""
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]
