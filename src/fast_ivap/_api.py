"""Public NumPy API: the :class:`IVAP` predictor and probability merging."""

from __future__ import annotations

import os
from types import ModuleType
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _reference

try:
    from . import _core
except ImportError:  # extension not built (e.g. pure-Python install)
    _core = None

Backend = Literal["auto", "cpp", "python"]

_BACKENDS: dict[str, ModuleType | None] = {"cpp": _core, "python": _reference}


def default_backend() -> str:
    """Name of the backend used when ``backend="auto"``.

    This is ``"cpp"`` when the compiled extension is available, ``"python"``
    otherwise. Set the environment variable ``FAST_IVAP_BACKEND`` to
    ``"python"`` or ``"cpp"`` to override it.
    """
    forced = os.environ.get("FAST_IVAP_BACKEND", "").strip().lower()
    if forced:
        return forced
    return "cpp" if _core is not None else "python"


def _get_backend(name: str) -> ModuleType:
    if name == "auto":
        name = default_backend()
    if name not in _BACKENDS:
        raise ValueError(f"unknown backend {name!r}; expected 'auto', 'cpp' or 'python'")
    module = _BACKENDS[name]
    if module is None:
        raise ImportError(
            "the compiled fast_ivap._core extension is not available; "
            "reinstall fast-ivap from a wheel or use backend='python'"
        )
    return module


def _as_scores(x: ArrayLike, name: str) -> NDArray[np.float64]:
    a = np.ascontiguousarray(x, dtype=np.float64)
    if a.ndim == 2 and a.shape[1] == 1:
        a = a.ravel()
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D (or a single column), got shape {np.shape(x)}")
    return a


def _as_labels(y: ArrayLike, n: int) -> NDArray[np.float64]:
    a = np.ascontiguousarray(y, dtype=np.float64).ravel()
    if a.shape[0] != n:
        raise ValueError(f"got {n} scores but {a.shape[0]} labels")
    if not np.all((a == 0) | (a == 1)):
        raise ValueError("labels must be 0 or 1 (or booleans)")
    return a


class IVAP:
    """Inductive Venn-ABERS predictor.

    Calibrates the scores of a binary classifier into probabilities that are
    valid (well-calibrated) under the sole assumption that calibration and
    test examples are i.i.d. (exchangeable).

    For each test score the predictor outputs two probabilities that the
    label is 1, ``p0 <= p1``. They are obtained by isotonic regression on the
    calibration set augmented with the test example labelled 0 and 1
    respectively. Use :meth:`predict_proba` for a single probability.

    Parameters
    ----------
    backend : {"auto", "cpp", "python"}
        Implementation to use. ``"auto"`` picks the compiled extension if
        available (see :func:`default_backend`).

    Attributes
    ----------
    scores_ : ndarray of shape (k,)
        Distinct calibration scores, sorted.
    f0_, f1_ : ndarray of shape (k + 1,)
        Lookup tables of p0 and p1 (see :mod:`fast_ivap._reference`).

    Examples
    --------
    >>> ivap = IVAP().fit([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1])
    >>> p0, p1 = ivap.predict_p0_p1([0.2, 0.6])
    """

    def __init__(self, backend: Backend = "auto"):
        self.backend = backend

    def fit(self, scores: ArrayLike, labels: ArrayLike) -> IVAP:
        """Fit on calibration scores and 0/1 labels. Runs in O(k log k)."""
        s = _as_scores(scores, "scores")
        y = _as_labels(labels, s.shape[0])
        if s.shape[0] == 0:
            raise ValueError("at least one calibration example is required")
        if not np.all(np.isfinite(s)):
            raise ValueError("calibration scores must be finite")
        self.scores_, self.f0_, self.f1_ = _get_backend(self.backend).fit(s, y)
        return self

    def predict_p0_p1(self, scores: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the pair ``(p0, p1)`` for each test score.

        Both are probabilities that the label is 1; ``p0 <= p1`` always holds.
        NaN scores give NaN probabilities.
        """
        if not hasattr(self, "scores_"):
            raise RuntimeError("this IVAP instance is not fitted yet; call fit() first")
        s = _as_scores(scores, "scores")
        return _get_backend(self.backend).predict(self.scores_, self.f0_, self.f1_, s)

    def predict_proba(self, scores: ArrayLike) -> NDArray[np.float64]:
        """Return a single probability of label 1 per test score.

        This is the log-minimax merge ``p1 / (1 - p0 + p1)`` of the pair
        output by :meth:`predict_p0_p1` (Section 4 of the paper).
        """
        return merge_log_minimax(*self.predict_p0_p1(scores))


def merge_log_minimax(p0: ArrayLike, p1: ArrayLike, axis: int | None = None) -> NDArray[np.float64]:
    """Merge Venn-ABERS outputs into one probability, minimax under log loss.

    With 1-D inputs this is ``p1 / (1 - p0 + p1)`` (log-minimax IVAP). When
    the inputs stack the outputs of K predictors along ``axis`` (as in a cross
    Venn-ABERS predictor, CVAP), the result is
    ``GM(p1) / (GM(1 - p0) + GM(p1))`` where GM is the geometric mean along
    ``axis``.
    """
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    if axis is not None:
        with np.errstate(divide="ignore"):
            gm1 = np.exp(np.mean(np.log(p1), axis=axis))
            gm0 = np.exp(np.mean(np.log1p(-p0), axis=axis))
        return gm1 / (gm0 + gm1)
    return p1 / (1.0 - p0 + p1)


def merge_brier(p0: ArrayLike, p1: ArrayLike, axis: int | None = None) -> NDArray[np.float64]:
    """Merge Venn-ABERS outputs into one probability, minimax under Brier loss.

    Computes ``p1 + (p0**2 - p1**2) / 2``, averaged along ``axis`` if given
    (Section 4 of the paper).
    """
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    p = p1 + 0.5 * (p0**2 - p1**2)
    return p if axis is None else np.mean(p, axis=axis)
