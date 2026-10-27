"""Compatibility shim for code written against the original VennABERS.py."""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

from ._api import IVAP


def ScoresToMultiProbs(  # noqa: N802 - historical name
    calibrPoints, testObjects
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Deprecated: use ``IVAP().fit(scores, labels).predict_p0_p1(test)``.

    ``calibrPoints`` is a sequence of ``(score, label)`` pairs and
    ``testObjects`` a sequence of test scores. Returns ``(p0, p1)``.

    Note that this function fixes two bugs of the original implementation
    (test scores equal to a calibration score, and test scores above all
    calibration scores), so results can differ from VennABERS 0.2 in those
    cases.
    """
    warnings.warn(
        "ScoresToMultiProbs is deprecated; use IVAP().fit(scores, labels).predict_p0_p1(test)",
        DeprecationWarning,
        stacklevel=2,
    )
    points = list(calibrPoints)
    scores = [p[0] for p in points]
    labels = [p[1] for p in points]
    return IVAP().fit(scores, labels).predict_p0_p1(testObjects)
