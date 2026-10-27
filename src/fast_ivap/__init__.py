"""Fast inductive Venn-ABERS predictors (IVAP).

Distribution-free calibration of binary classifier scores into probabilities,
using the O(k log k) algorithm of Vovk, Petej and Fedorova (NeurIPS 2015).

>>> from fast_ivap import IVAP
>>> ivap = IVAP().fit(calibration_scores, calibration_labels)  # doctest: +SKIP
>>> p0, p1 = ivap.predict_p0_p1(test_scores)                   # doctest: +SKIP

A scikit-learn compatible estimator is available as
:class:`fast_ivap.sklearn.VennAbersCalibrator` (requires scikit-learn).
"""

from importlib.metadata import PackageNotFoundError, version

from ._api import IVAP, default_backend, merge_brier, merge_log_minimax
from ._legacy import ScoresToMultiProbs

try:
    __version__ = version("fast-ivap")
except PackageNotFoundError:  # running from a source tree
    __version__ = "0+unknown"

__all__ = [
    "IVAP",
    "ScoresToMultiProbs",
    "default_backend",
    "merge_brier",
    "merge_log_minimax",
]
