"""Type stub for the compiled extension (src/bindings/core.cpp)."""

import numpy as np
from numpy.typing import NDArray

def fit(
    scores: NDArray[np.float64], labels: NDArray[np.float64]
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.int64],
]: ...
def predict(
    unique_scores: NDArray[np.float64],
    f0: NDArray[np.float64],
    f1: NDArray[np.float64],
    test_scores: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
def predict_blocks(
    unique_scores: NDArray[np.float64],
    blocks: NDArray[np.int64],
    test_scores: NDArray[np.float64],
) -> NDArray[np.int64]: ...
