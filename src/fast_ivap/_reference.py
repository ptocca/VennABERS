"""Pure-Python reference implementation of the fast IVAP algorithm.

This module follows Section 2 ("Computational details of IVAPs") of

    V. Vovk, I. Petej, V. Fedorova. *Large-scale probabilistic predictors with
    and without guarantees of validity*. NeurIPS 2015. arXiv:1511.00213.

It is deliberately written for readability rather than speed, and serves as

* the fallback backend when the compiled extension is not available, and
* an independent implementation the C++ backend is tested against.

Both backends share the same contract::

    fit(scores, labels) -> (unique_scores, f0, f1, order, blocks)
    predict(unique_scores, f0, f1, test_scores) -> (p0, p1)
    predict_blocks(unique_scores, blocks, test_scores) -> ranges

where, with k distinct calibration scores s'_1 < ... < s'_k,

* ``unique_scores[j] = s'_{j+1}``                 for j = 0..k-1
* ``f0[j] = F^0_j``   (f0 at s'_j, with F^0_0 = 0)    for j = 0..k
* ``f1[j] = F^1_{j+1}`` (f1 at s'_{j+1}, with F^1_{k+1} = 1) for j = 0..k
* ``order`` is the stable argsort of the n calibration scores (int64)
* ``blocks`` is an int64 array of shape (4, k + 1) with rows
  ``lo0, hi0, lo1, hi1``: column j gives, for a test score at the same
  position as ``f0[j]`` (rows 0-1) or ``f1[j]`` (rows 2-3), the isotonic block
  ("Venn category") of the test example as the half-open range ``[lo, hi)``
  of calibration items in sorted order, i.e. ``order[lo:hi]``. The test
  example itself is not counted; the range may be empty.
* ``predict_blocks`` returns an int64 array of shape (4, m) with rows
  ``lo0, hi0, lo1, hi1`` per test score, and -1 everywhere for NaN scores.

Inputs are assumed to be validated already (see :mod:`fast_ivap._api`).
"""

from __future__ import annotations

import numpy as np

Point = tuple[float, float]


def _cross(o: Point, a: Point, b: Point) -> float:
    """Return the z-component of (a - o) x (b - o).

    Positive when o -> a -> b turns left (counter-clockwise), zero when the
    three points are collinear, negative when it turns right.
    """
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _upper_probabilities(
    weights: np.ndarray, label_sums: np.ndarray
) -> tuple[list[float], list[int], list[int]]:
    """Compute F^1_1, ..., F^1_{k+1} (Algorithms 1 and 2 of the paper).

    ``weights[i]`` is the number of calibration examples with the i-th smallest
    distinct score and ``label_sums[i]`` the sum of their labels.

    F^1_i is the isotonic-regression value at a test score s placed just left
    of s'_i (or equal to it) when the test example is given label 1.

    Also returns, for the same positions, the isotonic block containing the
    test example as a half-open range ``[lo, hi)`` of sorted calibration items
    (the test example excluded).
    """
    k = len(weights)

    # Cumulative sum diagram (CSD), extended with P_{-1} = (-1, -1).
    # ``csd[j]`` holds P_{j-1}, so the paper's P_i lives at index i + 1.
    # P_{-1} -> P_0 is the segment of the test example (weight 1, label 1).
    xs = np.concatenate(([-1.0, 0.0], np.cumsum(weights, dtype=float)))
    ys = np.concatenate(([-1.0, 0.0], np.cumsum(label_sums, dtype=float)))
    csd: list[Point] = list(zip(xs.tolist(), ys.tolist(), strict=True))

    # Algorithm 1: Graham scan for the corners of the greatest convex minorant
    # (GCM), i.e. the lower convex hull. Leftmost corner ends up at the bottom.
    hull: list[Point] = [csd[0], csd[1]]
    for p in csd[2:]:
        while len(hull) > 1 and _cross(hull[-2], hull[-1], p) <= 0:  # non-left turn
            hull.pop()
        hull.append(p)

    # Algorithm 2: slide the test example's unit segment ("active interval")
    # rightwards one calibration score at a time, keeping the GCM up to date.
    # ``stack`` holds the GCM corners right of the active interval, plus the
    # active corner on top (so the leftmost corner is on top).
    #
    # With P_{-1} at x = -1, a point left of the active interval sits at
    # x = c - 1 and a point right of it at x = c, where c is the number of
    # calibration items before it. So the GCM piece over the active interval
    # spans the sorted calibration items [left.x + 1, right.x). Collinear
    # corners are never kept, so this piece is the whole level set of the
    # isotonic fit containing the test example.
    stack = hull[::-1]
    f1 = [0.0] * (k + 1)
    f1[k] = 1.0  # F^1_{k+1}: a label-1 test example right of all scores gets 1
    lo = [0] * (k + 1)
    hi = [0] * (k + 1)
    for i in range(1, k + 2):
        left, right = stack[-1], stack[-2]
        lo[i - 1] = int(left[0]) + 1
        hi[i - 1] = int(right[0])
        if i == k + 1:
            # Right of all scores: only the block is new, the slope is 1.
            break
        # F^1_i is the slope of the GCM over the active interval.
        f1[i - 1] = (right[1] - left[1]) / (right[0] - left[0])

        # Swap the active segment with the segment of s'_i. In the paper:
        # P_{i-1} := P_{i-2} + P_i - P_{i-1}, a reflection through the midpoint.
        (ax, ay), (bx, by), (cx, cy) = csd[i - 1], csd[i], csd[i + 1]
        moved = (ax + cx - bx, ay + cy - by)
        csd[i] = moved

        # If the moved point is on or above the current GCM, the GCM is unchanged.
        if _cross(left, right, moved) >= 0:
            continue

        # Otherwise it becomes the new active corner; drop corners that are no
        # longer on the lower hull as seen from it.
        stack.pop()
        while len(stack) > 1 and _cross(moved, stack[-1], stack[-2]) <= 0:
            stack.pop()
        stack.append(moved)

    return f1, lo, hi


def fit(
    scores: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pre-compute the IVAP lookup tables from calibration scores and labels."""
    n = len(scores)
    order = np.argsort(scores, kind="stable").astype(np.int64)
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    unique_scores, first_index, weights = np.unique(
        sorted_scores, return_index=True, return_counts=True
    )
    label_sums = np.add.reduceat(sorted_labels, first_index)

    f1, lo1, hi1 = map(np.array, _upper_probabilities(weights, label_sums))

    # F^0 follows from F^1 by symmetry (paper, after Algorithm 4): negating the
    # scores and flipping the labels turns "label 0" into "label 1", so
    #   F^0_j = 1 - F^1_{k+1-j}  computed on the mirrored calibration set.
    # Item i of the mirrored sorted order is item n-1-i of ours, so the
    # mirrored block [lo, hi) is our block [n - hi, n - lo).
    f1_m, lo_m, hi_m = map(
        np.array, _upper_probabilities(weights[::-1], (weights - label_sums)[::-1])
    )
    f0 = 1.0 - f1_m[::-1]
    blocks = np.array([n - hi_m[::-1], n - lo_m[::-1], lo1, hi1], dtype=np.int64)

    return unique_scores, f0, f1, order, blocks


def predict(
    unique_scores: np.ndarray, f0: np.ndarray, f1: np.ndarray, test_scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Look up (p0, p1) for each test score in O(log k)."""
    k = len(unique_scores)
    # j = number of distinct calibration scores strictly below s.
    j = np.searchsorted(unique_scores, test_scores, side="left")
    exact = (j < k) & (unique_scores[np.minimum(j, k - 1)] == test_scores)

    # s strictly between s'_j and s'_{j+1}: (F^0_j, F^1_{j+1})
    # s equal to s'_{j+1}:                  (F^0_{j+1}, F^1_{j+1})
    p0 = f0[j + exact]
    p1 = f1[j]

    nan = np.isnan(test_scores)
    if nan.any():
        p0 = np.where(nan, np.nan, p0)
        p1 = np.where(nan, np.nan, p1)
    return p0, p1


def predict_blocks(
    unique_scores: np.ndarray, blocks: np.ndarray, test_scores: np.ndarray
) -> np.ndarray:
    """Look up the isotonic blocks of each test score, with the ties of :func:`predict`."""
    k = len(unique_scores)
    j = np.searchsorted(unique_scores, test_scores, side="left")
    exact = (j < k) & (unique_scores[np.minimum(j, k - 1)] == test_scores)
    out = np.stack([blocks[0, j + exact], blocks[1, j + exact], blocks[2, j], blocks[3, j]])
    out[:, np.isnan(test_scores)] = -1
    return out
