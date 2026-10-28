// fast_ivap/ivap.hpp - header-only C++20 implementation of the fast
// Inductive Venn-ABERS Predictor (IVAP).
//
// Reference:
//   V. Vovk, I. Petej, V. Fedorova. "Large-scale probabilistic predictors with
//   and without guarantees of validity". NeurIPS 2015. arXiv:1511.00213.
//
// Usage:
//   auto cal = fast_ivap::fit(calibration_scores, calibration_labels);
//   auto [p0, p1] = fast_ivap::predict(cal, test_scores);
//   auto blocks = fast_ivap::predict_blocks(cal, test_scores);  // Venn categories
//
// Fitting costs O(k log k) for k calibration examples (dominated by sorting);
// each prediction costs O(log k). The library has no dependencies beyond the
// C++20 standard library.
//
// SPDX-License-Identifier: MIT
// Copyright (c) Paolo Toccaceli

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fast_ivap {

// Read-only view of a fitted predictor. With k distinct calibration scores
// s'_1 < ... < s'_k (all three spans as in the paper's notation):
//   scores[j] = s'_{j+1}                                       j = 0..k-1
//   f0[j]     = F^0_j,     the p0 at s'_j        (F^0_0 = 0)    j = 0..k
//   f1[j]     = F^1_{j+1}, the p1 at s'_{j+1}    (F^1_{k+1} = 1) j = 0..k
struct CalibratorView {
    std::span<const double> scores;
    std::span<const double> f0;
    std::span<const double> f1;
};

// Read-only view of the isotonic-block tables of a fitted predictor.
//
// For label e, the "block" of a test example is the level set of the isotonic
// fit on calibration data plus the test example that contains the test
// example. It is the Venn-ABERS category of the test example, and p_e is
// (sum of labels in the block + e) / (size of the block + 1). Blocks are
// half-open ranges [lo, hi) of calibration items sorted by score (stably, see
// Calibrator::order), and exclude the test example itself.
//   lo0[j], hi0[j]: block for label 0 at the position of f0[j]   j = 0..k
//   lo1[j], hi1[j]: block for label 1 at the position of f1[j]   j = 0..k
struct BlocksView {
    std::span<const double> scores;
    std::span<const std::int64_t> lo0;
    std::span<const std::int64_t> hi0;
    std::span<const std::int64_t> lo1;
    std::span<const std::int64_t> hi1;
};

// Isotonic blocks of a batch of test scores (see BlocksView).
struct Blocks {
    std::vector<std::int64_t> lo0;
    std::vector<std::int64_t> hi0;
    std::vector<std::int64_t> lo1;
    std::vector<std::int64_t> hi1;
};

// A fitted predictor that owns its lookup tables (see CalibratorView and
// BlocksView).
struct Calibrator {
    std::vector<double> scores;
    std::vector<double> f0;
    std::vector<double> f1;
    // Stable argsort of the calibration scores: order[r] is the input index
    // of the calibration item with rank r.
    std::vector<std::int64_t> order;
    Blocks blocks;  // per-position block tables, k + 1 entries each

    [[nodiscard]] CalibratorView view() const noexcept { return {scores, f0, f1}; }
    [[nodiscard]] BlocksView blocks_view() const noexcept {
        return {scores, blocks.lo0, blocks.hi0, blocks.lo1, blocks.hi1};
    }
};

namespace detail {

struct Point {
    double x;
    double y;
};

// z-component of (a - o) x (b - o): > 0 for a left turn o -> a -> b,
// 0 if collinear, < 0 for a right turn.
inline double cross(const Point& o, const Point& a, const Point& b) noexcept {
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

struct UpperResult {
    std::vector<double> f1;        // F^1_1, ..., F^1_{k+1}
    std::vector<std::int64_t> lo;  // block of the test example at the same
    std::vector<std::int64_t> hi;  // positions, as sorted items [lo, hi)
};

// Algorithms 1 and 2 of the paper: computes F^1_1, ..., F^1_{k+1} and the
// isotonic block containing the test example at each position.
//
// weights[i] is the number of calibration examples sharing the i-th smallest
// distinct score; label_sums[i] is the sum of their labels.
inline UpperResult upper_probabilities(std::span<const double> weights,
                                       std::span<const double> label_sums) {
    const std::size_t k = weights.size();

    // Cumulative sum diagram, extended with P_{-1} = (-1, -1): csd[j] = P_{j-1}.
    // The first segment P_{-1} -> P_0 belongs to the test example (label 1).
    std::vector<Point> csd(k + 2);
    csd[0] = {-1.0, -1.0};
    csd[1] = {0.0, 0.0};
    for (std::size_t i = 0; i < k; ++i) {
        csd[i + 2] = {csd[i + 1].x + weights[i], csd[i + 1].y + label_sums[i]};
    }

    // Algorithm 1: Graham scan for the corners of the greatest convex
    // minorant (lower convex hull), leftmost corner at the bottom.
    std::vector<Point> hull;
    hull.reserve(k + 2);
    for (const Point& p : csd) {
        while (hull.size() > 1 && cross(hull[hull.size() - 2], hull.back(), p) <= 0.0) {
            hull.pop_back();  // non-left turn: hull.back() is not a corner
        }
        hull.push_back(p);
    }

    // Algorithm 2: move the test example's unit segment (the "active
    // interval") rightwards past one calibration score at a time. The stack
    // holds the GCM corners to the right of it, leftmost (active) corner on top.
    //
    // A point left of the active interval sits at x = c - 1 and a point right
    // of it at x = c, where c counts the calibration items before it (x values
    // are integers, exact in double up to 2^53). So the GCM piece over the
    // active interval is the block of sorted items [left.x + 1, right.x).
    // Collinear corners are never kept, so it is a whole level set.
    std::vector<Point> stack(hull.rbegin(), hull.rend());
    UpperResult out{std::vector<double>(k + 1), std::vector<std::int64_t>(k + 1),
                    std::vector<std::int64_t>(k + 1)};
    out.f1[k] = 1.0;  // F^1_{k+1}: a label-1 test example right of all scores
    for (std::size_t i = 1; i <= k + 1; ++i) {
        const Point left = stack.back();
        const Point right = stack[stack.size() - 2];
        out.lo[i - 1] = static_cast<std::int64_t>(left.x) + 1;
        out.hi[i - 1] = static_cast<std::int64_t>(right.x);
        if (i == k + 1) {
            break;  // right of all scores: only the block is new (slope is 1)
        }
        out.f1[i - 1] = (right.y - left.y) / (right.x - left.x);

        // Swap the active segment with the segment of s'_i:
        // P_{i-1} := P_{i-2} + P_i - P_{i-1} (reflection through the midpoint).
        const Point moved{csd[i - 1].x + csd[i + 1].x - csd[i].x,
                          csd[i - 1].y + csd[i + 1].y - csd[i].y};
        csd[i] = moved;

        if (cross(left, right, moved) >= 0.0) {
            continue;  // on or above the GCM: nothing changes
        }
        // The moved point becomes the new active corner.
        stack.pop_back();
        while (stack.size() > 1 && cross(moved, stack.back(), stack[stack.size() - 2]) <= 0.0) {
            stack.pop_back();
        }
        stack.push_back(moved);
    }
    return out;
}

}  // namespace detail

// Fit an IVAP on calibration scores and binary (0/1) labels.
//
// Throws std::invalid_argument if the inputs are empty, of different lengths,
// contain non-finite scores, or labels other than 0 and 1.
inline Calibrator fit(std::span<const double> scores, std::span<const double> labels) {
    if (scores.size() != labels.size()) {
        throw std::invalid_argument("scores and labels must have the same length");
    }
    if (scores.empty()) {
        throw std::invalid_argument("at least one calibration example is required");
    }

    const std::size_t n = scores.size();
    // Sort (score, 2 * input index + label) pairs: ordering by index within
    // ties makes this a stable argsort, and carrying the label in the low bit
    // avoids a cache-unfriendly gather of the labels afterwards.
    std::vector<std::pair<double, std::int64_t>> ranked(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(scores[i])) {
            throw std::invalid_argument("calibration scores must be finite");
        }
        if (labels[i] != 0.0 && labels[i] != 1.0) {
            throw std::invalid_argument("labels must be 0 or 1");
        }
        ranked[i] = {scores[i], 2 * static_cast<std::int64_t>(i) + (labels[i] == 1.0 ? 1 : 0)};
    }
    std::sort(ranked.begin(), ranked.end());

    // Group equal scores: distinct score, multiplicity and label sum.
    Calibrator cal;
    cal.order.resize(n);
    std::vector<double> weights;
    std::vector<double> label_sums;
    for (std::size_t r = 0; r < n; ++r) {
        const auto [score, key] = ranked[r];
        const double label = static_cast<double>(key & 1);
        cal.order[r] = key >> 1;
        if (cal.scores.empty() || score != cal.scores.back()) {
            cal.scores.push_back(score);
            weights.push_back(0.0);
            label_sums.push_back(0.0);
        }
        weights.back() += 1.0;
        label_sums.back() += label;
    }
    const std::size_t k = cal.scores.size();

    detail::UpperResult upper = detail::upper_probabilities(weights, label_sums);
    cal.f1 = std::move(upper.f1);
    cal.blocks.lo1 = std::move(upper.lo);
    cal.blocks.hi1 = std::move(upper.hi);

    // F^0 by symmetry (paper, after Algorithm 4): mirror the scores and flip
    // the labels, then F^0_j = 1 - F^1_{k+1-j} of the mirrored problem.
    std::vector<double> mirrored_weights(weights.rbegin(), weights.rend());
    std::vector<double> mirrored_sums(k);
    for (std::size_t i = 0; i < k; ++i) {
        mirrored_sums[i] = weights[k - 1 - i] - label_sums[k - 1 - i];
    }
    // Sorted item r of the mirrored problem is our item n-1-r, so its block
    // [lo, hi) is our block [n - hi, n - lo).
    const detail::UpperResult mirrored =
        detail::upper_probabilities(mirrored_weights, mirrored_sums);
    const auto n_items = static_cast<std::int64_t>(n);
    cal.f0.resize(k + 1);
    cal.blocks.lo0.resize(k + 1);
    cal.blocks.hi0.resize(k + 1);
    for (std::size_t j = 0; j <= k; ++j) {
        cal.f0[j] = 1.0 - mirrored.f1[k - j];
        cal.blocks.lo0[j] = n_items - mirrored.hi[k - j];
        cal.blocks.hi0[j] = n_items - mirrored.lo[k - j];
    }
    return cal;
}

// Compute (p0, p1) for each test score, writing into p0 and p1 (which must
// have the same length as test_scores). A NaN test score yields NaN outputs.
inline void predict(const CalibratorView& cal, std::span<const double> test_scores,
                    std::span<double> p0, std::span<double> p1) {
    if (p0.size() != test_scores.size() || p1.size() != test_scores.size()) {
        throw std::invalid_argument("output spans must match the number of test scores");
    }
    const auto first = cal.scores.begin();
    const auto last = cal.scores.end();
    for (std::size_t n = 0; n < test_scores.size(); ++n) {
        const double s = test_scores[n];
        if (std::isnan(s)) {
            p0[n] = p1[n] = std::nan("");
            continue;
        }
        // j = number of distinct calibration scores strictly below s.
        const auto it = std::lower_bound(first, last, s);
        const auto j = static_cast<std::size_t>(it - first);
        const bool exact = it != last && *it == s;
        p0[n] = cal.f0[exact ? j + 1 : j];
        p1[n] = cal.f1[j];
    }
}

// Convenience overload returning freshly allocated (p0, p1) vectors.
inline std::pair<std::vector<double>, std::vector<double>> predict(
    const CalibratorView& cal, std::span<const double> test_scores) {
    std::vector<double> p0(test_scores.size());
    std::vector<double> p1(test_scores.size());
    predict(cal, test_scores, p0, p1);
    return {std::move(p0), std::move(p1)};
}

inline std::pair<std::vector<double>, std::vector<double>> predict(
    const Calibrator& cal, std::span<const double> test_scores) {
    return predict(cal.view(), test_scores);
}

// Write the isotonic blocks (Venn categories) of each test score, for labels
// 0 and 1, as ranges [lo, hi) of sorted calibration items (see BlocksView).
// Ties are treated exactly as in predict(). A NaN test score yields -1 in all
// four outputs. All output spans must have the length of test_scores.
inline void predict_blocks(const BlocksView& cal, std::span<const double> test_scores,
                           std::span<std::int64_t> lo0, std::span<std::int64_t> hi0,
                           std::span<std::int64_t> lo1, std::span<std::int64_t> hi1) {
    const std::size_t m = test_scores.size();
    if (lo0.size() != m || hi0.size() != m || lo1.size() != m || hi1.size() != m) {
        throw std::invalid_argument("output spans must match the number of test scores");
    }
    const auto first = cal.scores.begin();
    const auto last = cal.scores.end();
    for (std::size_t n = 0; n < m; ++n) {
        const double s = test_scores[n];
        if (std::isnan(s)) {
            lo0[n] = hi0[n] = lo1[n] = hi1[n] = -1;
            continue;
        }
        const auto it = std::lower_bound(first, last, s);
        const auto j = static_cast<std::size_t>(it - first);
        const std::size_t j0 = (it != last && *it == s) ? j + 1 : j;
        lo0[n] = cal.lo0[j0];
        hi0[n] = cal.hi0[j0];
        lo1[n] = cal.lo1[j];
        hi1[n] = cal.hi1[j];
    }
}

// Convenience overloads returning freshly allocated ranges.
inline Blocks predict_blocks(const BlocksView& cal, std::span<const double> test_scores) {
    const std::size_t m = test_scores.size();
    Blocks out{std::vector<std::int64_t>(m), std::vector<std::int64_t>(m),
               std::vector<std::int64_t>(m), std::vector<std::int64_t>(m)};
    predict_blocks(cal, test_scores, out.lo0, out.hi0, out.lo1, out.hi1);
    return out;
}

inline Blocks predict_blocks(const Calibrator& cal, std::span<const double> test_scores) {
    return predict_blocks(cal.blocks_view(), test_scores);
}

}  // namespace fast_ivap
