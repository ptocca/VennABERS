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

// A fitted predictor that owns its lookup tables (see CalibratorView).
struct Calibrator {
    std::vector<double> scores;
    std::vector<double> f0;
    std::vector<double> f1;

    [[nodiscard]] CalibratorView view() const noexcept { return {scores, f0, f1}; }
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

// Algorithms 1 and 2 of the paper: returns F^1_1, ..., F^1_{k+1}.
//
// weights[i] is the number of calibration examples sharing the i-th smallest
// distinct score; label_sums[i] is the sum of their labels.
inline std::vector<double> upper_probabilities(std::span<const double> weights,
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
    std::vector<Point> stack(hull.rbegin(), hull.rend());
    std::vector<double> f1(k + 1);
    f1[k] = 1.0;  // F^1_{k+1}: a label-1 test example right of all scores
    for (std::size_t i = 1; i <= k; ++i) {
        const Point left = stack.back();
        const Point right = stack[stack.size() - 2];
        f1[i - 1] = (right.y - left.y) / (right.x - left.x);

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
    return f1;
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

    std::vector<std::pair<double, double>> examples(scores.size());
    for (std::size_t i = 0; i < scores.size(); ++i) {
        if (!std::isfinite(scores[i])) {
            throw std::invalid_argument("calibration scores must be finite");
        }
        if (labels[i] != 0.0 && labels[i] != 1.0) {
            throw std::invalid_argument("labels must be 0 or 1");
        }
        examples[i] = {scores[i], labels[i]};
    }
    std::sort(examples.begin(), examples.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // Group equal scores: distinct score, multiplicity and label sum.
    Calibrator cal;
    std::vector<double> weights;
    std::vector<double> label_sums;
    for (const auto& [score, label] : examples) {
        if (cal.scores.empty() || score != cal.scores.back()) {
            cal.scores.push_back(score);
            weights.push_back(0.0);
            label_sums.push_back(0.0);
        }
        weights.back() += 1.0;
        label_sums.back() += label;
    }
    const std::size_t k = cal.scores.size();

    cal.f1 = detail::upper_probabilities(weights, label_sums);

    // F^0 by symmetry (paper, after Algorithm 4): mirror the scores and flip
    // the labels, then F^0_j = 1 - F^1_{k+1-j} of the mirrored problem.
    std::vector<double> mirrored_weights(weights.rbegin(), weights.rend());
    std::vector<double> mirrored_sums(k);
    for (std::size_t i = 0; i < k; ++i) {
        mirrored_sums[i] = weights[k - 1 - i] - label_sums[k - 1 - i];
    }
    const std::vector<double> f1_mirrored =
        detail::upper_probabilities(mirrored_weights, mirrored_sums);
    cal.f0.resize(k + 1);
    for (std::size_t j = 0; j <= k; ++j) {
        cal.f0[j] = 1.0 - f1_mirrored[k - j];
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

}  // namespace fast_ivap
