// C++ tests for fast_ivap/ivap.hpp: compare against IVAP computed from its
// definition with a straightforward pool-adjacent-violators (PAVA) isotonic
// regression. Build and run with CMake/CTest (see README).

#include <fast_ivap/ivap.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAILED: %s\n", what);
        ++failures;
    }
}

// Isotonic regression value at `target` for data (x, y), ties in x pooled.
double isotonic_at(std::vector<std::pair<double, double>> data, double target) {
    std::sort(data.begin(), data.end());
    struct Block {
        double x_last, sum, weight;
    };
    std::vector<Block> blocks;
    for (std::size_t i = 0; i < data.size();) {
        Block b{data[i].first, 0.0, 0.0};
        for (; i < data.size() && data[i].first == b.x_last; ++i) {
            b.sum += data[i].second;
            b.weight += 1.0;
        }
        blocks.push_back(b);
        while (blocks.size() > 1) {
            Block& prev = blocks[blocks.size() - 2];
            const Block& last = blocks.back();
            if (prev.sum / prev.weight <= last.sum / last.weight) break;
            prev = {last.x_last, prev.sum + last.sum, prev.weight + last.weight};
            blocks.pop_back();
        }
    }
    for (const Block& b : blocks) {
        if (target <= b.x_last) return b.sum / b.weight;
    }
    return std::nan("");
}

std::pair<double, double> ivap_by_definition(const std::vector<double>& scores,
                                             const std::vector<double>& labels, double s) {
    std::vector<std::pair<double, double>> data;
    for (std::size_t i = 0; i < scores.size(); ++i) data.emplace_back(scores[i], labels[i]);
    data.emplace_back(s, 0.0);
    const double p0 = isotonic_at(data, s);
    data.back().second = 1.0;
    const double p1 = isotonic_at(data, s);
    return {p0, p1};
}

void test_against_definition() {
    std::mt19937 rng(12345);
    for (int trial = 0; trial < 500; ++trial) {
        const int n = 1 + static_cast<int>(rng() % 40);
        const int levels = 1 + static_cast<int>(rng() % 12);
        std::vector<double> scores(n), labels(n);
        for (int i = 0; i < n; ++i) {
            scores[i] = static_cast<double>(rng() % levels);
            labels[i] = static_cast<double>(rng() % 2);
        }
        std::vector<double> test;
        for (double s = -1.0; s <= levels + 1.0; s += 0.5) test.push_back(s);

        const auto cal = fast_ivap::fit(scores, labels);
        const auto [p0, p1] = fast_ivap::predict(cal, test);
        for (std::size_t t = 0; t < test.size(); ++t) {
            const auto [d0, d1] = ivap_by_definition(scores, labels, test[t]);
            if (std::abs(p0[t] - d0) > 1e-12 || std::abs(p1[t] - d1) > 1e-12) {
                std::printf("trial %d, s=%g: got (%g, %g), expected (%g, %g)\n", trial, test[t],
                            p0[t], p1[t], d0, d1);
                check(false, "agreement with definition");
                return;
            }
        }
    }
}

// Isotonic fitted values at every point of (x, y), ties in x pooled, in input order.
std::vector<double> isotonic_fit(const std::vector<std::pair<double, double>>& data) {
    std::vector<std::pair<double, double>> sorted = data;
    std::sort(sorted.begin(), sorted.end());
    std::vector<double> xs;
    for (std::size_t i = 0; i < data.size(); ++i) xs.push_back(data[i].first);
    std::vector<double> fitted(data.size());
    for (std::size_t i = 0; i < data.size(); ++i) fitted[i] = isotonic_at(sorted, xs[i]);
    return fitted;
}

// Blocks: calibration items whose fitted value equals that of the test item,
// compared with order[lo, hi); and the identity p_e = (sum + e) / (size + 1).
void test_blocks_against_definition() {
    std::mt19937 rng(54321);
    for (int trial = 0; trial < 500; ++trial) {
        const int n = 1 + static_cast<int>(rng() % 30);
        const int levels = 1 + static_cast<int>(rng() % 8);
        std::vector<double> scores(n), labels(n);
        for (int i = 0; i < n; ++i) {
            scores[i] = static_cast<double>(rng() % levels);
            labels[i] = static_cast<double>(rng() % 2);
        }
        std::vector<double> test;
        for (double s = -1.0; s <= levels + 1.0; s += 0.5) test.push_back(s);

        const auto cal = fast_ivap::fit(scores, labels);
        const auto [p0, p1] = fast_ivap::predict(cal, test);
        const auto blocks = fast_ivap::predict_blocks(cal, test);
        for (std::size_t t = 0; t < test.size(); ++t) {
            for (int e = 0; e <= 1; ++e) {
                std::vector<std::pair<double, double>> data;
                for (int i = 0; i < n; ++i) data.emplace_back(scores[i], labels[i]);
                data.emplace_back(test[t], static_cast<double>(e));
                const std::vector<double> fitted = isotonic_fit(data);

                const auto lo = e == 0 ? blocks.lo0[t] : blocks.lo1[t];
                const auto hi = e == 0 ? blocks.hi0[t] : blocks.hi1[t];
                std::vector<bool> in_block(n, false);
                double label_sum = 0.0;
                for (auto r = lo; r < hi; ++r) {
                    const auto i = static_cast<std::size_t>(cal.order[static_cast<std::size_t>(r)]);
                    in_block[i] = true;
                    label_sum += labels[i];
                }
                bool ok = true;
                for (int i = 0; i < n; ++i) {
                    const bool expected = std::abs(fitted[i] - fitted[n]) <= 1e-12;
                    ok = ok && expected == in_block[i];
                }
                const double p = e == 0 ? p0[t] : p1[t];
                ok = ok && std::abs(p - (label_sum + e) / static_cast<double>(hi - lo + 1)) <= 1e-15;
                if (!ok) {
                    std::printf("trial %d, s=%g, e=%d: block [%lld, %lld) is wrong\n", trial, test[t],
                                e, static_cast<long long>(lo), static_cast<long long>(hi));
                    check(false, "blocks agree with definition");
                    return;
                }
            }
        }
    }
}

void test_invalid_input() {
    auto throws = [](std::vector<double> s, std::vector<double> y) {
        try {
            fast_ivap::fit(s, y);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    check(throws({}, {}), "empty input throws");
    check(throws({1.0, 2.0}, {0.0}), "length mismatch throws");
    check(throws({1.0}, {0.5}), "non-binary label throws");
    check(throws({std::nan("")}, {1.0}), "NaN score throws");
}

void test_nan_test_score() {
    const auto cal = fast_ivap::fit(std::vector<double>{1.0, 2.0}, std::vector<double>{0.0, 1.0});
    const auto [p0, p1] = fast_ivap::predict(cal, std::vector<double>{std::nan("")});
    check(std::isnan(p0[0]) && std::isnan(p1[0]), "NaN test score gives NaN");
    const auto blocks = fast_ivap::predict_blocks(cal, std::vector<double>{std::nan("")});
    check(blocks.lo0[0] == -1 && blocks.hi0[0] == -1 && blocks.lo1[0] == -1 && blocks.hi1[0] == -1,
          "NaN test score gives -1 block ranges");
}

}  // namespace

int main() {
    test_against_definition();
    test_blocks_against_definition();
    test_invalid_input();
    test_nan_test_score();
    if (failures == 0) std::printf("all C++ tests passed\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
