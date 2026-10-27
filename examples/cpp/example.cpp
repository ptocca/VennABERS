// Minimal example of using the header-only C++ library.
//
//   cmake -B build && cmake --build build && ./build/ivap_example

#include <fast_ivap/ivap.hpp>

#include <cstdio>
#include <vector>

int main() {
    const std::vector<double> scores{2.0, 3.0, 4.0, 5.0, 7.0, 8.0};
    const std::vector<double> labels{0.0, 1.0, 0.0, 0.0, 1.0, 1.0};
    const std::vector<double> test{1.0, 2.5, 7.2};

    const fast_ivap::Calibrator cal = fast_ivap::fit(scores, labels);
    const auto [p0, p1] = fast_ivap::predict(cal, test);

    for (std::size_t i = 0; i < test.size(); ++i) {
        std::printf("score %4.1f  ->  p0 = %.3f, p1 = %.3f\n", test[i], p0[i], p1[i]);
    }
}
