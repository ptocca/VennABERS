// Python bindings for the fast IVAP core (module fast_ivap._core).
//
// The functions mirror fast_ivap._reference: input arrays are borrowed
// without copying, and results are returned as NumPy arrays that own their
// memory. The GIL is released while the C++ code runs.

#include <fast_ivap/ivap.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <span>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace {

using InArray = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using OutArray = nb::ndarray<nb::numpy, double, nb::ndim<1>>;

std::span<const double> as_span(const InArray& a) { return {a.data(), a.shape(0)}; }

// Hand a std::vector over to NumPy without copying.
OutArray to_numpy(std::vector<double>&& v) {
    auto* owned = new std::vector<double>(std::move(v));
    nb::capsule deleter(owned, [](void* p) noexcept { delete static_cast<std::vector<double>*>(p); });
    return OutArray(owned->data(), {owned->size()}, deleter);
}

nb::tuple fit(const InArray& scores, const InArray& labels) {
    fast_ivap::Calibrator cal;
    {
        nb::gil_scoped_release release;
        cal = fast_ivap::fit(as_span(scores), as_span(labels));
    }
    return nb::make_tuple(to_numpy(std::move(cal.scores)), to_numpy(std::move(cal.f0)),
                          to_numpy(std::move(cal.f1)));
}

nb::tuple predict(const InArray& scores, const InArray& f0, const InArray& f1,
                  const InArray& test_scores) {
    const std::size_t k = scores.shape(0);
    if (k == 0 || f0.shape(0) != k + 1 || f1.shape(0) != k + 1) {
        throw std::invalid_argument("inconsistent IVAP lookup tables");
    }
    std::vector<double> p0(test_scores.shape(0));
    std::vector<double> p1(test_scores.shape(0));
    {
        nb::gil_scoped_release release;
        fast_ivap::predict({as_span(scores), as_span(f0), as_span(f1)}, as_span(test_scores), p0,
                           p1);
    }
    return nb::make_tuple(to_numpy(std::move(p0)), to_numpy(std::move(p1)));
}

}  // namespace

NB_MODULE(_core, m) {
    m.doc() = "C++ backend of fast_ivap (see fast_ivap._reference for the contract).";
    m.def("fit", &fit, nb::arg("scores"), nb::arg("labels"),
          "Return (unique_scores, f0, f1) lookup tables for calibration data.");
    m.def("predict", &predict, nb::arg("unique_scores"), nb::arg("f0"), nb::arg("f1"),
          nb::arg("test_scores"), "Return (p0, p1) for each test score.");
}
