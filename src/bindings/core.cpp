// Python bindings for the fast IVAP core (module fast_ivap._core).
//
// The functions mirror fast_ivap._reference: input arrays are borrowed
// without copying, and results are returned as NumPy arrays that own their
// memory. The GIL is released while the C++ code runs.

#include <fast_ivap/ivap.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cstdint>
#include <initializer_list>
#include <span>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace {

using InArray = nb::ndarray<const double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using BlocksArray = nb::ndarray<const std::int64_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

std::span<const double> as_span(const InArray& a) { return {a.data(), a.shape(0)}; }

// Hand a std::vector over to NumPy without copying, as an array of the given
// shape (1-D by default).
template <typename T>
nb::ndarray<nb::numpy, T> to_numpy(std::vector<T>&& v, std::initializer_list<std::size_t> shape) {
    auto* owned = new std::vector<T>(std::move(v));
    nb::capsule deleter(owned, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
    return nb::ndarray<nb::numpy, T>(owned->data(), shape, deleter);
}

template <typename T>
nb::ndarray<nb::numpy, T> to_numpy(std::vector<T>&& v) {
    const std::size_t size = v.size();
    return to_numpy(std::move(v), {size});
}

// Stack four equally long vectors into a (4, m) array with rows lo0, hi0, lo1, hi1.
nb::ndarray<nb::numpy, std::int64_t> stack_blocks(const fast_ivap::Blocks& b) {
    const std::size_t m = b.lo0.size();
    std::vector<std::int64_t> out;
    out.reserve(4 * m);
    for (const auto* row : {&b.lo0, &b.hi0, &b.lo1, &b.hi1}) {
        out.insert(out.end(), row->begin(), row->end());
    }
    return to_numpy(std::move(out), {4, m});
}

nb::tuple fit(const InArray& scores, const InArray& labels) {
    fast_ivap::Calibrator cal;
    {
        nb::gil_scoped_release release;
        cal = fast_ivap::fit(as_span(scores), as_span(labels));
    }
    auto blocks = stack_blocks(cal.blocks);
    return nb::make_tuple(to_numpy(std::move(cal.scores)), to_numpy(std::move(cal.f0)),
                          to_numpy(std::move(cal.f1)), to_numpy(std::move(cal.order)), blocks);
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

nb::ndarray<nb::numpy, std::int64_t> predict_blocks(const InArray& scores,
                                                    const BlocksArray& blocks,
                                                    const InArray& test_scores) {
    const std::size_t k = scores.shape(0);
    if (k == 0 || blocks.shape(0) != 4 || blocks.shape(1) != k + 1) {
        throw std::invalid_argument("inconsistent IVAP block tables");
    }
    const std::size_t m = test_scores.shape(0);
    std::vector<std::int64_t> out(4 * m);
    {
        nb::gil_scoped_release release;
        const std::int64_t* t = blocks.data();
        const std::span<std::int64_t> o(out);
        fast_ivap::predict_blocks(
            {as_span(scores), {t, k + 1}, {t + (k + 1), k + 1}, {t + 2 * (k + 1), k + 1},
             {t + 3 * (k + 1), k + 1}},
            as_span(test_scores), o.subspan(0, m), o.subspan(m, m), o.subspan(2 * m, m),
            o.subspan(3 * m, m));
    }
    return to_numpy(std::move(out), {4, m});
}

}  // namespace

NB_MODULE(_core, m) {
    m.doc() = "C++ backend of fast_ivap (see fast_ivap._reference for the contract).";
    m.def("fit", &fit, nb::arg("scores"), nb::arg("labels"),
          "Return (unique_scores, f0, f1, order, blocks) lookup tables for calibration data.");
    m.def("predict", &predict, nb::arg("unique_scores"), nb::arg("f0"), nb::arg("f1"),
          nb::arg("test_scores"), "Return (p0, p1) for each test score.");
    m.def("predict_blocks", &predict_blocks, nb::arg("unique_scores"), nb::arg("blocks"),
          nb::arg("test_scores"),
          "Return a (4, m) array of block ranges lo0, hi0, lo1, hi1 for each test score.");
}
