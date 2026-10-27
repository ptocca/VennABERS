# fast-ivap: fast inductive Venn-ABERS predictors

[![CI](https://github.com/ptocca/VennABERS/actions/workflows/ci.yml/badge.svg)](https://github.com/ptocca/VennABERS/actions/workflows/ci.yml)

`fast-ivap` turns the scores of any binary classifier into **well-calibrated
probabilities**, with no assumption beyond the calibration and test data being
i.i.d. It implements the inductive Venn-ABERS predictor (IVAP) with the
O(k log k) algorithm of

> V. Vovk, I. Petej, V. Fedorova. *Large-scale probabilistic predictors with
> and without guarantees of validity.* NeurIPS 2015.
> [arXiv:1511.00213](https://arxiv.org/abs/1511.00213)

The core is a small header-only C++20 library with Python bindings
([nanobind](https://github.com/wjakob/nanobind)), plus a pure-Python
reference implementation. A gentle introduction to Venn-ABERS predictors is in
this [tutorial](https://cml.rhul.ac.uk/people/ptocca/HomePage/Toccaceli_CP___Venn_Tutorial.pdf).

## What it computes

Given calibration scores and labels, and the score `s` of a test object, an
IVAP outputs two probabilities that the label is 1:

- `p0`: the isotonic-regression fit at `s` after adding `(s, 0)` to the
  calibration set,
- `p1`: the same after adding `(s, 1)`.

`p0 <= p1` always holds, and one of the two (which one depends on the unknown
true label) is perfectly calibrated. The gap `p1 - p0` is typically small and
shrinks as the calibration set grows. A single probability can be
obtained with the log-loss minimax merge

$$p = \frac{p_1}{1 - p_0 + p_1}.$$

Fitting takes O(k log k) for k calibration examples; each prediction takes
O(log k).

## Installation

```bash
pip install fast-ivap              # NumPy API
pip install "fast-ivap[sklearn]"   # plus the scikit-learn estimator
```

Until the package is on PyPI, install from GitHub (this builds from source, so
it needs a C++20 compiler; CMake is fetched automatically if missing):

```bash
pip install git+https://github.com/ptocca/VennABERS
```

Prebuilt wheels are produced for Linux (x86-64, aarch64; glibc and musl),
macOS (Intel and Apple silicon) and Windows (x86-64, ARM64), for CPython 3.11
and a single stable-ABI wheel for CPython 3.12 and later.

## Usage

### NumPy

```python
from fast_ivap import IVAP

ivap = IVAP().fit(cal_scores, cal_labels)  # labels are 0/1 or booleans
p0, p1 = ivap.predict_p0_p1(test_scores)  # Venn-ABERS pair
p = ivap.predict_proba(test_scores)  # merged probability of label 1
```

See [`examples/quickstart.py`](examples/quickstart.py) for a complete example
with a scikit-learn classifier.

For a cross Venn-ABERS predictor (CVAP), fit one IVAP per fold, stack the
outputs and merge them with geometric means:

```python
from fast_ivap import merge_log_minimax

p = merge_log_minimax(p0_folds, p1_folds, axis=0)  # arrays of shape (K, n)
```

`merge_brier` gives the Brier-loss minimax merge instead.

### scikit-learn

`VennAbersCalibrator` takes one score per example and follows the
scikit-learn classifier API (`predict_proba` returns `[1 - p, p]`):

```python
from fast_ivap.sklearn import VennAbersCalibrator

cal = VennAbersCalibrator().fit(clf.decision_function(X_cal), y_cal)
proba = cal.predict_proba(clf.decision_function(X_test))
p0, p1 = cal.predict_p0_p1(clf.decision_function(X_test))
```

### C++

The C++ library is the single header
[`include/fast_ivap/ivap.hpp`](include/fast_ivap/ivap.hpp) with no
dependencies. Copy it into your project, or install it and use CMake:

```bash
cmake -S . -B build && cmake --install build --prefix /your/prefix
```

```cmake
find_package(fast_ivap REQUIRED)
target_link_libraries(my_app PRIVATE fast_ivap::fast_ivap)
```

```cpp
#include <fast_ivap/ivap.hpp>

auto cal = fast_ivap::fit(scores, labels);           // any contiguous range of doubles
auto [p0, p1] = fast_ivap::predict(cal, test_scores);
```

## Implementation notes

- The C++ backend is used when available. `IVAP(backend="python")` or the
  environment variable `FAST_IVAP_BACKEND=python` select the pure-Python
  implementation in [`_reference.py`](src/fast_ivap/_reference.py), which
  follows the paper step by step. Both are tested against each other and
  against a brute-force IVAP computed from its definition.
- `p1` is computed with Algorithms 1 and 2 of the paper. Instead of the
  mirror-image Algorithms 3 and 4, `p0` is obtained from the same code by the
  symmetry noted in the paper: negate the scores, flip the labels, and take
  `1 - p1`.
- The paper defines the extra CSD point for Algorithms 3–4 as
  `P[k'+1] = P[k'] + (1, 1)`; it should be `P[k'] + (1, 0)`, because that
  segment represents the test object with label 0. The original
  implementation already used `(1, 0)`.
- A test score equal to a calibration score, and a test score outside the
  range of the calibration scores, are handled exactly as in the definition
  (two bugs in versions up to 0.2 affected these cases).
- A NaN test score yields NaN probabilities. Calibration scores must be finite.

## Development

```bash
uv sync --group dev      # build the extension and install dev tools
uv run pytest            # Python tests (both backends)
uv run ruff check . && uv run ruff format --check .

cmake -S . -B build && cmake --build build && ctest --test-dir build   # C++ tests
```

The GitHub Actions workflows run the tests on Linux, macOS and Windows, and
build wheels with [cibuildwheel](https://cibuildwheel.pypa.io/). Publishing a
GitHub release uploads the wheels and sdist to PyPI (trusted publishing).

## Version history

See [CHANGELOG.md](CHANGELOG.md).

## Acknowledgements

- The algorithm was invented by Prof. Vladimir Vovk at the
  [Centre for Reliable Machine Learning, Royal Holloway, University of London](https://cml.rhul.ac.uk/).
- This work was supported in part by funding from the European Union's Horizon
  2020 Research and Innovation programme under Grant Agreement no. 671555
  (ExCAPE).
- We are grateful to the Ministry of Education, Youth and Sports (Czech
  Republic) that supports the Large Infrastructures for Research, Experimental
  Development and Innovations project "IT4Innovations National Supercomputing
  Center - LM2015070" for providing access to the Solomon supercomputing
  cluster.
