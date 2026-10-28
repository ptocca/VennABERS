# Changelog

## 1.1.0 - 2026-10-28

### Added
- `IVAP.predict_blocks(scores)` returns the isotonic block (Venn-ABERS
  category) of each test example for labels 0 and 1, as ranges
  `(lo0, hi0, lo1, hi1)` of the calibration examples sorted by score.
- `IVAP.order_`, the stable argsort of the calibration scores, mapping these
  ranges to calibration indices.
- `IVAP.block_sums(scores, values)` sums per-example values (for example a
  one-hot multiclass label) over the blocks with prefix sums, for multiclass
  Venn predictors.
- C++: `fast_ivap::predict_blocks`, `Calibrator::order` and
  `Calibrator::blocks`.

### Changed
- `fit` also computes the block tables. `p0` and `p1` are bit-for-bit
  unchanged; fitting is up to about 30% slower (1M distinct scores).

## 1.0.0 - 2026-10-27

Complete rewrite, published under the name `fast-ivap` (import `fast_ivap`).

### Fixed
- `p0` and `p1` were wrong for a test score equal to one of the calibration
  scores.
- `p1` was wrong for a test score above all calibration scores (it must be 1).
- The pure-Python implementation failed on NumPy 2 (`np.cross` on 2-D vectors).
- Removed the module-level global state, so separate predictors no longer
  interfere with each other.

### Changed
- New API: `fast_ivap.IVAP` (`fit`, `predict_p0_p1`, `predict_proba`) and
  `fast_ivap.sklearn.VennAbersCalibrator`, whose `predict_proba` now returns
  an `(n, 2)` array as scikit-learn expects. `ScoresToMultiProbs` is kept as a
  deprecated shim.
- The C++ code is now a dependency-free header-only library, with nanobind
  bindings (zero-copy NumPy input, GIL released) built by scikit-build-core.
- Portable wheels for Linux, macOS and Windows via cibuildwheel.

### Added
- `merge_log_minimax` (IVAP and CVAP) and `merge_brier`.
- pytest suite checking both backends against IVAP computed from its
  definition; C++ tests via CTest; CI on Linux, macOS and Windows.

## 0.2 - 2020-07
- Fixed bug affecting the `p0` calculation; added a test notebook.
- (2024, branches `cpp_version` / `pip_package`) C++ port with pybind11 and a
  scikit-learn style wrapper.

## 0.1
- Initial pure-Python implementation.
