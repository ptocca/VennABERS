"""Calibrate a classifier's scores with an inductive Venn-ABERS predictor.

Run with:  python examples/quickstart.py   (needs scikit-learn)
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from fast_ivap import IVAP

X, y = make_classification(n_samples=20_000, n_features=20, flip_y=0.1, random_state=0)
X_train, X_rest, y_train, y_rest = train_test_split(X, y, train_size=0.5, random_state=0)
X_cal, X_test, y_cal, y_test = train_test_split(X_rest, y_rest, train_size=0.5, random_state=0)

# 1. Train any scoring classifier on the proper training set.
clf = LinearSVC().fit(X_train, y_train)

# 2. Fit the IVAP on the classifier's scores for a held-out calibration set.
ivap = IVAP().fit(clf.decision_function(X_cal), y_cal)

# 3. Predict: a pair (p0, p1) per test example, or a single merged probability.
test_scores = clf.decision_function(X_test)
p0, p1 = ivap.predict_p0_p1(test_scores)
p = ivap.predict_proba(test_scores)

print("first five test examples:")
for s, a, b, q, label in zip(test_scores[:5], p0, p1, p, y_test, strict=False):
    print(f"  score {s:+.2f}  p0 {a:.3f}  p1 {b:.3f}  p {q:.3f}  true label {label}")

print(f"\nmean width p1 - p0:   {np.mean(p1 - p0):.4f}")
print(f"Brier score:          {np.mean((p - y_test) ** 2):.4f}")
print(f"mean predicted prob.: {p.mean():.4f}   observed frequency: {y_test.mean():.4f}")
