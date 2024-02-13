import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import VennABERSlib

class VennABERS(BaseEstimator, ClassifierMixin):
    def __init__(self, demo_param='demo'):
        self.demo_param = demo_param
    def fit(self, X, y):
        """X: scores
           y: label values"""
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.X_ = X
        self.y_ = y

        # compute the F0 and F1 functions from the CSD
        self.F0,self.F1,self.ptsUnique = VennABERSlib.train(self.X_.reshape(-1,), self.y_)

        self.is_fitted_ = True

        return self

    def predict_proba(self, X):
        # Check if fit has been called
        check_is_fitted(self)
        # Input validation
        X = check_array(X)
        p0, p1 = VennABERSlib.predict(self.F0, self.F1, self.ptsUnique, X.reshape(-1,) )
        return np.array(p0), np.array(p1)
