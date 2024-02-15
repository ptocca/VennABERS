# Venn-ABERS Predictor
*(Preliminary documentation)*

This repository provides a Python implementation of the Venn-ABERS Predictor described in
[Vovk2015](http://alrw.net/articles/13.pdf).

In a nutshell, the Venn-ABERS Predictor can be viewed as a non-parametric distribution-free calibration function that maps *scores* (values output by a *scoring classifier*) to well-calibrated probabilities (corresponding to long-term frequencies under iid assumptions).    
A gentle introduction can be found in this [tutorial](https://cml.rhul.ac.uk/people/ptocca/HomePage/Toccaceli_CP___Venn_Tutorial.pdf).    
The Venn-ABERS predictor is calibrated with a matrix `(N,1)` of scores and a vector `(N,)` of labels taking values `0` or `1`.
Once calibrated, the predictor outputs a pair of vectors `(p0,p1)` given a matrix `(N,1)` of scores.
`p0`and `p1`represent a multi-valued probabilistic prediction. They both express the probability of the actual label being `1`, but under different hypotheses.

The implementation is in C++ and Python. The API conforms to the `scikit-learn` probabilistic predictor API.

```python
import VennABERS
vap = VennABERS.VennABERS()
VennABERS.fit(X,y)
p0,p1 = VennABERS.predict_proba(X_test)
```

The implementation uses the `pybind11` package for the Python-C++ bindings.

Under the `test` directory, you can find a Jupyter notebook that checks the correctness of the implementation.

On Linux systems, the `VennABERS` module can be installed with the following command:
```
pip install git+https://github.com/ptocca/VennABERS@pip_package
```
(It appears to work also on MacOS.)

A PyPi package will be provided in due course.


## Acknowlegdements
* The algorithm was invented by Prof. Vovk at the [Centre for Reliable Machine Learning, Royal Holloway, Univ. of London](https://cml.rhul.ac.uk/).
* This work was supported in part by funding from the European Union`s Horizon 2020 Research and Innovation programme under Grant Agreement no. 671555 (ExCAPE). 
* We are grateful to the Ministry of Education, Youth and Sports (Czech Republic) that supports the Large Infrastructures for Research, Experimental Development and Innovations project "IT4Innovations National Supercomputing Center - LM2015070" for providing access to the Solomon supercomputing cluster.