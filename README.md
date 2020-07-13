# Venn-ABERS Predictor
*(Preliminary documentation)*

The `VennABERS.py` file is a pure Python implementation of the fast Venn-ABERS Predictor described in
[Vovk2015](http://alrw.net/articles/13.pdf).

A Venn-ABERS predictor outputs two probability predictions for every test object.
In particular, the Venn-ABERS predictor implemented here is the inductive form of probability predictor, which relies on a calibration set.
In a nutshell, the Venn-ABERS predictor can be viewed as a distribution-free calibration function that maps scores output by a *scoring classifier* to well-calibrated probabilities.
A gentle introduction can be found in the tutorial [Toccaceli2017](http://clrc.rhul.ac.uk/copa2017/presentations/VennTutorialCOPA2017.pdf).

The function that implements the Venn-ABERS Predictor is `ScoresToMultiProbs()`.

```python
p0,p1 = ScoresToMultiProbs(calibrPts,testScores)
```

calibrPts: a list of pairs (score,label) corresponding to the scores and labels of the calibration examples. The score is a float and the label is an integer  meant to take values 0 or 1.

testScores: a list of floats corresponding to the scores for the test objects.

The function returns a pair of Numpy arrays with the probabilistic predictions.

## Version History
    - 0.1 - Initial implementation
    - 0.2 - 2020-07 Fixed bug affecting p_0 calculation, added test notebook

## Acknowlegdements
* Work done with funding from the European Union`s Horizon 2020 Research and Innovation programme under Grant Agreement no. 671555 (ExCAPE). 
* We are grateful for the help in conducting experiments to the Ministry of Education, Youth and Sports (Czech Republic) that supports the Large Infrastructures for Research, Experimental Development and Innovations project "IT4Innovations National Supercomputing Center - LM2015070".
