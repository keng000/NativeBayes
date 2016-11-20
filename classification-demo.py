# -*- coding: utf-8 -*-
from native_bayes_model import NativeBayes
import numpy as np

data = np.genfromtxt('vote_filled.tsv',dtype=int)

X = data[:,:-1]
y = data[:, -1]

CL = NativeBayes()
CL.fit(X,y)

predict_y = CL.predict(X[:10,:])

for i in xrange(10):
    print i,y[i],predict_y[i]
