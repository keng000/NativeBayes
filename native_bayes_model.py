# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import sklearn

class NativeBayes(object):
    def __init__(self):
        self.pY_ = None
        self.pXgY_ = None

    def fit(self,X,y):
        n_samples = X.shape[0]  # 訓練データ数
        n_features = X.shape[1] # 特徴量の数
        n_classes = 2 # クラス数
        n_fvalues = 2 # 特徴数

        if n_samples != len(y):
            # 訓練データの入力と出力のデータ数が一致していない場合、エラー
            raise ValueError("number of data between import and export is not matched")

        nY = np.zeros(n_classes,dtype=int)
        # 数え方に注意
        for i in xrange(n_samples):
            nY[y[i]] += 1

        self.pY_ = np.empty(n_classes,dtype=float)
        for i in xrange(n_classes):
            self.pY_[i] = nY[i] / float(n_samples)

        nXY = np.zeros((n_features,n_fvalues,n_classes),dtype=int)
        # 数え方に注意
        for i in xrange(n_samples):
            for j in xrange(n_features):
                nXY[j,X[i,j],y[i]] += 1

        self.pXgY_ = np.empty((n_features,n_fvalues,n_classes),dtype=float)
        for i in xrange(n_features):
            for j in xrange(n_fvalues):
                for k in xrange(n_classes):
                    self.pXgY_[i,j,k] = nXY[i,j,k] / float(nY[k])

    def predict(self,X):
        n_samples = X.shape[0]
        n_features = X.shape[1]

        y = np.empty(n_samples,dtype=int)

        for i ,xi in enumerate(X):
            logpXY = np.log(self.pY_)

            for j in xrange(n_features):
                logpXY += np.log(self.pXgY_[j,xi[j],:])

            y[i] = np.argmax(logpXY)

        return y
