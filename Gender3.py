import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing , svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style

from svmutil import *

df = pd.read_csv('train - Copy - Copy.csv')
df = shuffle(df)
X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X = preprocessing.normalize(X)
X = X.tolist()
y=df['label'].tolist()

Xtrain=X[1:2217]
Ytrain=y[1:2217]
Xtest=X[2218:-20]
Ytest=y[2218:-20]
Xtry=X[-20:]
Ytry=y[-20:]

print(len(Xtrain))
print(len(Xtest))
print(len(Xtry))

m = svm_train(Ytrain, Xtrain,'-s 0 -t 2 -g 0.05 -c 10  -q');
[predTrain,accuracyTrain,probabilities] = svm_predict(Ytrain, Xtrain, m);
print(accuracyTrain)

[predTest,accuracyTest,probabilities] = svm_predict(Ytest, Xtest, m);
print(accuracyTest)

[predTest,accuracyTest,probabilities] = svm_predict(Ytry, Xtry, m);
print(accuracyTest)
