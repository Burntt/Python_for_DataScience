import pandas as pd
import numpy as np
import pickle
from experiments.exp_7_4.transformer import Transformer_7_4
from experiments.base.classifier import LogisticCustomClassifier

X_train, X_test, y_train, y_test = pickle.load(open('./split.pickle', 'rb'))

features = 'screenName'
target = 'Bin_High_Retweet_Count'

classifier = LogisticCustomClassifier('experiments/exp_7_4/config.yaml', Transformer_7_4, 'experiments/exp_7_4/result/')
classifier.fit(X_train, y_train[target])
y_pred = classifier.predict(X_test)
classifier.generate_results(y_pred, y_test[target])
