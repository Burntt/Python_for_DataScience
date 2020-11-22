import pandas as pd
import numpy as np
import pickle
from experiments.exp_3_1.transformer import Transformer_3_1
from experiments.base.classifier import LogisticCustomClassifier

X_train, X_test, y_train, y_test = pickle.load(open('./data/split.pickle', 'rb'))

features = ['Characteristic Path Length', 'Avg.num.Neighbours',
           'NeighborhoodConnectivity', 'Outdegree', 'Stress',
           'PartnerOfMultiEdgedNodePairs', 'EdgeCount', 'BetweennessCentrality',
           'Indegree', 'Eccentricity', 'ClosenessCentrality',
           'AverageShortestPathLength', 'ClusteringCoefficient']

target = ['Many_Neighbours']

classifier = LogisticCustomClassifier('experiments/exp_3_1/config.yaml', Transformer_3_1, 'experiments/exp_3_1/result/')

classifier.fit(X_train[features], y_train[target])
y_pred = classifier.predict(X_test[features])
#classifier.generate_results(y_pred, y_test[target])
