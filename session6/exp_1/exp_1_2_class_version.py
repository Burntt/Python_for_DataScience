
from exp_1.exp_1_1_class_version import Exp1_1Classifier

from sklearn import linear_model

class Exp1_2Classifier(Exp1_1Classifier):
    
    def __init__(self):
        self.classifier = linear_model.LogisticRegression(
            penalty = 'l1',
            random_state = 2019
        )
        print('Exp_1_2 classifier initialized') 