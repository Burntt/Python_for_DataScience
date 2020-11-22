
from sklearn import preprocessing
from sklearn import linear_model
from exp_1.exp_1_1 import feature_transformation, train, predict, classifier, fit_classifier
from exp_1 import exp_1_1

def init_classifier():
#     global classifier
    exp_1_1.classifier = linear_model.LogisticRegression(
        penalty = 'l1',
        random_state = 2022
    )
    print('Exp_1_2 classifier initialized')

def predict_proba(self, test_df):
    transformed_features = self.feature_transformation(test_df)
    return self.fit_classifier.predict_proba(transformed_features)