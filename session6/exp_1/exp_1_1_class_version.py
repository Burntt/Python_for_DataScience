
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.base import BaseEstimator, ClassifierMixin

class Exp1_1Classifier(BaseEstimator, ClassifierMixin):
    
    classifier = None
    fit_classifier = None
    
    
    def __init__(self):
        self.classifier = linear_model.LogisticRegression(
            penalty = 'l2',
            random_state = 2019
        )
    
    def feature_transformation(self, train_df):

        numeric_features_sudataset = train_df[ 
            ['HP', 'Attack','Defense', 'Sp_Atk', 'Sp_Def', 'Speed'] ]
        numeric_features_sudataset

        preprocessor_exp_1_1 = preprocessing.PolynomialFeatures(
            degree = 4)

        polynomial_features_pokemon = \
            preprocessor_exp_1_1.fit_transform(numeric_features_sudataset)

        return polynomial_features_pokemon



    def fit(self, train_df):

        transformed_features = self.feature_transformation(train_df)
        self.fit_classifier = self.classifier.fit(
            transformed_features, train_df['HighStage'])

    def predict(self, test_df):

        transformed_features = self.feature_transformation(test_df)
        return self.fit_classifier.predict(transformed_features)
    
    def predict_proba(self, test_df):
        transformed_features = self.feature_transformation(test_df)
        return self.fit_classifier.predict_proba(transformed_features)