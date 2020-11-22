
from exp_1.exp_1_1_Classifier import Exp1_1Classifier

import yaml

from sklearn import preprocessing
from sklearn import linear_model

class Exp1_3Classifier(Exp1_1Classifier):
    
    configs = None
    
    def __init__(self, yaml_config_file):
        
        with open(yaml_config_file) as f:
            self.configs = yaml.load(f, Loader=yaml.Loader)
    
        
        self.classifier = linear_model.LogisticRegression(
            penalty = self.configs['regularization'],
            random_state = 2019
        )
        print('Usign configuration file %s:'%yaml_config_file)
        print(self.configs)
        
    
    def feature_transformation(self, train_df):

        numeric_features_sudataset = train_df[  self.configs['features'] ]
        numeric_features_sudataset

        preprocessor_exp_1_1 = preprocessing.PolynomialFeatures(
            degree = self.configs['polynomial_degree'])

        polynomial_features_pokemon = preprocessor_exp_1_1.fit_transform(numeric_features_sudataset)

        return polynomial_features_pokemon

    