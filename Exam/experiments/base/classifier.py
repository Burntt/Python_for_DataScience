
from sklearn import preprocessing, decomposition, linear_model
from sklearn.base import BaseEstimator, ClassifierMixin
from datetime import datetime
import scipy
import numpy as np
import yaml
from sklearn import metrics

class LogisticCustomClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, yaml_config_file, transformer, result_folder_path):
        with open(yaml_config_file) as f:
            self.configs = yaml.load(f, Loader=yaml.Loader)
        
        self.classifier = linear_model.LogisticRegression(
            penalty = self.configs['logistic_penalty'],
            random_state = self.configs['logistic_rand_state'],
            solver = 'saga'
        )
        
        self.transformer = transformer
        
        if self.configs['pca_components'] != None:
            self.n_components = self.configs['pca_components']
        else:
            self.n_components = None
        
        self.result_folder_path = result_folder_path
        
        print('Using configuration file %s:'%yaml_config_file)
        
    
    def feature_transformation(self, X):        
        # custom transformation
        t = self.transformer()
        transformed_X = t.fit_transform(X)
        
        # check sparse
        if scipy.sparse.issparse(transformed_X):
            transformed_X = transformed_X.todense()
        
       # scale data / optional, better to do it separately  
        scaler = preprocessing.StandardScaler(with_mean=False)
        transformed_df = scaler.fit_transform(transformed_X)

        pca = decomposition.PCA()
        transformed_df = pca.fit_transform(transformed_X)
        
        # select components with sum variance > 0.9 to determine the n_components var
        if self.n_components == 'None':
            ratios_sum = 0
            ratios_index = 0
            ratios = list(pca.explained_variance_ratio_)

            while ratios_sum < 0.9 and ratios_index < len(ratios) - 1:
                ratios_index += 1
                ratios_sum += ratios[ratios_index]
                
            self.n_components = ratios_index
        
        # PCA reduction
        pca = decomposition.PCA(n_components=self.n_components)
        transformed_X = pca.fit_transform(transformed_X)
        
        return transformed_X
    
    def generate_results(self, y_pred, y_test):
        results = {
            'values': {
                'pca_components': self.n_components,
                'accuracy': metrics.accuracy_score(y_pred, y_test).item(),
                'precision': metrics.precision_score(y_pred, y_test).item(),
                'recall': metrics.recall_score(y_pred, y_test).item(),
                'mutual_info_score': metrics.adjusted_mutual_info_score(y_pred, y_test).item()
            },
            'config': self.configs
        }
        
        time_obj = datetime.now()
        file_path = self.result_folder_path + 'result_' + time_obj.strftime("%d-%b-%Y-%H-%M-%S") + '.yaml'

        with open(file_path, 'w') as file:
            yaml.dump(results, file)
            
        print('------------------')
        print(results['values'])
        print('------------------')
        print("Results are saved to:", file_path)
        
    
    def fit(self, X, y):
        transformed_features = self.feature_transformation(X)
        self.fit_classifier = self.classifier.fit(transformed_features, y)

    def predict(self, X):
        transformed_features = self.feature_transformation(X)
        return self.fit_classifier.predict(transformed_features)
