import numpy as np 
import pandas as pd
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from nltk import pos_tag
from nltk import word_tokenize
import xgboost as xgb

from experiments.base.transformer import BaseTransformer

class Transformer_3_1(BaseTransformer):
    def __init__(self):
        nltk.download('tagsets')
        return None

    def fit_transform(self, X):
        D_train = xgb.DMatrix(X)
        param = {
            'eta': 0.3, 
            'max_depth': 3,  
            'objective': 'multi:softprob',  
            'num_class': 3} 
        steps = 20  # The number of training iterations
        transformer_df = xgb.train(param, D_train, steps)
        return transformer_df
    
    
