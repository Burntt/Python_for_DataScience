import numpy as np 
import pandas as pd
import nltk
import unicodedata
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from nltk import pos_tag
from nltk import word_tokenize

from experiments.base.transformer import BaseTransformer

class Transformer_4_1(BaseTransformer):
    def __init__(self):
        return None

    def fit_transform(self, X):
        Here I would have performed the below part if I had the time
        
# Create a custom classifier with pipeline of polynomial feature generation for numeric features, PCA dimensionality reduction and linear regression with parameters of regularization l1

# Perform gridSearch for optimal classifier parameters:
# -- polynomial degree from 4 to 9

# -- PCA degreee from 35 to 45

# -- l1 regularization 0.098 to 0.196

# Measure quality results on test subsample with metric adjusted_mutual_info_score and store result in .yaml file
