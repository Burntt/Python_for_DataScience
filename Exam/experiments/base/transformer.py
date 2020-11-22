import numpy as np 
import pandas as pd
import nltk
import unicodedata
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer

class BaseTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None

    def fit_transform(self, X):
        return X
