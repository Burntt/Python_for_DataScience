### Custom Transformer for 1-gram tokenization
import numpy as np 
import pandas as pd
import nltk
import unicodedata
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer

from experiments.base.transformer import BaseTransformer


class Transformer_7_1(BaseTransformer):
    def __init__(self):
        return None

    def fit_transform(self, X):
        X = X.apply(self.clean_text)
        vectorizer = CountVectorizer()
        transformer_df = vectorizer.fit_transform(X)
        return transformer_df
