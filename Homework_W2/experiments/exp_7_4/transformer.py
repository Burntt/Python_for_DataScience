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


class Transformer_7_4(BaseTransformer):
    def __init__(self):
        nltk.download('tagsets')
        return None

    def fit_transform(self, X):
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 2))
        transformer_df = vectorizer.fit_transform(X['screenName'])
        return transformer_df
