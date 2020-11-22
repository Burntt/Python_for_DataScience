import numpy as np 
import pandas as pd
import nltk
import unicodedata
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from nltk.tokenize import word_tokenize

from experiments.base.transformer import BaseTransformer


class Transformer_5_1(BaseTransformer):
    def __init__(self):
        nltk.download('punkt')
        return None
    
    def number_of_words(self, sentence):
        sentence_array = word_tokenize(sentence)
        words = [word for word in sentence_array if word.isalpha()]
        return len(words)
    
    def number_of_chars(self, sentence):
        return len(sentence)

    def number_of_hashtags(self, sentence):
        return len(re.findall('#\w+', sentence))
    
    def fit_transform(self, X):
        X['number_of_words'] = X['text'].apply(self.number_of_words)
        X['number_of_chars'] = X['text'].apply(self.number_of_chars)
        X['number_of_hashtags'] = X['text'].apply(self.number_of_hashtags)

        poly = preprocessing.PolynomialFeatures(degree=3)
        transformed_df = poly.fit_transform(X.drop('text', axis=1))
        return transformed_df
