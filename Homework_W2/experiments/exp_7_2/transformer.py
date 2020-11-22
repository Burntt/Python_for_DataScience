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


class Transformer_7_2(BaseTransformer):
    def __init__(self):
        nltk.download('tagsets')
        return None

    def fit_transform(self, X):        
        X['text'] = X['text'].apply(self.clean_text)
        X['tokens'] = X['text'].apply(word_tokenize)
        X['pos_tokens'] = X['tokens'].apply(pos_tag)
        X['nn_tokens'] = X['pos_tokens'].apply(lambda token_list: [token[0] for token in token_list if token[1] == 'NN'])
        X['nn_tokens'] = X['nn_tokens'].apply(lambda x: ' '.join(x))
        
        vectorizer = CountVectorizer()
        transformer_df = vectorizer.fit_transform(X['nn_tokens'])
        return transformer_df
