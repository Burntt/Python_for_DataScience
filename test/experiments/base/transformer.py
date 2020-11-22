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
    
    def clean_text(self, text):
        wnl = nltk.stem.WordNetLemmatizer()
        stopwords = nltk.corpus.stopwords.words('english') + ['RT']
        text = (unicodedata.normalize('NFKD', text)
            .encode('ascii', 'ignore')
            .decode('utf-8', 'ignore'))
        words = re.sub(r'[^\w\s]', '', text) 
        words = re.sub(r'(http.+)', '', words).split()
        return ' '.join([word for word in words if word not in stopwords])
