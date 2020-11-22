import numpy as np 
import pandas as pd
import nltk
import unicodedata
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import PolynomialFeatures
from nltk import pos_tag
from nltk import word_tokenize

from experiments.base.transformer import BaseTransformer


class Transformer_7_5(BaseTransformer):
    def __init__(self):
        return None

    def fit_transform(self, X):
        X = X[['text']]

        heroes = ["Iron Man", "Thor", "Captain America", "Captain Marvel", "Black Widow", "Hawkeye", "Hulk", "Vision", "Scarlet Witch", "War Machine", "Falcon", "StarLord", "Rocket Raccoon", "Groot", "Gamora", "Drax", "Mantis", "Nebula", "Doctor Strange", "Wong", "Spider-Man", "Spiderman", "Winter Soldier", "Heimdall", "Black Panther", "Okoye", "Shuri", "M’Baku", "Eitiri", "Nick Fury", "Maria Hill", "Pepper Potts", "William “Thunderbolt” Ross", "Ned", "Thanos", "Loki", "the Collector", "Cull Obsidian", "Ebony Maw", "Proxima Midnight", "Corvus Glaive", "Red Skull", "The Wasp"]
        heroes = [h.lower().replace(" ", "") for h in heroes]

        cv = CountVectorizer(vocabulary=heroes)

        heroes_count_df = pd.DataFrame.sparse.from_spmatrix(cv.fit_transform(X['text']), 
                           X.index,
                           cv.get_feature_names())
        
        heroes_count_arr = heroes_count_df.to_numpy()
        
        poly = PolynomialFeatures(degree=3)
        transformed_df = poly.fit_transform(heroes_count_arr)
        return transformed_df
