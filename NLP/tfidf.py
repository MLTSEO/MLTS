
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nlp.normalization import normalize_corpus



def get_tfidf(items, **data):

    ngram_range = data.get('ngram_range',(1, 4))
    min_df = data.get('min_df',3)
    top_n = data.get('top_n', 10)

    items = normalize_corpus(items)
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df = min_df)

    tvec_weights = vectorizer.fit_transform(items)

    weights = np.asarray(tvec_weights.mean(axis=0)).ravel().tolist()

    weights_df = pd.DataFrame({'term': vectorizer.get_feature_names(), 'weight': weights})

    top_features = weights_df.sort_values(by='weight', ascending=False).head(top_n)['term'].tolist()

    return top_features
