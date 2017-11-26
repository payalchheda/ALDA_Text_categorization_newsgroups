# Reference - https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a

#from __future__ import print_function

#from pprint import pprint

from time import time
import logging

from sklearn.datasets import fetch_20newsgroups
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import nltk
from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_selection import SelectKBest, chi2

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import cross_val_score

# fetch the dataset (both train and test); we will be doing cross validation
# include all 20 categories
# remove unnecessary things - headers, footers, quotes
dataset = fetch_20newsgroups(subset='all', categories=None, remove=('headers', 'footers', 'quotes'))
print("%d documents" % len(dataset.filenames))
print("%d categories" % len(dataset.target_names))

# Bag of words - text files into numerical feature vectors
# use count vectorization
# we segment each text file into words (for English splitting by space), 
# and count # of times each word occurs in each document 
# and finally assign each word an integer id. 
# Each unique word in our dictionary will correspond to a feature (descriptive feature).
# Additionally, we remove stop words & do stemming to reduce number of dimensions
stemmer = SnowballStemmer("english", ignore_stopwords=True)

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

stemmed_count_vect = StemmedCountVectorizer(stop_words='english', max_df=0.5)
X_counts = stemmed_count_vect.fit_transform(dataset.data)
print X_counts.shape

# TF: Just counting the number of words in each document has 1 issue: 
# it will give more weightage to longer documents than shorter documents. To avoid this, 
# we can use frequency (TF - Term Frequencies) i.e. #count(word) / #Total words, in each document.
# TF-IDF: Finally, we can even reduce the weightage of more common words like (the, is, an etc.) 
# which occurs in all document. This is called as TF-IDF i.e Term Frequency times inverse document frequency.
tfidf_transformer = TfidfTransformer(sublinear_tf=True)
X_tfidf = tfidf_transformer.fit_transform(X_counts)
print X_tfidf.shape

# Use Chi square test to reduce dimensionality
ch2 = SelectKBest(chi2, k=10000)
X_ch2 = ch2.fit_transform(X_tfidf, dataset.target)
print X_ch2.shape

print X_ch2.__dict__
#fit naive baiyes
clf_mnb = MultinomialNB(alpha=.01)

print X_ch2.data.shape
print dataset.target.shape
scores = cross_val_score(clf_mnb, X_ch2, dataset.target, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))




