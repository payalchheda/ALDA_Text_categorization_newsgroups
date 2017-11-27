# Reference - https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a

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


from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

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


pipeline = Pipeline([
    ('vect', StemmedCountVectorizer(stop_words='english', max_df=0.5)),
    ('tfidf', TfidfTransformer()),
    ('reduce_dim', ch2),
    ('clf', MultinomialNB()),
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': (0.01, 0.00001, 0.000001),
    'clf__fit_prior': (True, False)
}

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    print(parameters)
    t0 = time()
    grid_search.fit(dataset.data, dataset.target)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

