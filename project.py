import logging
import numpy as np
import sys
from time import time
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics

import nltk
from nltk.stem.snowball import SnowballStemmer

# Remove the quotes, header and footer from the dataset
print("Loading 20 newsgroups dataset for categories:")
# List of 20 categories
categories = [
        'sci.crypt', 
        'sci.electronics', 
        'sci.med', 
        'sci.space',
        'talk.religion.misc',
        'alt.atheism',
        'soc.religion.christian',
        'talk.politics.misc', 
        'talk.politics.guns',
        'talk.politics.mideast',
        'misc.forsale',
        'rec.autos', 
        'rec.motorcycles', 
        'rec.sport.baseball', 
        'rec.sport.hockey',
        'comp.graphics', 
        'comp.os.ms-windows.misc', 
        'comp.sys.ibm.pc.hardware', 
        'comp.sys.mac.hardware', 
        'comp.windows.x'
    ]
remove = ('headers', 'footers', 'quotes')
dataset = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42, remove=remove)
print("Data loaded with size: " + str(len(dataset.data)))
target_names = dataset.target_names

# Let us start with vectorization
stemmer = SnowballStemmer("english", ignore_stopwords=True)

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

stemmed_count_vect = StemmedCountVectorizer(stop_words='english', max_df=0.5)
data_count = stemmed_count_vect.fit_transform(dataset.data)

transformer = TfidfTransformer(sublinear_tf=True)
x = transformer.fit_transform(data_count)

## First way of dimension reduction using Chi Square Test
feature_size = 11000
print("Extracting %d best features by a chi-squared test:" % feature_size)
ch2 = SelectKBest(chi2, k=feature_size)
X = ch2.fit_transform(x, dataset.target)

def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."

# Benchmark classifiers
def benchmark(clf):
    print('_' * 80)
    print(clf)
    #t0 = time()
    #clf.fit(new_x_train, y_train)
    #train_time = time() - t0
    #print("train time: %0.3fs" % train_time)

    #t0 = time()
    #pred = clf.predict(new_x_test)
    #test_time = time() - t0
    #print("test time:  %0.3fs" % test_time)

    #score = metrics.accuracy_score(y_test, pred)
    #print("accuracy:   %0.3f" % score)

    #if hasattr(clf, 'coef_'):
    #    print("dimensionality: %d" % clf.coef_.shape[1])
    #    print("density: %f" % density(clf.coef_))

        
    #    print("top 10 keywords per class:")
    #    for i, label in enumerate(target_names):
    #        top10 = np.argsort(clf.coef_[i])[-10:]
            #print(trim("%s: %s" % (label, " ".join(feature_names[top10]))))
    #    print()

    #print("classification report:")
    #print(metrics.classification_report(y_test, pred, target_names=target_names))
    #print("confusion matrix:")
    #print(metrics.confusion_matrix(y_test, pred))

    #print()
    clf_descr = str(clf).split('(')[0]
    #return clf_descr, score, train_time, test_time
    t0 = time()
    y_pred = cross_val_predict(clf, X, dataset.target, cv=5)
    train_time = time() - t0
    
    print("classification report:")
    print(metrics.classification_report(dataset.target, y_pred, target_names=target_names))

    print("confusion matrix:")
    print(metrics.confusion_matrix(dataset.target, y_pred))

    score = metrics.accuracy_score(dataset.target, y_pred)
    print("accuracy:   %0.3f" % score)
    return clf_descr, score, train_time

results = []
#for clf, name in (
#        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
#        (Perceptron(n_iter=50), "Perceptron"),
#        (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
#        (KNeighborsClassifier(n_neighbors=10), "kNN"),
#        (RandomForestClassifier(n_estimators=100), "Random forest")):
#    print('=' * 80)
#    print(name)
#    results.append(benchmark(clf))
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        (Perceptron(n_iter=50), "Perceptron"),
        (KNeighborsClassifier(n_neighbors=10), "kNN"),
        (RandomForestClassifier(n_estimators=100), "Random forest")):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf))

for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(LinearSVC(penalty=penalty, dual=False,
                                       tol=1e-3)))

    # Train SGD model
    #results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
    #                                       penalty=penalty)))

# Train SGD with Elastic Net penalty
#print('=' * 80)
#print("Elastic-Net penalty")
#results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
#                                       penalty="elasticnet")))

# Train NearestCentroid without threshold
#print('=' * 80)
#print("NearestCentroid (aka Rocchio classifier)")
#results.append(benchmark(NearestCentroid()))

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results.append(benchmark(MultinomialNB(alpha=.01)))
#results.append(benchmark(BernoulliNB(alpha=.01)))

#print('=' * 80)
#print("LinearSVC with L1-based feature selection")
# The smaller C, the stronger the regularization.
# The more regularization, the more sparsity.
results.append(benchmark(Pipeline([
  ('feature_selection', SelectFromModel(LinearSVC(penalty="l1", dual=False,
                                                  tol=1e-3))),
  ('classification', LinearSVC(penalty="l2"))])))

# make some plots

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(3)]

#clf_names, score, training_time, test_time = results
#training_time = np.array(training_time) / np.max(training_time)
#test_time = np.array(test_time) / np.max(test_time)

clf_names, score, training_time = results
training_time = np.array(training_time) / np.max(training_time)

plt.figure(figsize=(12, 8))
plt.title("Score")
#plt.barh(indices, score, .2, label="score", color='navy')
#plt.barh(indices + .3, training_time, .2, label="training time",
#         color='c')
#plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
plt.barh(indices, score, .2, label="score", color='navy')
plt.barh(indices + .3, training_time, .2, label="training time", color='darkorange')
plt.yticks(())
plt.legend(loc='best')
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-.3, i, c)

plt.show()