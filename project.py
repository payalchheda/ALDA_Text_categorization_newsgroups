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
feature_size = 18000
print("Extracting %d best features by a chi-squared test:" % feature_size)
ch2 = SelectKBest(chi2, k=feature_size)
X = ch2.fit_transform(x, dataset.target)

def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."

# Benchmark classifiers
def benchmark(clf, name):
    print('_' * 80)
    print(clf)
    clf_descr = str(name).split('(')[0]
    
    t0 = time()
    y_pred = cross_val_predict(clf, X, dataset.target, cv=5)
    train_time = time() - t0
    
    print("classification report:")
    print(metrics.classification_report(dataset.target, y_pred, target_names=target_names))

    print("confusion matrix:")
    print(metrics.confusion_matrix(dataset.target, y_pred))

    accuracy = metrics.accuracy_score(dataset.target, y_pred)
    scores = metrics.precision_recall_fscore_support(dataset.target, y_pred, average='macro')
    precision = scores[0]
    recall = scores[1]
    fscore = scores[2]
    print("\nAccuracy:   %0.3f" % accuracy)
    print("\nPrecision: %0.3f" % precision)
    print("\nRecall %0.3f" % recall)
    print("\nF-measure: %0.3f" % fscore)

    return clf_descr, accuracy, train_time, precision, recall, fscore

results = []
# (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
for clf, name in (
        (Perceptron(n_iter=50), "ANN Classifier"),
        (KNeighborsClassifier(n_neighbors=10), "kNN Classifier"),
        (RandomForestClassifier(n_estimators=100), "Random forest Classifier")):
    print('=' * 80)
    print(name)
    results.append(benchmark(clf, name))

for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    # Train Liblinear model
    results.append(benchmark(LinearSVC(penalty=penalty, dual=False,
                                       tol=1e-3), "Linear SVM - " + penalty))

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results.append(benchmark(MultinomialNB(alpha=.01), "Naive Bayes Classifier"))

# make some plots

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(6)]

clf_names, accuracy, training_time, precision, recall, fscore = results
training_time = np.array(training_time) / np.max(training_time)    

fig, ax = plt.subplots(figsize=(20,15))

ax.set_title("Scores for Models")
ax.set_xlabel("Models")
ax.set_xticks(indices + 0.35)
ax.set_xticklabels(clf_names)
bar1 = ax.bar(indices, accuracy, 0.1, color='b', alpha=0.75, linewidth=0)
bar2 = ax.bar(indices + 0.2, precision, 0.1, color='g', alpha=0.75, linewidth=0)
bar3 = ax.bar(indices + 0.4, recall, 0.1, color='r', alpha=0.75, linewidth=0)
bar4 = ax.bar(indices + 0.6, fscore, 0.1, color='k', alpha=0.75, linewidth=0)

ax.legend((bar1[0], bar2[0], bar3[0], bar4[0]), ('Accuracy', 'Precision', 'Recall', 'F-Measure'), loc='center left', bbox_to_anchor=(1, 0.5))

def autolabel(rects, ax):
    # Get y-axis height to calculate label position from.
    (y_bottom, y_top) = ax.get_ylim()
    y_height = y_top - y_bottom

    for rect in rects:
        height = rect.get_height()

        # Fraction of axis height taken up by this rectangle
        p_height = (height / y_height)

        if p_height > 0.95:
            label_position = height - (y_height * 0.05)
        else:
            label_position = height + (y_height * 0.01)

        
        height = height * 100
        ax.text(rect.get_x() + rect.get_width()/2, label_position,
                '%.1f' % float(height),
                ha='center', va='bottom',color='k',fontsize=12)

autolabel(bar1, ax)
autolabel(bar2, ax)
autolabel(bar3, ax)
autolabel(bar4, ax)

plt.show()