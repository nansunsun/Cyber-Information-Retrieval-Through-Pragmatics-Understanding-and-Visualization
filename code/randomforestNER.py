#Data analysis
import pandas as pd
import numpy as np

#Data visualisation
import matplotlib.pyplot as plt
import seaborn as sns

import nltk


#Modeling
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn_crfsuite import CRF, scorers, metrics
import sklearn_crfsuite
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.metrics import classification_report, make_scorer
import scipy.stats
import eli5



data = pd.read_csv(r"../data/nugget_label_summary_withpos.csv", encoding="latin1").fillna(method="ffill")
data.head()

data['Tag'].replace('0', 'O', inplace=True)
#data = data.drop(['POS'], axis =1)
#data = data.fillna(method="ffill")
data = data.drop(data.index[0:14])
data.tail(12)
words = set(list(data['Word'].values))
words.add('PADword')
n_words = len(words)
n_words
print(n_words)
tags = list(set(data["Tag"].values))
print(tags)
n_tags = len(tags)
n_tags
print(n_tags)

ners = list(set(data["NER"].values))
n_ners=len(ners)
print(n_ners)


pos = list(set(data["PoS"].values))
n_pos=len(pos)
print(n_pos)

lemmas = list(set(data["Lemma"].values))
n_lemmas=len(lemmas)
print(n_lemmas)

class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t, p, n, l) for w, t, p, n, l in zip(s["Word"].values.tolist(), s["Tag"].values.tolist(), s["PoS"].values.tolist(),s["NER"].values.tolist(),s["Lemma"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

#sentence example
getter = SentenceGetter(data)
sent = getter.get_next()


#sentences number
sentences = getter.sentences
print(len(sentences))

plt.style.use("ggplot")
plt.hist([len(s) for s in sentences], bins=50)
plt.show()

#Lets find out the longest sentence length in the dataset
maxlen = max([len(s) for s in sentences])
print ('Maximum sentence length:', maxlen)

#Words tagged as B-org
data.loc[data['Tag'] == 'B-Databreach', 'Word'].head()
data.loc[data['Tag'] == 'I-Databreach', 'Word'].head()

#Words distribution across Tags without O tag
plt.figure(figsize=(15, 5))
ax = sns.countplot('Tag', data=data.loc[data['Tag'] != 'O'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")
plt.tight_layout()
plt.show()

#Words distribution across POS
plt.figure(figsize=(15, 5))
ax = sns.countplot('PoS', data=data, orient='h')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")
plt.tight_layout()
plt.show()


#Random Forest classifier

#Simple feature map to feed arrays into the classifier.
def feature_map(word):
    return np.array([word.istitle(), word.islower(), word.isupper(), len(word),
                     word.isdigit(),  word.isalpha()])
words = [feature_map(w) for w in data["Word"].values.tolist()]
tags = data["Tag"].values.tolist()

#Random Forest classifier
# pred = cross_val_predict(RandomForestClassifier(n_estimators=20),X=words, y=tags, cv=5)
#
# from sklearn.metrics import classification_report
# report = classification_report(y_pred=pred, y_true=tags)
# print(report)

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label, postag, ner, lemma in sent]
#Creating the train and test set
from sklearn.model_selection import train_test_split
X = [sent2features(s) for s in sentences]
y = [sent2labels(s) for s in sentences]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

#Creating the CRF model
crf = CRF(algorithm='lbfgs',
          c1=0.1,
          c2=0.1,
          max_iterations=100,
          all_possible_transitions=False)
crf.fit(X_train, y_train)
#We predcit using the same 5 fold cross validation
# pred = cross_val_predict(estimator=crf, X=X, y=y, cv=5)
#
# report = flat_classification_report(y_pred=pred, y_true=y)
# print(report)
new_classes = classes.copy()
new_classes.pop()
new_classes
y_pred = crf.predict(X_test)
print(metrics.flat_classification_report(y_test, y_pred, labels = new_classes))
#
# #Tuning the parameters manually, setting c1 = 10
# crf2 = CRF(algorithm='lbfgs',
#           c1=10,
#           c2=0.1,
#           max_iterations=100,
#           all_possible_transitions=False)
#
# pred = cross_val_predict(estimator=crf2, X=X, y=y, cv=5)
# report = flat_classification_report(y_pred=pred, y_true=y)
# print(report)
#
# crf2.fit(X, y)