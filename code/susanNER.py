import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
df = pd.read_csv(r"../data/summary.csv", encoding="ISO-8859-1")
df.head()
df.isnull().sum()
df = df.fillna(method='ffill')
print(df['Sentence #'].nunique(), df.Word.nunique(), df.Tag.nunique())
df.groupby('Tag').size().reset_index(name='counts')

X = df.drop('Tag', axis=1)
v = DictVectorizer(sparse=False)
X = v.fit_transform(X.to_dict('records'))
y = df.Tag.values
classes = np.unique(y)
classes = classes.tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
# print(X_train.shape, y_train.shape)
#
# #out of core algorithm
# per = Perceptron(verbose=10, n_jobs=-1, max_iter=5)
# per.partial_fit(X_train, y_train, classes)
# # remove o
# new_classes = classes.copy()
# new_classes.pop()
# new_classes
#
# print(classification_report(y_pred=per.predict(X_test), y_true=y_test, labels=new_classes))