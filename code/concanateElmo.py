import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K
from keras.models import Model, Input
from keras.layers.merge import add
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Lambda
from keras.layers import concatenate
from keras.models import load_model


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

plt.style.use("ggplot")
data = pd.read_csv("../data/argument_label_summary_withpos 12.35.33 pm.csv", encoding="latin1")
data['Tag'].replace('0', 'O', inplace=True)
#data = data.drop(['POS'], axis =1)
#data = data.fillna(method="ffill")
data = data.drop(data.index[0:14])

data.tail(12)
words = set(list(data['Word'].values))
words.add('PADword')
n_words = len(words)
n_words
#print(n_words)
tags = list(set(data["Tag"].values))
n_tags = len(tags)
n_tags
#print(n_tags)
#print(data)

ners = list(set(data["NER"].values))
n_ners=len(ners)
print(n_ners)

# in charge of converting every sentence with its named entities (tags) into a list of tuples [(word, named entity), …]
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
#print(sent)

#sentences number
sentences = getter.sentences
#print(len(sentences))


largest_sen = max(len(sen) for sen in sentences)
print('biggest sentence has {} words'.format(largest_sen))


plt.hist([len(sen) for sen in sentences], bins= 50)
plt.show()

#In order to feed our sentences into a LSTM network, they all need to be the same size
#add padding words
max_len = largest_sen
X = [[w[0]for w in s] for s in sentences]
new_X = []
for seq in X:
    new_seq = []
    for i in range(max_len):
        try:
            new_seq.append(seq[i])
        except:
            new_seq.append("PADword")
    new_X.append(new_seq)

N = [[w[4]for w in s] for s in sentences]
new_N = []
for ner in N:
    new_ner = []
    for i in range(max_len):
        try:
            new_ner.append(ner[i])
        except:
            new_ner.append("O")
    new_N.append(new_ner)

from keras.preprocessing.sequence import pad_sequences
tags2index = {t:i for i,t in enumerate(tags)}
y = [[tags2index[w[1]] for w in s] for s in sentences]
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tags2index["O"])
#truncating= “post”

y[15]

#split data into training and testing
#load elmo embedding
X_tr, X_te, y_tr, y_te = train_test_split(new_X, y, test_size=0.1, random_state=2018)
Ner_tr, Ner_te = train_test_split(new_N, test_size=0.1, random_state=2018)

sess = tf.compat.v1.Session()
K.set_session(sess)
elmo_model = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
sess.run(tf.compat.v1.global_variables_initializer())
sess.run(tf.compat.v1.tables_initializer())

#convert sentences into elmo embedding
batch_size = 32
def ElmoEmbedding(x):
    return elmo_model(inputs={"tokens": tf.squeeze(tf.cast(x,    tf.string)),"sequence_len": tf.constant(batch_size*[max_len])
                     },
                      signature="tokens",
                      as_dict=True)["elmo"]


#build neural network
input_text = Input(shape=(max_len,), dtype=tf.string)
ner_text = Input(shape=(max_len,), dtype=tf.string)



embedding_text = Lambda(ElmoEmbedding, output_shape=(max_len, 1024))(input_text)
embedding_ner = Lambda(ElmoEmbedding, output_shape=(max_len, 1024))(ner_text)
#embedding_ner = Embedding(output_dim=512, input_dim=10000, input_length=100)(ner_text)
embedding = concatenate([embedding_text, embedding_ner])

x = Bidirectional(LSTM(units=512, return_sequences=True,
                       recurrent_dropout=0.2, dropout=0.2))(embedding)
x_rnn = Bidirectional(LSTM(units=512, return_sequences=True,
                           recurrent_dropout=0.2, dropout=0.2))(x)
x = add([x, x_rnn])  # residual connection to the first biLSTM
out = TimeDistributed(Dense(n_tags, activation="softmax"))(x)

print(input_text)
model = Model([input_text, ner_text], out)
model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["acc"])

print(model.summary())
X_tr, X_val = X_tr[:455*batch_size], X_tr[-45*batch_size:]
#455 45
y_tr, y_val = y_tr[:455*batch_size], y_tr[-45*batch_size:]
y_tr = y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)
y_val = y_val.reshape(y_val.shape[0], y_val.shape[1], 1)

Ner_tr, Ner_val = Ner_tr[:455*batch_size], Ner_tr[-45*batch_size:]



history = model.fit([np.array(X_tr), np.array(Ner_tr)], y_tr, validation_data=([np.array(X_val),np.array(Ner_val)], y_val), batch_size= 32, epochs= 7, verbose=1)


from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

X_te = X_te[:50 * batch_size]
Ner_te = Ner_te[:50 * batch_size]
#50
# test_pred = model.predict([np.array(X_te),np.array(Ner_te)], verbose=1,batch_size=32)
test_pred = model.predict([np.array(X_te),np.array(Ner_te)], verbose=1,batch_size=32)


idx2tag = {i: w for w, i in tags2index.items()}


def pred2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i].replace("PADword", "O"))
        out.append(out_i)
    return out


def test2label(pred):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            out_i.append(idx2tag[p].replace("PADword", "O"))
        out.append(out_i)
    return out


pred_labels = pred2label(test_pred)
test_labels = test2label(y_te[:50 * batch_size])

#change 32 to 16
print(classification_report(test_labels, pred_labels))


index_number = i
prediction_results = pd.DataFrame()
for i in range(0,len(pred_labels)):
    fourth_sentence =  list(zip(X_te[i],pred_labels[i], test_labels[i]))
    fourth_sentence_result = pd.DataFrame(fourth_sentence, columns = ['word','pred_labels' ,'test_labels'])
    fourth_sentence_result['Predict sentence #']= i+1
    fourth_sentence_result = fourth_sentence_result[fourth_sentence_result.word != 'PADword']
    prediction_results = prediction_results.append(fourth_sentence_result)

import pickle
with open("test_list.txt", "rb") as fp:   # Unpickling
  b = pickle.load(fp)
test_pred = model.predict(np.array(b[:50*32]), verbose=1,batch_size=32)

pred_labels = pred2label(test_pred)
index_number = i
prediction_results = pd.DataFrame()
for i in range(0,len(pred_labels)):
     fourth_sentence =  list(zip(b[i],pred_labels[i]))
     fourth_sentence_result = pd.DataFrame(fourth_sentence, columns = ['word','pred_labels'])
     fourth_sentence_result['Predict sentence #']= i+1
     fourth_sentence_result = fourth_sentence_result[fourth_sentence_result.word != 'PADword']
     prediction_results = prediction_results.append(fourth_sentence_result)