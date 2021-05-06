import pandas as pd
predictdata = pd.read_csv("../data/labelledTweets4/tweets_tokens4.csv", encoding="latin1")

words = set(list(predictdata['Word'].values))
words.add('PADword')
n_words = len(words)
n_words

class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w,t) for w,t in zip(s["Word"].values.tolist(), s["NER"].values.tolist())]
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
getter = SentenceGetter(predictdata)
sent = getter.get_next()
#print(sent)

#sentences number
sentences = getter.sentences
#print(len(sentences))


largest_sen = max(len(sen) for sen in sentences)
print('biggest sentence has {} words'.format(largest_sen))

max_len = 100
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

import pickle

with open("test_list.txt", "wb") as fp:   #Pickling
    pickle.dump(new_X, fp)

with open("test_list.txt", "rb") as fp:   # Unpickling
    b = pickle.load(fp)

    # # test_pred = model.predict(np.array(b[:558 * 32]), verbose=1, batch_size=32)
    # pred_labels = pred2label(test_pred)
    # #
    # index_number = i
    #
    # prediction_results = pd.DataFrame()
    #
    # for i in range(0, len(pred_labels)):
    #     fourth_sentence = list(zip(b[i], pred_labels[i]))
    #     fourth_sentence_result = pd.DataFrame(fourth_sentence, columns=['word', 'pred_labels'])
    #     fourth_sentence_result['Predict sentence #'] = i + 1
    #     fourth_sentence_result = fourth_sentence_result[fourth_sentence_result.word != 'PADword']
    #     prediction_results = prediction_results.append(fourth_sentence_result)
    # #
    # # prediction_results.to_csv('../data/labelledTweets1/predictionresult1.csv', index = False)
    # #
