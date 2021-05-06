import os
import glob
import pandas as pd

path = "../data/labelledTweets4"
all_files = glob.glob(os.path.join(path, "*.csv"))
names = [os.path.basename(x) for x in glob.glob(path+'\*.csv')]
df = pd.DataFrame()

for file_ in all_files:
    file_df = pd.read_csv(file_, header=0)
    file_df['file_name'] = file_
    df = df.append(file_df)

df = df[['sentences.index','originalText','pos','ner','lemma','file_name']]
df.columns=['Sentence #', 'Word','pos','ner','lemma','file_name']


df['group id'] = df.groupby(['Sentence #','file_name']).ngroup()
#

df = df[['group id','Word','pos','ner','lemma']]
df.columns=['Sentence #', 'Word', 'PoS','NER','Lemma']

#
df.to_csv('../data/labelledTweets4/tweets_tokens4.csv', index = False)


