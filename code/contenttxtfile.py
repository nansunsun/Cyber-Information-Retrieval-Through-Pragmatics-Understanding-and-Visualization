import re
from pycorenlp.corenlp import StanfordCoreNLP
import json
import nltk
from nltk.corpus import stopwords
import os
import pandas as pd
import csv



host = "http://localhost"
port = "9000"
nlp = StanfordCoreNLP(host + ":" + port)


with open('../data/source/4.txt', 'r') as myfile:
  data = myfile.read()

#extract text
  pattern = r"<text>(.*?)</text>"
  text = re.findall(pattern, data, flags=re.DOTALL)
  text = ''.join(text)

# remove stop words
#stopwords_en = set(stopwords.words('english'))
#sents = nltk.sent_tokenize(text)

#sents_rm_stopwords = []
#for sent in sents:
#    sents_rm_stopwords.append(' '.join(w for w in nltk.word_tokenize(sent) if w.lower() not in stopwords_en))

#text=''.join(sents_rm_stopwords)
#print(text)

#core nlp annotation
output = nlp.annotate(
    text,
    properties={
       "outputFormat": "json",
       "annotators": "tokenize, ssplit,pos"
    }
)

print(output)




import json
import pandas as pd
from pandas.io.json import json_normalize

output=json_normalize(output,record_path=['sentences','tokens'],meta=[['sentences','index']])
output=output.drop(['after', 'before'], axis=1)

# Get the DataFrame column names as a list
clist = list(output.columns)

# Rearrange list the way you like
clist_new = clist[-1:]+clist[:-1]   # brings the last column in the first place

# Pass the new list to the DataFrame - like a key list in a dict
output = output[clist_new]

export_csv = output.to_csv(r'../data/4.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

    #to_csv("filename.csv")

#open('names.csv', 'w')
#pprint(output)
#with open('example2','w') as outfile:
  #  csv.dump(output,outfile)

