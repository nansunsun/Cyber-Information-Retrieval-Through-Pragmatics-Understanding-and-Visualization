from pycorenlp.corenlp import StanfordCoreNLP
import pandas as pd
from pandas.io.json import json_normalize


host = "http://localhost"
port = "9000"
nlp = StanfordCoreNLP(host + ":" + port)


twitter_cyberattack= pd.read_json('../data/twitterData/cyberattack.json', encoding='utf-8')
twitter_Cybersecurity=pd.read_json('../data/twitterData/cybersecurity.json', encoding='utf-8')
twitter_databreach1=pd.read_json('../data/twitterData/databreach1.json', encoding='utf-8')
twitter_databreach2=pd.read_json('../data/twitterData/databreach2.json', encoding='utf-8')
twitter_phishing= pd.read_json('../data/twitterData/phishing.json', encoding='utf-8')
twitter_ransomware= pd.read_json('../data/twitterData/ransomware.json', encoding='utf-8')
twitter_vulnerability= pd.read_json('../data/twitterData/vulnerability.json', encoding='utf-8')
twitter = pd.concat([twitter_cyberattack, twitter_Cybersecurity, twitter_databreach1,twitter_databreach2,twitter_phishing,twitter_ransomware,twitter_vulnerability], ignore_index=True)
twitter = twitter[['id','text','timestamp']]

length  = len(twitter)

for i in range(26781,length):
    text = twitter['text'][i]
    output = nlp.annotate(
        text,
        properties={
            "outputFormat": "json",
            "annotators": "tokenize,ssplit,pos,lemma,ner"
        }
    )

    output = json_normalize(output, record_path=['sentences', 'tokens'], meta=[['sentences', 'index']])
    output = output.drop(['after', 'before'], axis=1)

    # Get the DataFrame column names as a list
    clist = list(output.columns)

    # Rearrange list the way you like
    clist_new = clist[-1:] + clist[:-1]  # brings the last column in the first place

    # Pass the new list to the DataFrame - like a key list in a dict
    output = output[clist_new]

    # Don't forget to add '.csv' at the end of the path
    filenumber = str(i)
    i=i+1
    filename = '../data/labelledTwitter/' + filenumber + '.csv'
    with open(filename,'w') as outfile:
        output.to_csv(filename, index=None,
                                header=True)


