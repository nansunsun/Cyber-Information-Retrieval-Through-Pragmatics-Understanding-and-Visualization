import pandas as pd
import re
from pycorenlp.corenlp import StanfordCoreNLP
import json
import nltk
from nltk.corpus import stopwords
import os
from string import digits
import json
import pandas as pd
from pandas.io.json import json_normalize


twitter_cyberattack= pd.read_json('../data/twitterData/cyberattack.json', encoding='utf-8')
twitter_Cybersecurity=pd.read_json('../data/twitterData/cybersecurity.json', encoding='utf-8')
twitter_databreach1=pd.read_json('../data/twitterData/databreach1.json', encoding='utf-8')
twitter_databreach2=pd.read_json('../data/twitterData/databreach2.json', encoding='utf-8')
twitter_phishing= pd.read_json('../data/twitterData/phishing.json', encoding='utf-8')
twitter_ransomware= pd.read_json('../data/twitterData/ransomware.json', encoding='utf-8')
twitter_vulnerability= pd.read_json('../data/twitterData/vulnerability.json', encoding='utf-8')
twitter = pd.concat([twitter_cyberattack, twitter_Cybersecurity, twitter_databreach1,twitter_databreach2,twitter_phishing,twitter_ransomware,twitter_vulnerability], ignore_index=True)
twitter = twitter[['fullname','id','text','timestamp','likes','replies','retweets','text','timestamp','url','user']]

twitter['url'] = 'twitter.com' + twitter['url'].astype(str)
twitter.to_csv('../data/finaldataset/TwitterTraffic.csv', index = False)


# combine article
article = pd.DataFrame(columns=['text', 'title', 'source', 'date'])

directory = '../data/source'

for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        filename_open ='../data/source/' + filename
        myfile = open(filename_open)
        data = myfile.read()

#extract text
        pattern = r"<text>(.*?)</text>"
        text = re.findall(pattern, data, flags=re.DOTALL)
        text = ''.join(text)

        pattern_title = r"<title>(.*?)</title>"
        title = re.findall(pattern_title, data, flags=re.DOTALL)
        title = ''.join(title)

        pattern_source = r"<source>(.*?)</source>"
        source = re.findall(pattern_source, data, flags=re.DOTALL)
        source = ''.join(source)

        pattern_date = r"<date>(.*?)</date>"
        date = re.findall(pattern_date, data, flags=re.DOTALL)
        date = ''.join(date)

        article = article.append({'text': text, 'title':title, 'source':source,'date':date}, ignore_index=True)

article.to_csv('../data/finaldataset/articleTraffic.csv', index = False)





