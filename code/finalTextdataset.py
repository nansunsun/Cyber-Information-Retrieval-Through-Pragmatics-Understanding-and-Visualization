import pandas as pd
nugget = pd.read_csv("../data/labelledTweets1/predictionresult_nugget1.csv", encoding="latin1")
argument= pd.read_csv("../data/labelledTweets1/predictionresult_argument1.csv", encoding="latin1")

nugget = nugget[['Predict sentence #','word','pred_labels']]
nugget.columns=['ID','word','nugget']

argument = argument[['Predict sentence #','word','pred_labels']]
argument.columns=['ID','word','argument']

nugget['argument']=argument['argument']

text_tweet = nugget
indexNames = text_tweet[(text_tweet['argument'] == 'O')&(text_tweet['nugget'] == 'O')  ].index
text_tweet.drop(indexNames , inplace=True)

text_tweet.to_csv('../data/labelledTweets1/tweets_nugget_argument1.csv', index = False)

## append tweets_nugget_argument 1-4 together and
# reorder index number


tweets_nugget_argument1 = pd.read_csv("../data/labelledTweets1/tweets_nugget_argument1.csv", encoding="latin1")
tweets_nugget_argument2 = pd.read_csv("../data/labelledTweets2/tweets_nugget_argument2.csv", encoding="latin1")
tweets_nugget_argument3 = pd.read_csv("../data/labelledTweets3/tweets_nugget_argument3.csv", encoding="latin1")
tweets_nugget_argument4 = pd.read_csv("../data/labelledTweets4/tweets_nugget_argument4.csv", encoding="latin1")

tweets_nugget_argument1['file_name'] = '1'
tweets_nugget_argument2['file_name'] = '2'
tweets_nugget_argument3['file_name'] = '3'
tweets_nugget_argument4['file_name'] = '4'

tweets_text = pd.concat([tweets_nugget_argument1,tweets_nugget_argument2,tweets_nugget_argument3,tweets_nugget_argument4])
tweets_text['group id'] = tweets_text.groupby(['ID','file_name']).ngroup()
tweets_text = tweets_text[['group id','word','nugget','argument']]
tweets_text.columns=['ID','word','nugget','argument']

text_tweet.to_csv('../data/finaldataset/Tweets_text.csv', index = False)


## article argument and nugget

nugget_article = pd.read_csv("../data/nugget_label_summary_withpos.csv", encoding="latin1")
argument_article= pd.read_csv("../data/argument_label_summary_withpos 12.35.33 pm.csv", encoding="latin1")

nugget_article = nugget_article[['Sentence #','Word','Tag']]
nugget_article.columns=['ID','word','nugget']


argument_article = argument_article[['Sentence #','Word','Tag']]
argument_article.columns=['ID','word','argument']

nugget_article['argument']=argument_article['argument']

text_article = nugget_article
indexNames = text_article[(text_article['argument'] == 'O')&(text_article['nugget'] == 'O')  ].index
text_article.drop(indexNames , inplace=True)

text_article.to_csv('../data/finaldataset/Article_text.csv', index = False)


##append twitter and article
tweet_text = pd.read_csv("../data/finaldataset/Tweets_text.csv", encoding="latin1")
article_text= pd.read_csv("../data/finaldataset/Article_text.csv", encoding="latin1")
tweet_text['file_name'] = 'Tweet'
article_text['file_name'] = 'Article'

text = pd.concat([tweet_text,article_text])
text['group id'] = text.groupby(['ID','file_name']).ngroup()
text = text[['group id','word','nugget','argument']]
text.columns=['ID','word','nugget','argument']



text.to_csv('../data/finaldataset/text.csv', index = False)

