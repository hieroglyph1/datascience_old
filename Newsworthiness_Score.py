
# coding: utf-8

# In[7]:

import py_stringmatching.tokenizer as tk
import string
import nltk
import pickle
from nltk.stem.porter import PorterStemmer
import sys

#-----------------------------------------------------------------------------#
import warnings
warnings.filterwarnings('ignore')
#-----------------------------------------------------------------------------#

import string

import pandas as pd
import pandas.io.sql as psql
import psycopg2 as pg
from datetime import datetime

# First time do: pip install pymongo
from pymongo import MongoClient
from sqlalchemy import create_engine


# # Set up

# In[ ]:

# param = "Feb01_024"
param = sys.argv[1]
print(param)
param = param.split("_")
time_partition = param[0]
topic_num = param[1]


# In[268]:

# Connect to database
conn = pg.connect(
    database="dse203", 
    user="postgres", 
    password="newPass"
)


# In[192]:

# Obtain hashtags mapped to that topic
# Temporarily used for now until mongo collections are sorted by topic and time partition
hashtags_query = """
SELECT hashtag FROM hashtag_topic WHERE topic=""" + topic_num + """ AND date='""" + time_partition + """'"""


topic_hashtags = pd.read_sql(hashtags_query, conn)
topic_hashtags.head()


# In[217]:

topic_hashtags = list(set(topic_hashtags['hashtag'].values))
print(sorted(topic_hashtags))


# In[3]:

client = MongoClient()

# Database name on Mongo: DSE203-Project, collection name:Tweets
db = client['DSE203-Project']

# TODO: Switch to later, To access proper collection
#collection = db[param]


collection = db['Tweets']



# # Load Data

# In[221]:

#test_tweets_query = collection.find({'entities.hashtags.text':'RightToTry'})

# Finds all tweets with hashtags associated with passed in topic id
# Temporary until collections are sorted by time partition and topic
test_tweets_query = collection.find({'entities.hashtags.text':{ '$in': topic_hashtags }})

# Use later
# test_tweets_query = collection.find({})


# In[222]:

test_tweets = []
for tweet in test_tweets_query:
    test_tweets.append(tweet)


# In[317]:

orig_tweet_texts = []
for tweet in test_tweets:
    orig_tweet_texts.append(tweet['text'])


# # Preprocessing tweet text

# In[11]:

nltk.download('wordnet')


# In[12]:

from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# In[319]:

orig_tweet_texts


# In[95]:

def stemmingLambda(x):
    retVal = ""
    x = ws_tok.tokenize(x)
    for w in x:
        if w not in stop_words:
            retVal += porter_stemmer.stem(w)
            retVal += " "
    return retVal


# In[321]:

from nltk.stem import WordNetLemmatizer

porter_stemmer = PorterStemmer()
table = str.maketrans({key: None for key in string.punctuation})

ws_tok = tk.whitespace_tokenizer.WhitespaceTokenizer()
processed_tweet_texts = list(map(lambda x: stemmingLambda(x.translate(table).lower()), orig_tweet_texts)) 


# In[322]:

processed_tweet_texts


# # Calculating set intersection for reuters lexicon

# In[227]:

# Used first time through
"""
from nltk.corpus import reuters
reuters_lexicon_set = set()
for file in reuters.fileids():
    curr_doc = reuters.raw(file)
    curr_doc = curr_doc.translate(table).lower()
    curr_doc = stemmingLambda(curr_doc)
    tokens = curr_doc.strip().split(' ')
    for token in tokens:
        reuters_lexicon_set.add(token)
        
#Pickling processed lexicon
with open("reuters_processed_lexicon.pickle", "wb") as file:   
    pickle.dump(reuters_lexicon_set, file)
"""
# Unpickling
with open("reuters_processed_lexicon.pickle", "rb") as fp:   
    reuters_lexicon_set = pickle.load(fp)


# In[307]:

reuters_lexicon_set


# In[228]:

len(reuters_lexicon_set)


# In[229]:

processed_tweet_texts = list(map(lambda x: set(x.strip().split(' ')),processed_tweet_texts))


# In[230]:

processed_tweet_texts


# In[296]:

reuters_intersections = []
max_intersections = 0
for tweet_text in processed_tweet_texts:
    matches = len(tweet_text.intersection(reuters_lexicon_set))
    if matches > max_intersections:
        max_intersections = matches
    reuters_intersections.append(matches)
    
reuters_intersections = [x / max_intersections for x in reuters_intersections]


# In[297]:

for ele in reuters_intersections:
    print(ele)


# # Calculate set intersection for article lexicion

# In[257]:

# Obtain article group lexicon
corpus_query = """
SELECT corpus FROM topic_corpus WHERE topic=""" + topic_num + """ AND date='""" + time_partition + """'"""


topic_corpus = pd.read_sql(corpus_query, conn)
topic_corpus = topic_corpus['corpus'].values[0]
topic_corpus_set = set(topic_corpus.split(' '))


# In[308]:

len(topic_corpus_set)


# In[309]:

topic_corpus_set


# In[298]:

article_topic_intersections = []
max_intersections = 0
for tweet_text in processed_tweet_texts:
    matches = len(tweet_text.intersection(topic_corpus_set))
    if matches > max_intersections:
        max_intersections = matches
    article_topic_intersections.append(matches)
    
article_topic_intersections = [x / max_intersections for x in article_topic_intersections]


# In[299]:

for ele in article_topic_intersections:
    print(ele)


# # Calculate Newsworthiness Scores

# In[300]:

newsworthiness_scores = []
for i in range(len(processed_tweet_texts)):
    newsworthiness_scores.append(1/(abs(reuters_intersections[i]-article_topic_intersections[i])+1))
print(newsworthiness_scores)


# # Uploading scores to database

# In[325]:

newsworthiness_dataframe = pd.DataFrame({'text':orig_tweet_texts})
newsworthiness_dataframe['topic'] = topic_num
newsworthiness_dataframe['partition'] = time_partition
newsworthiness_dataframe['score'] = newsworthiness_scores


# In[326]:

newsworthiness_dataframe


# In[328]:

engine = create_engine('postgresql://postgres:newPass@localhost:5432/dse203', echo=False)

newsworthiness_dataframe.to_sql('newsworthiness', con=engine, if_exists='append')


# In[ ]:



