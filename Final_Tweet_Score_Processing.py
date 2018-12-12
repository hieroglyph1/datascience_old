
# coding: utf-8

# In[1]:

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

from nltk.corpus import stopwords
nltk.download('stopwords')


nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer


# # Set up

# In[2]:

def start_up():
    # Connect to database
    conn = pg.connect(
        database="dse203", 
        user="postgres", 
        password="newPass"
    )
    
    return conn
    


# # Preprocessing tweet text

# In[3]:

def stemmingLambda(ws_tok,x):
    porter_stemmer = PorterStemmer()
    retVal = ""
    x = ws_tok.tokenize(x)
    stop_words = set(stopwords.words('english'))
    for w in x:
        if w not in stop_words:
            retVal += porter_stemmer.stem(w)
            retVal += " "
    return retVal


# In[4]:

def process_tweets(orig_tweet_texts):
    table = str.maketrans({key: None for key in string.punctuation})

    ws_tok = tk.whitespace_tokenizer.WhitespaceTokenizer()
    processed_tweet_texts = list(map(lambda x: stemmingLambda(ws_tok,x.translate(table).lower()), orig_tweet_texts)) 
    processed_tweet_texts = list(map(lambda x: set(x.strip().split(' ')),processed_tweet_texts))
    return processed_tweet_texts


# # Calculating set intersection for reuters lexicon

# In[5]:

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


# In[6]:

def get_newsworthiness(processed_tweet_texts):
    # Unpickling
    with open("reuters_processed_lexicon.pickle", "rb") as fp:   
        reuters_lexicon_set = pickle.load(fp)

    reuters_intersections = []

    for tweet_text in processed_tweet_texts:
        matches = len(tweet_text.intersection(reuters_lexicon_set))/len(tweet_text)
        reuters_intersections.append(matches)
        
    return reuters_intersections


# # Calculate set intersection for article lexicion

# In[11]:

def get_uniqueness(processed_tweet_texts,topic_num,time_partition,conn):
    # Obtain article group lexicon
    corpus_query = """
    SELECT corpus FROM topic_corpus WHERE topic=""" + topic_num + """ AND date='""" + time_partition + """'"""
    
    topic_corpus = pd.read_sql(corpus_query, conn)
    topic_corpus = topic_corpus['corpus'].values[0]
    topic_corpus_set = set(topic_corpus.split(' '))
    
    article_topic_intersections = []
    for tweet_text in processed_tweet_texts:
        matches = 1 - ((len(tweet_text.intersection(topic_corpus_set))/len(tweet_text)))
        article_topic_intersections.append(matches)
        
    return article_topic_intersections


# # Uploading scores to database

# In[13]:

def upload_scores(orig_tweet_texts,topic_num,time_partition,newsworthiness,uniqueness):
    newsworthiness_dataframe = pd.DataFrame({'text':orig_tweet_texts})
    newsworthiness_dataframe['topic'] = topic_num
    newsworthiness_dataframe['partition'] = time_partition
    newsworthiness_dataframe['newsworthiness'] = newsworthiness
    newsworthiness_dataframe['uniqueness'] = uniqueness
        
    engine = create_engine('postgresql://postgres:newPass@localhost:5432/dse203', echo=False)

    newsworthiness_dataframe.to_sql('tweet_text_scores', con=engine, if_exists='append')


# # Overall Process

# In[9]:

def run(tweet_list,topic,partition):
    if len(tweet_list) == 0:
        return
    orig_tweet_texts = tweet_list
    topic_num = topic
    time_partition = partition
    conn = start_up()
    processed_tweets = process_tweets(orig_tweet_texts)
    newsworthiness = get_newsworthiness(processed_tweets)
    uniqueness = get_uniqueness(processed_tweets,topic_num,time_partition,conn)
    upload_scores(orig_tweet_texts,topic_num,time_partition,newsworthiness,uniqueness)

