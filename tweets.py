
# coding: utf-8

# In[18]:


import pandas as pd
import pandas.io.sql as psql
import psycopg2 as pg
import py_stringmatching.tokenizer as tk
import string
import nltk
import sys
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models, similarities
import numpy as np
from sqlalchemy import create_engine


# In[19]:


#nltk.download('wordnet')


# In[20]:


from nltk.corpus import stopwords
#nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# In[21]:


partition = sys.argv[1]
print(partition)
filename = "/Users/jb1115/Downloads/Tweet_data/" + partition + "Tweets.json"
json_file = open(filename, "r")
tweet_json = json_file.read() 
tweets = pd.read_json(tweet_json, lines=True)
print(filename)


# In[22]:



tweets['textDoc'] = tweets['textDoc'].apply(lambda x:''.join(x))


# In[23]:


table = str.maketrans({key: None for key in string.punctuation})

ws_tok = tk.whitespace_tokenizer.WhitespaceTokenizer()
tweets['textDoc'] = tweets['textDoc'].apply(lambda x: x.translate(table))


# In[24]:


tweets['textDoc'] = tweets['textDoc'].apply(lambda x: x.lower())


# In[25]:


def stemmingLambda(x):
    retVal = ""
    x = ws_tok.tokenize(x)
    for w in x:
        if w not in stop_words:
            retVal += porter_stemmer.stem(w)
            retVal += " "
    return retVal


# In[26]:


from nltk.stem import WordNetLemmatizer

porter_stemmer = PorterStemmer()
tweets['textDoc'] = tweets['textDoc'].apply(stemmingLambda)
print(partition + " cleaned")


# In[27]:



dictionary = corpora.Dictionary.load(partition + '.dict')
print(dictionary)


# In[28]:


model_name = partition + "Model"
model = models.LdaModel.load(model_name)
print(model_name + " loaded")


# In[29]:


result = model.print_topics(205)


# In[30]:


topicFrame = pd.DataFrame(result)
topicFrame['date'] = partition


# In[31]:


i = 0
topics = {}
for index, row in tweets.iterrows():
    doc = row['textDoc']
    hashtag = row['_id']
    docToken = doc.split();
    doc_bow = dictionary.doc2bow(docToken)
    doc_lda = model[doc_bow]
    maxVal = 0
    for val in doc_lda:
        if val[1] > maxVal:
            maxVal = val[1]
            loc = val[0]
    if(maxVal > 0):
            topics[hashtag] = loc
    maxVal = 0


# In[32]:


print("topic sorted")
corpusFrame = pd.DataFrame.from_dict(topics, orient='index')


# In[33]:


corpusFrame['date'] = partition
corpusFrame['hashtag'] = corpusFrame.index
corpusFrame = corpusFrame.rename(index=str, columns={0:"topic", "date":"date", "hashtag": "hashtag"})


# In[34]:


engine = create_engine('postgresql://postgres:tiger@localhost:5432/dse203', echo=False)

corpusFrame.to_sql('hashtag_topic', con=engine, if_exists='append')


# In[35]:


topicFrame = topicFrame.rename(index=str, columns={0:"topic", 1:"terms"})


# In[36]:


engine = create_engine('postgresql://postgres:tiger@localhost:5432/dse203', echo=False)
topicFrame.to_sql('topic_terms', con=engine, if_exists='append')
print(partition + " done")

