
# coding: utf-8

# In[21]:


import pandas as pd
import sys
import pandas.io.sql as psql
import psycopg2 as pg
import py_stringmatching.tokenizer as tk
import string
import nltk
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models, similarities
import numpy as np
from sqlalchemy import create_engine


# In[22]:


nltk.download('wordnet')


# In[23]:


from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


# In[ ]:


partition = sys.argv[1]
other_format = sys.argv[2]
print(partition)


# In[ ]:



query = 'select news.article, news.date1 from fuzzymatcher.news as news where date1 = \'' + other_format+ '\''
    
print(query)


# In[24]:


conn = pg.connect(database="dse203",user="postgres", password="newPass")
news = pd.read_sql(query, conn)


# In[25]:


table = str.maketrans({key: None for key in string.punctuation})

ws_tok = tk.whitespace_tokenizer.WhitespaceTokenizer()
news['article'] = news['article'].apply(lambda x: x.translate(table))


# In[26]:


news['article'] = news['article'].apply(lambda x: x.lower())


# In[27]:


def stemmingLambda(x):
    retVal = ""
    x = ws_tok.tokenize(x)
    for w in x:
        if w not in stop_words:
            retVal += porter_stemmer.stem(w)
            retVal += " "
    return retVal


# In[28]:


from nltk.stem import WordNetLemmatizer

porter_stemmer = PorterStemmer()
news['article'] = news['article'].apply(stemmingLambda)
print(partition + " cleaned")


# In[29]:


#from sqlalchemy import create_engine
#engine = create_engine('postgresql://postgres:tiger@localhost:5432/dse203', echo=False)

#news.to_sql('fuzzymatcher.clean_news', con=engine, if_exists='append')


# In[30]:


array_of_contents = [[token for token in contents.split()] for contents in news['article']]
dictionary = corpora.Dictionary(array_of_contents)
print(dictionary)
dictionary.save(partition + '.dict')


# In[31]:


corpus = [dictionary.doc2bow(text.split()) for text in news['article']]


# In[33]:


np.shape(corpus)


# In[35]:


model = models.LdaModel(corpus, id2word=dictionary, num_topics=205)


# In[37]:


model_name = partition + "Model"
model.save(model_name)
print(partition + " modeled")


# In[41]:


i = 0
topics = {}
dates = {}
for index, row in news.iterrows():
    doc = row['article']
    docToken = doc.split();
    doc_bow = dictionary.doc2bow(docToken)
    doc_lda = model[doc_bow]
    maxVal = 0
    for val in doc_lda:
        if val[1] > maxVal:
            maxVal = val[1]
            loc = val[0]
    if(maxVal > 0):
        date = row['date1']
        if (loc, date) in dates:
            dates[(loc, date)] += 1
        else:
            dates[(loc, date)] = 1
        if loc in topics:
            topics[loc] +=doc
        else:
            topics[loc] = doc
    maxVal = 0


# In[50]:


corpusFrame = pd.DataFrame.from_dict(topics, orient='index')
dateFrame = pd.DataFrame.from_dict(dates, orient='index')
dateFrame["topic"] = dateFrame.index.values
dateFrame["date"] = dateFrame["topic"].apply(lambda x: x[1])
dateFrame["topic"] = dateFrame["topic"].apply(lambda x: x[0])


# In[80]:


corpusFrame['date'] = partition
corpusFrame['topic'] = corpusFrame.index
corpusFrame = corpusFrame.rename(index=str, columns={0:"corpus", "date":"date", "topic": "topic"})
corpusFrame.head()


# In[87]:


engine = create_engine('postgresql://postgres:tiger@localhost:5432/dse203', echo=False)

corpusFrame.to_sql('topic_corpus', con=engine, if_exists='append')


# In[51]:


dateFrame = dateFrame.rename(index=str, columns={0:"count", "date":"date", "topic": "topic"})

engine = create_engine('postgresql://postgres:tiger@localhost:5432/dse203', echo=False)
dateFrame.to_sql('topic_date_count', con=engine, if_exists='append')

