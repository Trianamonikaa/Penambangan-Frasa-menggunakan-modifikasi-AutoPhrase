
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
pd.options.display.max_colwidth = 200
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import pandas as pd
df = pd.read_csv('Data.csv') #untuk membaca data trainingnya
frasa = df.Text
label = df.Label


# In[3]:


corpus = np.array(frasa)
corpus_df = pd.DataFrame({'Frasa': frasa, 
                          'Category': label})
corpus_df = corpus_df[['Frasa', 'Category']]
corpus_df


# In[4]:



wpt = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')

def normalize_document(doc):
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = wpt.tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc

normalize_corpus = np.vectorize(normalize_document)


# In[5]:


norm_corpus = normalize_corpus(corpus)
norm_corpus


# ### meload data corpus yang diambil dari mongodb

# In[6]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import numpy as np
# from tqdm import tqdm_notebook
import os
import cv2
# import imutils
import nltk 
import string
import re
# import pickle
import numpy as np
import pandas as pd
import pymongo
import json
from nltk.corpus import stopwords
from nltk.tag import CRFTagger
# from nltk import everygrams
# from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from six.moves import urllib
# from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import string


# In[7]:


def load_data():
    client = pymongo.MongoClient("mongodb://localhost:27017")
    database = client.ta2
    collection = database.alamatIP

    hasil = collection.find({}, {"_id": 0, "directlink" : 0, "int32":0})
#     hasil = collection.find({})

    hasils = []

    for document in hasil:

        hasils.append(json.dumps(document['paragraf']))
#         hasils.append(json.dumps(document['label']))
    return hasils
data = load_data() # data =  list


# In[8]:


def tokenize(text):
    # obtains tokens with a least 1 alphabet
    tokens = []
    for i in text:
        pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(i.lower())

def remove_punctuation(list_tandabaca):
    sentence = []
    for row in list_tandabaca:
        
        row = ''.join([i for i in row if not i.isdigit()])
        row = re.sub('r ^ https ?\/\/:[\\n\\r]','', row ,flags = re.MULTILINE)
#         row = re.sub(':.[\\n\\]',' ', row ,flags = re.MULTILINE)
        words = [w.replace('[br]', '<br />') for w in row]
        translator = str.maketrans('', '', string.punctuation)
  
        no_punc = row.translate(translator)
        sentence.append(no_punc)
    return sentence

def caseFolding(tokens):
    list_token = []
    for i in tokens:
        tokens = [kata.lower() for kata in i]
        list_token.append(tokens)
    return list_token

def gabungkan(token):
    merged_list = []
    for l in token:
        merged_list += l
#     print(merged_list)
    return merged_list

def mapping(tokens):
    word_to_id = dict()
    id_to_word = dict()

    for i, token in enumerate(set(tokens)):
#         print('i dan token',i, token)
        word_to_id[token] = i
        id_to_word[i] = token
#         print('hasil id to word dan sebaliknya ',id_to_word[i],word_to_id[token])

    return word_to_id, id_to_word

def generate_training_data(tokens, word_to_id, window_size):
    N = len(tokens)
    X, Y = [], []

    for i in range(N):
        nbr_inds = list(range(max(0, i - window_size), i)) +                    list(range(i + 1, min(N, i + window_size + 1)))
        for j in nbr_inds:
            X.append(word_to_id[tokens[i]])
            Y.append(word_to_id[tokens[j]])
            
    X = np.array(X)
    X = np.expand_dims(X, axis=0)
    Y = np.array(Y)
    Y = np.expand_dims(Y, axis=0)
            
    return X, Y


# In[9]:


def remove_n(data):
    datas = re.findall("['\w']+", data[0]) #data[0] = string , datas  = list
    datastr = str(datas) #datastr = string
    strings = ""
    for x in datastr.split(' '):
        if "[" in x:
            x = x.replace("['", "")
        if "," in x:
            x = x.replace(",", "")
        if "]" in x:
            x = x.replace("]", "")
        strings += x
        strings += ' '

    strings = strings.replace("'", " ")
    strings = strings.replace("  ", "")
 
    new_s = ""
    for x in strings.split(" "):
        #print(x)
        if len(x) == 1 and x == "n":
            continue
        elif len(x) >= 2:
            if x[0] == "n" and (ord(x[1]) >= 65 and ord(x[1]) <= 90):
                new_s += x[1:]
                new_s += ' '
            else:
                new_s += x
                new_s += ' '
    #make a list
    list_data = []
    list_data.append(new_s)
    return list_data


# print(remove_n(data))


# In[35]:


removen = remove_n(data)
tokenisasi = tokenize(removen)


# In[34]:


print(tokenize(removen))


# In[25]:


# words_list = [('this', 'is', 'a', 'foo', 'bar', 'sentences'),
#                ('is', 'a', 'foo', 'bar', 'sentences', 'and'),
#                ('a', 'foo', 'bar', 'sentences', 'and', 'i'),
#                ('foo', 'bar', 'sentences', 'and', 'i', 'want'),
#                ('bar', 'sentences', 'and', 'i', 'want', 'to'),
#                ('sentences', 'and', 'i', 'want', 'to', 'ngramize'),
#                ('and', 'i', 'want', 'to', 'ngramize', 'it')]
# new_list = []
# for words in words_list:
#     new_list.append(' '.join(words)) # <---------------

# new_list


# ### pembuatan vocabulary

# In[12]:


from nltk import ngrams
n = 6
sixgrams = ngrams(str(removen).split(), n)
list_baru = []
for grams in sixgrams:
    list_baru.append(' '.join(grams))


# In[13]:


print(list_baru)


# In[37]:


# from openpyxl import load_workbook
# wb  = load_workbook('cika.xlsx')
# sheet1= wb.create_sheet('Sheet1',0)
# sheet1 = wb['Sheet1']
# for x in nini:
#     sheet1.append(x)
    
# wb.save('cika.xlsx')


# In[17]:


# print(nini)


# In[41]:


from keras.preprocessing import text

tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(list_baru)

word2id = tokenizer.word_index
id2word = {v:k for k, v in word2id.items()}

vocab_size = len(word2id) + 1 
embed_size = 100

wids = [[word2id[w] for w in text.text_to_word_sequence(doc)] for doc in list_baru]
print('Vocabulary Size:', vocab_size)
print('Vocabulary Sample:', list(word2id.items())[:10])


# In[15]:


print(wids)


# In[16]:


print(id2word)


# In[17]:


print(len(id2word))


# In[42]:


from keras.preprocessing.sequence import skipgrams

# generate skip-grams
skip_grams = [skipgrams(wid, vocabulary_size=vocab_size, window_size=10) for wid in wids]

# view sample skip-grams
pairs, labels = skip_grams[0][0], skip_grams[0][1]
# print(skip_grams)
for i in range(20):
    print("({:s} ({:d}), {:s} ({:d})) -> {:d}".format(
          id2word[pairs[i][0]], pairs[i][0], 
          id2word[pairs[i][1]], pairs[i][1],
          labels[i]))


# In[21]:


print(skip_grams[0])


# ### build skipgram model architecture

# In[43]:


from keras.layers.merge import concatenate
from keras.layers.merge import Add

from keras.layers.core import Dense, Reshape
from keras.layers.embeddings import Embedding
from keras.models import Sequential

# build skip-gram architecture
word_model = Sequential()
word_model.add(Embedding(vocab_size, embed_size,
                         embeddings_initializer="glorot_uniform",
                         input_length=1))
word_model.add(Reshape((embed_size, )))

context_model = Sequential()
context_model.add(Embedding(vocab_size, embed_size,
                  embeddings_initializer="glorot_uniform",
                  input_length=1))
context_model.add(Reshape((embed_size,)))

model = Sequential()
model.add(Add([word_model, context_model], mode="dot"))
model.add(Dense(1, kernel_initializer="glorot_uniform", activation="sigmoid"))
model.compile(loss="mean_squared_error", optimizer="rmsprop")


# view model summary
print(model.summary())

# visualize model structure
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model, show_shapes=True, show_layer_names=False, 
                 rankdir='TB').create(prog='dot', format='svg'))

