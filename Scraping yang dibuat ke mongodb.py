
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import numpy as np
# from tqdm import tqdm_notebook
import os
# import cv2
import imutils
import nltk 
import string
import re
# import pickle
import numpy as np
import pandas as pd
import pymongo
import json
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import string


# In[13]:


def load_data():
    client = pymongo.MongoClient("mongodb://localhost:27017")
    database = client.ta2
    collection = database.data


# In[2]:


client = pymongo.MongoClient("mongodb://localhost:27017")
database = client.ta2
collection = database.data


# In[3]:


#menginport library beautifulsoup4
from bs4 import BeautifulSoup
from urllib.request import urlopen
import requests
import re
import pandas as pd
#mengakses halaman web yang ingin diakses

page_link = "https://id.wikipedia.org/wiki/Ekologi_pertanian"
base_link = "https://id.wikipedia.org"
page_response = urlopen(page_link)
soup = BeautifulSoup(page_response, "html.parser")
links = soup.find_all("a",href=True)
import csv
import pandas as pd
urls =[]
isiteks = []

x = dict()

for link in links:
    if re.search('/wiki/',str(link['href'])) and ':' not in link['href'] and link['href'] != '/wiki/Halaman_Utama':
        x[link['href']] = 0
        

for link in links: 
    #expression untuk mengekstrak url dari html link
    if re.search('/wiki/',str(link['href'])) and ':' not in link['href'] and link['href'] != '/wiki/Halaman_Utama' and x[link['href']] == 0:
#         print(link['href'])
        x[link['href']] = 1
        #url lengkap
        url_a = base_link + link['href'] 
#         print(url_a )
        directlink = url_a
#         base_link = 'https://id.wikipedia.org'
        page_response = requests.get(url_a)   
        soup = BeautifulSoup(page_response.content, "html.parser")
        paragraf = ''
        for i in range(0, len(soup.findAll('p'))):
            newparagraf = soup.find_all('p')[i].text
#             judul= soup.find_all("h1",href=True)

            paragraf = paragraf + newparagraf
   
        isiteks.append({'directlink' : url_a, 'paragraf' : paragraf})

# df = pd.DataFrame(isiteks, columns=['directlink','paragraf'])
# df.to_csv('Pertanian.csv', index=True, encoding='utf-8', sep = ',')
# print(isiteks)
buat = collection.insert_many(isiteks)

