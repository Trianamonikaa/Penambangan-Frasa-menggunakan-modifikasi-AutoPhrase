
# coding: utf-8

# ### import library

# In[1]:


import pandas as pd
import numpy as np
from collections import Counter
import re

# languange processing imports
import nltk
# preprocessing imports
from sklearn.preprocessing import LabelEncoder

# model imports
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.word2vec import Word2Vec
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
# hyperparameter training imports
from sklearn.model_selection import GridSearchCV
from gensim.corpora import Dictionary
# visualization imports
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import base64
import io
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()  # defines the st


# ### upload data buat jadi data training

# In[5]:


import numpy as np
import csv

df = pd.read_csv('kumpulan_label.csv',encoding = 'unicode_escape') #jadi nama filenya adalah kumpulan_label.csv
frasa_df = df[df.label=='frasa']
nonfrasa_df = df[df.label=='bukan frasa'][:2700]
   
# print(type(nonfrasa_df))
train_data = frasa_df.append(nonfrasa_df)


# In[6]:


#kalau mau lihat grafik dari pembagian datanya
fig, ax = plt.subplots(1,1,figsize=(8,6))

author_vc = train_data.label.value_counts()

ax.bar(range(2), author_vc)
ax.set_xticks(range(2))
ax.set_xticklabels(author_vc.index, fontsize=16)

for rect, c, value in zip(ax.patches, ['b', 'r', 'g'], author_vc.values):
    rect.set_color(c)
    height = rect.get_height()
    width = rect.get_width()
    x_loc = rect.get_x()
    ax.text(x_loc + width/2, 0.9*height, value, ha='center', va='center', fontsize=18, color='white')


# In[7]:


document_lengths = np.array(list(map(len, train_data.kata.str.split(' '))))

print("The average number of words in a document is: {}.".format(np.mean(document_lengths)))
print("The minimum number of words in a document is: {}.".format(min(document_lengths)))
print("The maximum number of words in a document is: {}.".format(max(document_lengths)))


# In[8]:


#tampilan dokumen dengan panjang 6
train_data[document_lengths ==  6]


# In[9]:


our_special_word = 'qwerty'

def remove_ascii_words(df): # untuk menghilangkan kata-kata yang masih mengandung ascii
    """ removes non-ascii characters from the 'texts' column in df.
    It returns the words containig non-ascii characers.
    """
    non_ascii_words = []
    for i in range(len(df)):
        for word in df.loc[i, 'kata'].split(' '):
            if any([ord(character) >= 128 for character in word]):
                non_ascii_words.append(word)
                df.loc[i, 'kata'] = df.loc[i, 'kata'].replace(word, our_special_word)
    return non_ascii_words

non_ascii_words = train_data.kata

# print("Replaced {} words with characters with an ordinal >= 128 in the train data.".format(
#     len(non_ascii_words)))


# In[10]:


def get_good_tokens(sentence): #menghilangkan punctuation
    replaced_punctation = list(map(lambda token: re.sub('[^0-9A-Za-z!?]+', '', token), sentence))
    removed_punctation = list(filter(lambda token: token, replaced_punctation))
    return removed_punctation


# In[8]:


import nltk
nltk.download('punkt')


# #### melakukan preprocessing pada data teks supaya bentuknya bisa diolah

# In[12]:


def w2v_preprocessing(df):
    """ All the preprocessing steps for word2vec are done in this function.
    All mutations are done on the dataframe itself. So this function returns
    nothing.
    """
    df['text'] = df.kata.str.lower()
    df['document_sentences'] = df.text.str.split('.')  # split texts into individual sentences
    df['tokenized_sentences'] = list(map(lambda sentences:
                                         list(map(nltk.word_tokenize, sentences)),
                                         df.document_sentences))  # tokenize sentences
    
    print(df.text)
    print(df.document_sentences)
    print(df.tokenized_sentences)

w2v_preprocessing(train_data)


# In[13]:


def lda_get_good_tokens(df):
    df['text'] = df.text.str.lower()
    df['tokenized_text'] = list(map(nltk.word_tokenize, df.text))
    df['tokenized_text'] = list(map(get_good_tokens, df.tokenized_text))
    print(df['tokenized_text'])
lda_get_good_tokens(train_data)


# In[14]:


# kita gabungkan dia datanya dengan cara di concatenated
tokenized_only_dict = Counter(np.concatenate(train_data.tokenized_text.values)) 

tokenized_only_df = pd.DataFrame.from_dict(tokenized_only_dict, orient='index')
tokenized_only_df.rename(columns={0: 'count'}, inplace=True)


# In[15]:


def remove_stopwords(df):
    """ Removes stopwords based on a known set of stopwords
    available in the nltk package. In addition, we include our
    made up word in here.
    """
    # Luckily nltk already has a set of stopwords that we can remove from the texts.
    stopwords = nltk.corpus.stopwords.words('english')
    # we'll add our own special word in here 'qwerty'
    stopwords.append(our_special_word)

    df['stopwords_removed'] = list(map(lambda doc:
                                       [word for word in doc if word not in stopwords],
                                       df['tokenized_text']))

remove_stopwords(train_data)


# In[16]:


def stem_words(df):
    lemm = nltk.stem.WordNetLemmatizer()
    df['lemmatized_text'] = list(map(lambda sentence:
                                     list(map(lemm.lemmatize, sentence)),
                                     df.stopwords_removed))

    p_stemmer = nltk.stem.porter.PorterStemmer()
    df['stemmed_text'] = list(map(lambda sentence:
                                  list(map(p_stemmer.stem, sentence)),
                                  df.lemmatized_text))

stem_words(train_data)


# #### build vocabulary buat data training

# In[17]:


dictionary = Dictionary(documents=train_data.stemmed_text.values)

print("Found {} words.".format(len(dictionary.values())))


# In[18]:


dictionary.filter_extremes(no_above=0.8, no_below=3)

dictionary.compactify()  # Reindexes the remaining words after filtering
print("Left with {} words.".format(len(dictionary.values())))


# In[19]:


def document_to_bow(df):
    df['bow'] = list(map(lambda doc: dictionary.doc2bow(doc), df.stemmed_text))
    print(df['bow'])
document_to_bow(train_data)


# In[20]:


def lda_preprocessing(df):
    """ All the preprocessing steps for LDA are combined in this function.
    All mutations are done on the dataframe itself. So this function returns
    nothing.
    """
    lda_get_good_tokens(df)
    remove_stopwords(df)
    stem_words(df)
    document_to_bow(df)


# In[21]:


cleansed_words_df = pd.DataFrame.from_dict(dictionary.token2id, orient='index')
cleansed_words_df.rename(columns={0: 'id'}, inplace=True)

cleansed_words_df['count'] = list(map(lambda id_: dictionary.dfs.get(id_), cleansed_words_df.id))
del cleansed_words_df['id']


# In[22]:


cleansed_words_df.sort_values('count', ascending=False, inplace=True)


# In[23]:



frasa_words = list(np.concatenate(train_data.loc[train_data.label == 'frasa', 'stemmed_text'].values))
bukanfrasa_words = list(np.concatenate(train_data.loc[train_data.label == 'bukan frasa', 'stemmed_text'].values))


# In[24]:


frasa_word_frequencies = {word: frasa_words.count(word) for word in cleansed_words_df.index[:50]}
bukanfrasa_word_frequencies = {word: bukanfrasa_words.count(word) for word in cleansed_words_df.index[:50]}


# In[25]:


frequencies_df = pd.DataFrame(index=cleansed_words_df.index[:50])


# In[26]:


frequencies_df['frasa_freq'] = list(map(lambda word:
                                      frasa_word_frequencies[word],
                                      frequencies_df.index))
frequencies_df['frasa_bukanfrasa_freq'] = list(map(lambda word:
                                          frasa_word_frequencies[word] + bukanfrasa_word_frequencies[word],
                                          frequencies_df.index))


# In[27]:


fig, ax = plt.subplots(1,1,figsize=(20,5))

nr_top_words = len(frequencies_df)
nrs = list(range(nr_top_words))
sns.barplot(nrs, frequencies_df['frasa_bukanfrasa_freq'].values, color='g', ax=ax, label="bukan frasa")
sns.barplot(nrs, frequencies_df['frasa_freq'].values, color='r', ax=ax, label="frasa")

ax.set_title("frekuensi kemunculan kata sesuai dengan labelnya", fontsize=16)
ax.legend(prop={'size': 16})
ax.set_xticks(nrs)
ax.set_xticklabels(frequencies_df.index, fontsize=14, rotation=90);


# In[28]:


sentences = []
for sentence_group in train_data.tokenized_sentences:
    sentences.extend(sentence_group)

print("Number of sentences: {}.".format(len(sentences)))
print("Number of texts: {}.".format(len(train_data)))


# In[29]:


get_ipython().run_cell_magic('time', '', '# Set values for various parameters\nnum_features = 200    # Word vector dimensionality\nmin_word_count = 3    # Minimum word count\nnum_workers = 4       # Number of threads to run in parallel\ncontext = 6           # Context window size\ndownsampling = 1e-3   # Downsample setting for frequent words\n\n# Initialize and train the model\nW2Vmodel = Word2Vec(sentences=sentences,\n                    sg=1,\n                    hs=0,\n                    workers=num_workers,\n                    size=num_features,\n                    min_count=min_word_count,\n                    window=context,\n                    sample=downsampling,\n                    negative=5,\n                    iter=6)')


# In[30]:


def get_w2v_features(w2v_model, sentence_group):
    """ Transform a sentence_group (containing multiple lists
    of words) into a feature vector. It averages out all the
    word vectors of the sentence_group.
    """
    words = np.concatenate(sentence_group)  # words in text
    index2word_set = set(w2v_model.wv.vocab.keys())  # words known to model
    
    featureVec = np.zeros(w2v_model.vector_size, dtype="float32")
    
    # Initialize a counter for number of words in a review
    nwords = 0
    # Loop over each word in the comment and, if it is in the model's vocabulary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            featureVec = np.add(featureVec, w2v_model[word])
            nwords += 1.

    # Divide the result by the number of words to get the average
    if nwords > 0:
        featureVec = np.divide(featureVec, nwords)
    return featureVec

train_data['w2v_features'] = list(map(lambda sen_group:
                                      get_w2v_features(W2Vmodel, sen_group),
                                      train_data.tokenized_sentences))


# In[31]:


frasa_w2v_distribution = train_data.loc[train_data.label == 'frasa', 'w2v_features'].mean()
bukanfrasa_w2v_distribution = train_data.loc[train_data.label == 'bukan frasa', 'w2v_features'].mean()


# In[32]:


label_encoder = LabelEncoder()

label_encoder.fit(train_data.label)
train_data['label'] = label_encoder.transform(train_data.label)


# In[33]:


def get_cross_validated_model(model, param_grid, X, y, nr_folds=5):
    """ Trains a model by doing a grid search combined with cross validation.
    args:
        model: your model
        param_grid: dict of parameter values for the grid search
    returns:
        Model trained on entire dataset with hyperparameters chosen from best results in the grid search.
    """
    # train the model (since the evaluation is based on the logloss, we'll use neg_log_loss here)
    grid_cv = GridSearchCV(model, param_grid=param_grid, scoring='neg_log_loss', cv=nr_folds, n_jobs=-1, verbose=True)
    best_model = grid_cv.fit(X, y)
    # show top models with parameter values
    result_df = pd.DataFrame(best_model.cv_results_)
    show_columns = ['mean_test_score', 'mean_train_score', 'rank_test_score']
    for col in result_df.columns:
        if col.startswith('param_'):
            show_columns.append(col)
    display(result_df[show_columns].sort_values(by='rank_test_score').head())
    return best_model


# In[34]:


X_train_w2v = np.array(list(map(np.array, train_data.w2v_features)))


# In[35]:


models = dict()


# #### penggunaan gridsearch dengan parameter yang paling baik

# In[36]:


from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
parameters = {'kernel':['linear'], 'C':[1000]}
svc = svm.SVC(gamma=0.001)
clf = GridSearchCV(SVC(), parameters, cv=5)

clf.fit(X_train_w2v, train_data.label)


# #### mengupload data_test dari mongodb untuk disimpan menjadi bentuk dict

# In[40]:


import pymongo
import json

def load_data():
    client = pymongo.MongoClient("mongodb://localhost:27017")
    database = client.ta2
    collection = database.datasetautophrase

    hasil = collection.find({}, {"_id": 0,  "int32":0})
#     hasil = collection.find({})

    hasils = []

    for document in hasil:

        hasils.append((json.dumps(document['directlink']),json.dumps(document['paragraf'])))
#         hasils.append(json.dumps(document['label']))
    return hasils
data = load_data() # data =  list


# #### pendefinisian fungsi

# In[37]:


def tokenize(text): # untuk melakukan tokenisasi
    # obtains tokens with a least 1 alphabet
    tokens = []
    for i in text:
        pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(i.lower())

def remove_punctuation(list_tandabaca): #untuk menghapus tanda baca
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



# In[38]:


def remove_n(data): #fungsi ini berguna untuk menghilangkan bentuk (//n) dan membuat jadi lower case
    datas = re.findall("['\w']+", str(data[221:])) #data[0] = string , datas  = list
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
    list_data,res = [], []
    list_data.append(new_s)
    for i in list_data:
        i = re.sub(" \d+", "", i)
        res.append(i)
        
    return res


# In[41]:


#sekarang masukkan fungsinya ke dalam variabel
yes = remove_n(data)


# In[42]:


#cuman untuk nyimpan si data ke dalam variabel baru
data_new = data


# #### preprocessing untuk data dict

# #### dict itu berisi x dan y, dimana x adalah directlink dan y yang berisi isi dari dokumen

# In[44]:


from nltk import everygrams

counter = 0

for x, y in data_new:
    y = remove_n(y) # jadi cuman y-nya aja yang mau diedit 
    coba = everygrams(y[0].split(),2,6) # ditambanglah si y jadi ngram, mulai dari 2 hingga 6
    lalas=[]
    for i in coba:
        lala = ' '.join(i) #kita joinkan bentuknya, yang awalnya ('aku', 'adalah') menjadi ('aku adalah')
        lalas.append(lala)
    y = lalas
#     print(y)
    
    
    data_new[counter] = (x,y) #dilakukan supaya nanti tiap beda i nya, maka bisa disesuaikan sama yang lain
    counter += 1


# #### data yang ada pada dict itu dimasukkan ke dalam file csv

# In[46]:


#dalam file csv, ada dua kolom, yaitu kolom dengan head link dan kata (yang berisi hasil penambangan n-gram)
hasil = []
for i in range(len(data_new)):
    hasil.append({'Link' : data_new[i][0], 'kata' : data_new[i][1]})

df = pd.DataFrame(hasil , columns=['Link', 'kata'])
df.to_csv('DataLabel1.csv', index=True, encoding='utf-8', sep = ',')


# #### mau coba tahapan preprocessing pada data labelnya, karena udah beda bentuk sama yang jadi data training

# #### seharunys proses selanjutnya sama dengan yang ada pada data training, hanya aja bentuk dari data test udah beda, jadi harus di modif gitu

# In[47]:


def w2v_preprocessinglabel(df):
    """ All the preprocessing steps for word2vec are done in this function.
    All mutations are done on the dataframe itself. So this function returns
    nothing.
    """
    res,ress,res2 = [],[],[]
    for i in range(len(df['kata'])):
        hasil = []
        for j in df['kata'][i].split(','):
            j = j.lower()
            j = re.sub(r'[^\w\s]', '', j)
            hasil.append(j)
        res.append(hasil)
    df['textlabel']= pd.Series(res)
    
    for i in range(len(df.textlabel)):
        hasil = []
        for j in df.textlabel[i]:
            hasil.append([j])
        ress.append(hasil)
    #print(ress)
    df['document_sentenceslabel'] = pd.Series(ress)
            
    for i in range(len(df.document_sentenceslabel)):
        hasil = []
        for j in df.document_sentenceslabel[i]:
            kata = []
            for k in j[0].split():
                kata.append(k)
            hasil.append(kata)
            #print(hasil)
        res2.append(hasil)
        
    df['tokenized_sentenceslabel'] = pd.Series(res2)
    print(df['tokenized_sentenceslabel'])  # sampai disini, udah sama bentuknya sama prosesnya si data training


# In[182]:


w2v_preprocessinglabel(test_data)


# #### mencoba untuk mengekstrak data test supaya bisa dibaca oleh komputer

# In[120]:


def get_w2v_featureslabel(w2v_model, sentence_group):
    """ Transform a sentence_group (containing multiple lists
    of words) into a feature vector. It averages out all the
    word vectors of the sentence_group.
    """
    words = np.concatenate(sentence_group)  # words in text
#     print(words)
    index2word_set = set(w2v_model.wv.vocab.keys())  # words known to model
#     print(index2word_set)
    featureVec = np.zeros(w2v_model.vector_size, dtype="float32")
#     print(featureVec)
    # Initialize a counter for number of words in a review
    nwords = 0
    # Loop over each word in the comment and, if it is in the model's vocabulary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            featureVec = np.add(featureVec, w2v_model[word])
            nwords += 1.

    # Divide the result by the number of words to get the average
    if nwords > 0:
        featureVec = np.divide(featureVec, nwords)
    return featureVec


# print(lala)


# In[180]:


test_data = pd.read_csv('data_yang_akan_dilabeli.csv')


# In[ ]:


w2v_preprocessinglabel(test_data)


# #### inilah dia bagian yang mau diesktrak

# In[121]:


#dia makek fungsi get_w2v_featureslabel untuk menghasilkan fitur dengan menggunakan model W2Vmodel
test_data['w2v_features'] = list(map(lambda sen_group:
                                      get_w2v_featureslabel(W2Vmodel, sen_group),
                                      test_data.tokenized_sentenceslabel))
#     lala.append(test_data['w2v_features'])


# In[122]:


print(test_data['w2v_features'])


# In[68]:


print(test_data.tokenized_sentenceslabel)


# #### karena data_train.tokenizedsentences itu bentuknya pd.Series, jadi seharusnya data_test.tokenized_sentenceslabel itu juga jadi bentuk series

# In[183]:


lala = []
for i in range(len(test_data.tokenized_sentenceslabel)):
    for x in test_data.tokenized_sentenceslabel[1]:
        hasil = pd.Series((v[0] for v in x)) #mau buat dia jadi bentuk series
        print(hasil)
#        


# In[ ]:


X_test_w2v = np.array(list(map(np.array, test_data.w2v_features)))


# #### untuk memprediksi hasil dari klasifikasi si model clf tadi

# In[ ]:


submission_predictions = clf.predict(X_test_w2v)


# #### percobaan

# In[185]:


for i in train_data.kata:
    print(i)


# In[192]:


for i in test_data.kata[:2]:
    print(i)


# In[126]:


# print(len(test_data.tokenized_sentenceslabel))
for i in test_data.tokenized_sentenceslabel:
    print(i[1])
    print(len(i))
#     for j in i:
#         print('yang ini j ya', j)


# In[90]:


X_test_w2v = np.array(list(map(np.array, lala)))
print(X_test_w2v)


# In[93]:


print(len((X_test_w2v)))


# In[123]:


X_test_w2v = np.array(list(map(np.array, test_data.w2v_features)))


# In[124]:


submission_predictions = clf.predict(X_test_w2v)


# In[125]:


print(submission_predictions)

