import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import csv
import seaborn as sns

nltk.download('stopwords')
nltk.download('punkt')

stopwords = nltk.corpus.stopwords.words('english')

#print(os.getcwd())

onlyfiles = [f for f in listdir('D:\Django\ll erc-20 text') if isfile(join('D:\Django\ll erc-20 text', f))]

texts = []
for text in onlyfiles:
    f = open("D:\Django\ll erc-20 text\{}".format(text),'r', encoding="utf8")
    texts.append(f.read())

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

totalvocab_stemmed = []
totalvocab_tokenized = []
for i in texts:
    allwords_stemmed = tokenize_and_stem(i) 
    totalvocab_stemmed.extend(allwords_stemmed) 
    
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)

vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
#print ('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')

# print(vocab_frame.head())

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
terms = tfidf_vectorizer.get_feature_names()


#Compute distances
from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)


dist_frame = pd.DataFrame(dist, index=onlyfiles , columns=onlyfiles)

dpi = 72.27
matrix_height_pt = float(20 * 94)
matrix_height_in = matrix_height_pt / dpi
top_margin = 0.04  
bottom_margin = 0.04
figure_height = matrix_height_in / (1 - top_margin - bottom_margin)
fig, ax = plt.subplots(
        figsize=(figure_height,figure_height), 
        gridspec_kw=dict(top=1-top_margin, bottom=bottom_margin))


ax = sns.heatmap(dist_frame, ax=ax)


plt.savefig('test.png')

#Hierarchal Clustering
from scipy.cluster.hierarchy import ward,dendrogram

linkage_matrix = ward(dist)


fig, ax = plt.subplots(figsize=(15, 20)) 
ax = dendrogram(linkage_matrix, orientation="right", labels=onlyfiles)

plt.tick_params(\
    axis= 'x',
    which='both',
    bottom='off',
    top='off',
    labelbottom='off')

plt.tight_layout()


plt.savefig('ward_clusters.png', dpi=200)


frame = pd.DataFrame(linkage_matrix,index = onlyfiles[1:])
plt.subplots(figsize=(15,100))
sns.set(font_scale=0.5)
sns.clustermap(frame,col_cluster=False,yticklabels=True)
plt.show()




