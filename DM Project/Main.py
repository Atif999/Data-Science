import pandas as pd

import numpy as np
import math
from nltk.tokenize import  word_tokenize
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


def tf(question, word):
    if word not in question:
        return 0
    count = dict(Counter(question))
    q_len = len(question)
    return float(count[word]) / float(q_len)


train=pd.read_csv('train.csv')
train_col = train[['id', 'question1', 'question2', 'is_duplicate']]
tfe=TfidfVectorizer(analyzer='word',use_idf=False)




#print(train.shape)
#print(train_col)

for row in train_col.itertuples():
    if len(str(row[2])) > 10 and len(str(row[3])) > 10:
        wordvec1 = word_tokenize(row[2].lower())
        wordvec2 = word_tokenize(row[3].lower())
        words = wordvec1 + wordvec2
        words = list(set([word for word in words if word != '?']))
        #print(words)
        vec1 = []
        vec2 = []
        for word in words:

            #vec1.append(tf(wordvec1,word))
            #vec2.append(tf(wordvec2,word))

            vec1.append(tfe.fit_transform(wordvec1,word))
            vec2.append(tfe.fit_transform(wordvec2,word))

            #v1 = np.array(vec1).reshape(1,-1)
            #v2 = np.array(vec2).reshape(1,-1)

            #v1 = np.array(vec1)
            #v2 = np.array(vec2)


            #print(vec1)
            #print(vec2)

            #result=np.dot(v1, v2) / (np.sqrt(np.sum(v1 ** 2)) * np.sqrt(np.sum(v2 ** 2)))

            result=cosine_similarity(vec1,vec2)

            #print(vec1)
            #print(vec2)

            print(str(row[1]) + "," + str(result)+ '\n')

