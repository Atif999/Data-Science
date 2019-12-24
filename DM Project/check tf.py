import pandas as pd

import numpy as np
import math
from nltk.tokenize import  word_tokenize
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer




train=pd.read_csv('train.csv')
train_col = train[['id', 'question1', 'question2', 'is_duplicate']]
tfe=TfidfVectorizer(analyzer='word',use_idf=False)


tfe.fit(pd.concat((train_col['question1'],train_col['question2'])).unique())
trainq1_trans = tfe.transform(train_col['question1'].values)
trainq2_trans = tfe(train_col['question2'].values)

result=cosine_similarity(trainq1_trans,trainq2_trans)

print(result)
