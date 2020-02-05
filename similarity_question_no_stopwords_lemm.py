import pandas as pd
import numpy as np
import nltk
#nltk.download('punkt')
from sklearn.metrics.pairwise import pairwise_distances, linear_kernel, cosine_similarity,euclidean_distances,manhattan_distances

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import jaccard_similarity_score as jsc
from sklearn.model_selection import train_test_split
from  sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


#
df = pd.read_csv("train.csv")
df.question1 = df.question1.str.lower()
df.question2 = df.question2.str.lower()
df = df.dropna(axis=0)
#
# # print(df.describe())
# # print(df.info())
#
# similar_count=df.groupby('is_duplicate').count()
# #print(similar_count['id'])
#
duplicate=df[df.is_duplicate ==1].head(25000)
non_duplicate=df[df.is_duplicate ==0].head(25000)
#
data=pd.concat([duplicate,non_duplicate])
# #print(data.head())
#
import string
data2=data.copy()
def remove_puc(text):
    no_puc=''.join([c for c in text if c not in string.punctuation])
    return no_puc
#
#
# # data2['question1_tokens'] = data2.question1.map(nltk.word_tokenize)
# # data2['question2_tokens'] = data2.question2.map(nltk.word_tokenize)
#
data2['question1']=data2['question1'].apply(lambda x:remove_puc(x))
data2['question2']=data2['question2'].apply(lambda x:remove_puc(x))
#
#
tfidf = TfidfVectorizer(use_idf=True, ngram_range=(1,1))
data2['tfidf'] = data2[['question1', 'question2']].apply(lambda x:tfidf.fit_transform(x), axis=1)

data2['cosine_similarity'] = data2['tfidf'].apply(lambda x: cosine_similarity(x))
data2['euclidean_distances'] = data2['tfidf'].apply(lambda x: euclidean_distances(x))
data2['manhattan_distances'] = data2['tfidf'].apply(lambda x: manhattan_distances(x))
#data2['jaccard'] = data2['tfidf'].apply(lambda x: jsc)


sim=pd.DataFrame(columns={'id','cosine','euclidean','manhattan'})
sim['id']=data2['id'].copy()
sim['cosine']=data2['cosine_similarity'].copy()
sim['euclidean']=data2['euclidean_distances'].copy()
sim['manhattan']=data2['manhattan_distances'].copy()
sim['is_duplicated']=data2['is_duplicate'].copy()
#
# #print(sim.head())
# # print(data2.head(5))
# # print(data2.tail(5))
#
def get_val(arr):
  return arr[0][1]


sim['cosine'] = sim['cosine'].apply(lambda x: get_val(x))
sim['euclidean'] = sim['euclidean'].apply(lambda x: get_val(x))
sim['manhattan'] = sim['manhattan'].apply(lambda x: get_val(x))
#
sim.to_csv('sim_data.csv')

data_sim=pd.read_csv('sim_data.csv')
# print(data_sim.info())


#
x_train,x_test,y_train,y_test=train_test_split(data_sim[['cosine','euclidean','manhattan']],data_sim['is_duplicated'],test_size=0.2,random_state=2)

sgd=SGDClassifier(loss='hinge',penalty='l2',alpha=1e-3,random_state=42)

sgd.fit(x_train,y_train)
pred=sgd.predict(x_test)
print(accuracy_score(y_test,pred))