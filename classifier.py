#### building automl covid 19 fake news classifier ####

# importy all required modules
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import pickle
from supervised.automl import AutoML
from sklearn.metrics import f1_score, precision_score, recall_score

# import data sets as splitted by the author - train - val - test
df_train = pd.read_csv(r'data\newdata\Constraint_Train.csv')
df_val = pd.read_csv(r'data\newdata\Constraint_Val.csv')
df_test = pd.read_csv(r'data\newdata\english_test_with_labels.csv')

# rename columns
df_train.columns = ['id', 'Text', 'Label']
df_val.columns = ['id', 'Text', 'Label']
df_test.columns = ['id', 'Text', 'Label']

# append train vald
data = df_train.append([df_val])

# select 40% of the data
data = data.sample(frac = 0.4, random_state = 29)
df_test = df_test.sample(frac = 0.4, random_state = 29)

# remap labels
data = data.replace({'Label': {'real': 0, 'fake': 1}})
df_test = df_test.replace({'Label': {'real': 0, 'fake': 1}})

# show first 5 rws for validation
print(data.head(5))

# Data Exploration and Data Engineering
data['Label'].value_counts()

# Filling Missing Values
data.isnull().sum()
data=data.fillna(' ')
print(data.isnull().sum())
data['Text'] = data['Text'].str.replace('[^\w\s]','')
data['Text'] = data['Text'].str.lower()

# Word Cloud Visualization
plt.figure(figsize = (20,20)) 
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(df_val.Text))
plt.imshow(wc, interpolation = 'bilinear')

# second cloud
plt.figure(figsize = (20,20)) 
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(data[data.Label == 1].Text))
plt.imshow(wc, interpolation = 'bilinear')

# Train-Test Split
X_train = data.Text.reset_index(drop = True)
y_train = data.Label.reset_index(drop = True)
X_test = df_test.Text.reset_index(drop = True)
y_test = df_test.Label.reset_index(drop = True)

# Validate the shapes of training and test
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Cleaning Data using NLP Techniques
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.80)  
tfidf_train = tfidf_vectorizer.fit_transform(X_train) 
tfidf_test = tfidf_vectorizer.transform(X_test)

# Implementing Classification Algorithm
# autoML
automl = AutoML(mode="Compete")
tfidf_train = tfidf_train.toarray() 
automl.fit(tfidf_train, y_train)

# predictions and metrics
prediction = automl.predict(tfidf_test.toarray() )
print('Accuracy of AutoML on test set:', accuracy_score(y_test, prediction))
print('F1 score of AutoML on test set:', f1_score(y_test, prediction, average='macro'))
print('Precision of AutoML on test set:', precision_score(y_test, prediction, average='macro'))
print('Recall of AutoML on test set:', recall_score(y_test, prediction, average='macro'))

# confusion matrix
result = confusion_matrix(y_test.values, prediction)
print(result)


# save the model as pkl
Pkl_Filename = "automl_compete_model.pkl"
with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(automl, file)

