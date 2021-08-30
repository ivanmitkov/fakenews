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

# import
df_fakenews = pd.read_csv(r'data\fakeNews.csv')[['Text', 'Binary Label']]
df_truenews = pd.read_csv(r'data\trueNews.csv')[['Text', 'Label']]

# data preparation
df_fakenews.columns = ['Text', 'Label']
data = df_fakenews.append(df_truenews).reset_index(drop = True)

# show
print(data.head(5))

# Data Exploration and Data Engineering
data['Label'].value_counts()

# Filling Missing Values
data.isnull().sum()
data=data.fillna(' ')
print(data.isnull().sum())
data['Text'] = data['Text'].str.replace('[^\w\s]','')
data['Text'] = data['Text'].str.lower()

# data validation
print(data['Text'][0])
print(data['Text'][45])

# Word Cloud Visualization
plt.figure(figsize = (20,20)) 
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(data[data.Label == 0].Text))
plt.imshow(wc, interpolation = 'bilinear')

# second cloud
plt.figure(figsize = (20,20)) 
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = STOPWORDS).generate(" ".join(data[data.Label == 1].Text))
plt.imshow(wc, interpolation = 'bilinear')

# Train-Test Split
y = data.Label
print(y)

data.drop("Label", axis=1,inplace=True)
X_train, X_test, y_train, y_test = train_test_split(data['Text'], y, test_size=0.2,random_state=102)

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

prediction = automl.predict(tfidf_test.toarray() )
print('Accuracy of AutoML on test set:', accuracy_score(y_test, prediction))
result = confusion_matrix(y_test, prediction)
print(prediction)


# save the model
Pkl_Filename = "automl_compete_model.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(automl, file)
