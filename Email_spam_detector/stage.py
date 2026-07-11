import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
data=pd.read_csv("Email_spam_detector/SMS_spam_dataset.csv")
df=pd.DataFrame(data)

print(df.columns)
print(df.isnull().sum())
df=df.drop("id",axis=1)
print(df.columns)

#encoding the label:-
le=LabelEncoder()
df["spamORham"]=le.fit_transform(df["spamORham"])

X=df["Message"]
y=df["spamORham"]

#splitting the data:-
X_train,X_test,y_train,y_test=train_test_split(X,y)

#Using the CountVectorizer:-
cv=CountVectorizer()# counting the number of times each word is repeated in each document 
X_train=cv.fit_transform(X_train)
#fit():-find the unique words from the training data and carete the vocalubaly
#transform():-it convert each word from the vaocab and converts it into a vector of word counts
X_test=cv.transform(X_test)#it learns from the same earlier created training vocabulary and returns the vector count of word

#MultinomialNB:-
model=MultinomialNB()
model.fit(X_train,y_train)#traingin the model
y_pred=model.predict(X_test)#making the prediction

#mdel evaluation:
accuracy=accuracy_score(y_test,y_pred)
precision=precision_score(y_test,y_pred)
recall=recall_score(y_test,y_pred)
f1=f1_score(y_test,y_pred)
cm=confusion_matrix(y_test,y_pred)
print("accuracy:",accuracy)
print("precision:",precision)
print("recall:",recall)
print("f1-score",f1)
print("confusion-matrix:",cm)