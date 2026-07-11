import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

data=pd.read_csv("Email_spam_detector/SMS_spam_dataset.csv")
df=pd.DataFrame(data)

df=df.drop("id",axis=1)
#encoding the label:
le=LabelEncoder()
df["spamORham"]=le.fit_transform(df["spamORham"])

#splitting  the data:
X=df["Message"]
y=df["spamORham"]
X_train,X_test,y_train,y_test=train_test_split(X,y)

#using the CountVectoorizer:-
cv=CountVectorizer(binary=True)
X_train=cv.fit_transform(X_train)
X_test=cv.transform(X_test)

#making the Bernoulli Naive Bayse:-
model=BernoulliNB()
model.fit(X_train,y_train)# training the data
y_pred=model.predict(X_test)

#model evaluation:
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