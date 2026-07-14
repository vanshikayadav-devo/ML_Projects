import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,f1_score,confusion_matrix

data=pd.read_csv("Titanic_Survival_Predictor/Titanic_train_dataset.csv")
df=pd.DataFrame(data)

print(df.columns)
print(df.info())
print(df.isnull().sum())
df["Age"]=df["Age"].fillna(df["Age"].mean())
df["Cabin"]=df["Cabin"].fillna(df["Cabin"].mode()[0])
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])

print(df.isnull().sum())

#encoding:
le=LabelEncoder()
df["Sex"]=le.fit_transform(df["Sex"])
df["Ticket"]=le.fit_transform(df["Ticket"])
df["Cabin"]=le.fit_transform(df["Cabin"])
df["Embarked"]=le.fit_transform(df["Embarked"])
print(df.head(5))

X_drop=df.drop(["Name","Survived"],axis=1)
y=df["Survived"]

#spitting the data:
X_train,X_test,y_train,y_test=train_test_split(X_drop,y,test_size=0.2,random_state=42)

#RandomForest Classifier:
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

#model evaluation:
accuracy=accuracy_score(y_test,y_pred)
precision=precision_score(y_test,y_pred)
f1=f1_score(y_test,y_pred)
cm=confusion_matrix(y_test,y_pred)

print("accuracy:",accuracy)
print("precision:",precision)
print("f1_score:",f1)
print("confusion-matrix:",cm)