import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score,confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
data=pd.read_csv("Titanic_Survival_Predictor/Titanic_train_dataset.csv")
df=pd.DataFrame(data)
print(df.columns)
print(df.isnull().sum())
df["Age"]=df["Age"].fillna(df["Age"].mean())
df["Cabin"]=df["Cabin"].fillna(df["Cabin"].mode()[0])
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])

print(df.isnull().sum())

#encoding :
print(df.info())
le=LabelEncoder()
df["Sex"]=le.fit_transform(df["Sex"])
df["Ticket"]=le.fit_transform(df["Ticket"])
df["Cabin"]=le.fit_transform(df["Cabin"])
df["Embarked"]=le.fit_transform(df["Embarked"])

X_drop=df.drop(["Survived","Name"],axis=1)
y=df["Survived"]

#Splitting the data:
X_train,X_test,y_train,y_test=train_test_split(X_drop,y,test_size=0.2,random_state=42)

#Model training and testing:
model=ExtraTreesClassifier(n_estimators=100,
                           criterion="gini",
                           max_depth=None,
                           random_state=42)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

#model evaluation:
accuracy=accuracy_score(y_test,y_pred)
precision=precision_score(y_test,y_pred)
recall=recall_score(y_test,y_pred )
f1_score=f1_score(y_test,y_pred)
cm=confusion_matrix(y_test,y_pred)

print("accuracy:",accuracy)
print("Precision:",precision)
print("Recall_score:",recall)
print("F1_score:",f1_score)
print("confusion-matrix:",cm)

disp=ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not-Survived","Survived"]#cm is basically means its using the above confusion-matrix
)
disp.plot(cmap="Greens")
plt.title("Confusion-Matrix")
plt.show()
