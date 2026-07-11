import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
data=pd.read_csv("Iris_Species_predictor/iris_dataset.csv")
df=pd.DataFrame(data)
print(df.head(5))
print(df.columns)
#labelEncoding
le=LabelEncoder()
df["species"]=le.fit_transform(df["species"])

print(df.head(5))
species=le.classes_#after the labelEncoding ,it returns the array where,0->setosa,1->versicolor,2->virginica
X_drop=df.drop("species",axis=1)
y=df["species"]

#StandardScaling:-
scaler=StandardScaler()
X_drop=pd.DataFrame(scaler.fit_transform(X_drop),columns=X_drop.columns)#since the result aftre scaling is a numpynd array so it scales the datafarme+ keep it as a dataframe

#splitting the data:-
X_train,X_test,y_train,y_test=train_test_split(X_drop,y,test_size=0.2,random_state=42)

#KNeighborsClassifier:-
model=KNeighborsClassifier(n_neighbors=3,metric="manhattan")
model.fit(X_train,y_train)
predict=model.predict(X_test)
print(species[predict[0]])

#model_Evaluation:-
accuracy=accuracy_score(y_test,predict)
precision=precision_score(y_test,predict, average="weighted")
recall=recall_score(y_test,predict, average="weighted")
f1=f1_score(y_test,predict, average="weighted")
cm = confusion_matrix(y_test, predict)

print(cm)
print(accuracy)
print(precision)
print(recall)
print(f1)




