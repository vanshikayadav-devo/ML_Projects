import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
data=pd.read_csv("California_Housing_Predictor/housing.csv")
df=pd.DataFrame(data)
print(df.columns)
print(df.isnull().sum())
df["total_bedrooms"]=df["total_bedrooms"].fillna(df["total_bedrooms"].mean())
print(df.isnull().sum())

#encoding
le=LabelEncoder()
df["ocean_proximity"]=le.fit_transform(df["ocean_proximity"])
print(le.classes_)

X_drop=df.drop('median_house_value',axis=1)
y=df['median_house_value']

#Scaling:
scaler=StandardScaler()
X_drop=pd.DataFrame(scaler.fit_transform(X_drop),columns=X_drop.columns)

#splitting the data:
X_train,X_test,y_train,y_test=train_test_split(X_drop,y,test_size=0.2,random_state=42)

#KNN:
model=KNeighborsRegressor(n_neighbors=7,metric="euclidean")
model.fit(X_train,y_train)
pred=model.predict(X_test)

#model evaluation:
mae=mean_absolute_error(y_test,pred)
mse=mean_squared_error(y_test,pred)
rmse=np.sqrt(mse)
r2=r2_score(y_test,pred)
print(mae)
print(mse)
print(rmse)
print(r2)







