import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

data=pd.read_csv("California_Housing_Predictor/housing.csv")
df=pd.DataFrame(data)

print(df.columns)
print(df.info())
print(df.isnull().sum())
df["total_bedrooms"]=df["total_bedrooms"].fillna(df["total_bedrooms"].mean())

print(df.isnull().sum())

#encoding:
le=LabelEncoder()
df["ocean_proximity"]=le.fit_transform(df["ocean_proximity"])
print(df.head(5))

X_drop=df.drop(["median_house_value"],axis=1)
y=df["median_house_value"]

#spitting the data:
X_train,X_test,y_train,y_test=train_test_split(X_drop,y,test_size=0.2,random_state=42)

#RandomForest Classifier:
model=RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

#model evaluation:
mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
rmse=np.sqrt(mse)

print("mean absolute error:",mae)
print("mean square error:",mse)
print("Root mean square error:",rmse)
print("r2-score:",r2)