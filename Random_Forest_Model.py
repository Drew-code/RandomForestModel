import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

stock = pd.read_csv(r"D:\Datasets\Amazon_Stock.csv")
stock = stock.drop('time_stamp',axis=1)
print(stock.info())
print(stock.head())

X = stock.drop(['Change'],axis=1)
y = stock['Change']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
rfc_prediction = rfc.predict(X_test)

print("Random Forest Classification report")
print(classification_report(y_test,rfc_prediction))
print("Random Forest Confusion Matrix")
print(confusion_matrix(y_test,rfc_prediction))