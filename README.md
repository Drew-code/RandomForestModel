# Random Forest Model
A quick start guide to using random forest models in Python  
## Introduction  
Random Forest Modeling is great to use when decision tree modeling results seem to be strongly correlated to one variable 
in the dataset. Random forest picks as the the name implies, random rows out of the dataset and removes variables at random in those selections. It then averages out the results of those datasets to give you a more unbiased prediction. 
Often times this is run in conjuction with a decision tree model. the dataset that will be used in this example, are amazon 
stock prices that were pulled. It shows weather or not the stock changed a certain percentage.
 
## Prerequisites
1. Python 3.7 or newer  
2. Scikit-Learn module for Python  
3. Pandas module for Python  
4. Numpy Modules for Python
  
## Walkthough  
Start by importing all modules you will need at the beginning of your script. These include: Pandas, Scikit-Learn,  
and Numpy.  
```
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
```  
Next import the csv file and clean it up a little. Below we are importing the csv file as an object "stock"  
then dropping any unwanted columns or variables by setting "axis=1". If you do not set this, it will remove rows instead.  
I then like to print the info as shown below, to look at the data I am working with. Printing the head of your data will  
also give you a snap shot of the first five rows of your data, so you are able to visualize what you are working with.  
```
stock = pd.read_csv(r"D:\Datasets\Amazon_Stock.csv")
stock = stock.drop('time_stamp',axis=1)
print(stock.info())
print(stock.head())
```  
Now it is time to declare your X and Y variables.  X is going to be your independant variables; the variables you know  
and want to use to predict the Y variable. Y is your dependant variable; the data point you would like to predict.  
As shown below, we are going to use our whole dataframe which the exception of the y_variable to create our X variables.  
We then create our y by using the y_variable in our data that we want to predict.  
```
X = stock.drop(['Change'],axis=1)
y = stock['Change']
```  
Now that the variables have been established, it is time to split the data into two parts. The training set  
is what our model is going to look at and learn from. The model will then try to apply what it has learned to the test data.  
The "test_size" argument, is set to 0.3 in this example. That means that 70% of the data will be used to train the model  
and the remaining 30% will be used to test how accurate the model is. The "random_state" parameter sets the random  
number generator. This makes it so we are able to replicate results.  
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```


With all of the data cleaned and prepped, all that is left to do is create the model and start training it.  
First, create an instance of the model that is being used, in this example it is Random Forest Classifier. We than have to enter the number of estimators or the number of trees in the forest. Next we have to fit   the model,this is another way of saying train the model. It needs to be fed both X and y train, so the model can see all the   variables and then see what the correct result was for each line of data. Finally it is time to predict. The model is given X_test data to predict what the y_test   result is.
```
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
rfc_prediction = rfc.predict(X_test)

```  
To see how accurate the model is, we need to print out a couple of reports. The main report I use is the confusion matrix.  
This shows the number of true positives, false positives, true negatives and false negatives. It also gives an overall accuracy  
score. There are other great reports you can use in conjuction with this, such as the classification report. This is great  
if you have more than two groups the result can end up in.  
```
print("Random Forest Classification report")
print(classification_report(y_test,rfc_prediction))
print("Random Forest Confusion Matrix")
print(confusion_matrix(y_test,rfc_prediction))
```
