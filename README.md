import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("diabetes.csv")

dataset.info()

dataset.shape()

dataset.describe()

dataset.head()

#train and test
from sklearn.model_selection import train_test_split
x = df.iloc(:,df.columns!="Outcome")
y= df.iloc(:,df.columns=="Outcome")
print(x) 
print(y) 

xtrain , xtest , ytrain , ytest = train_test_split(x , y , test_size(0.2)

#algorithm
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(xtrain , ytrain_values_ravel())
predict_output = model.predict(xtest)

print(predict_output)
from sklearn.matrics import accuracy_score
acc = accuracy_score(predict_output , ytest)
print( 'The accuracy score for RF' , acc )

