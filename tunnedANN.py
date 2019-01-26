# -*- coding: utf-8 -*-
"""
Created on Sun May 20 22:43:19 2018

@author: Dipanjan Chowdhury
"""


#Phase 1- Data Pre Processing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
train_x=train.iloc[:,[1,2,3,4,5,6,8,9,10,11,12,13,14]]
train_y=train.iloc[:,-1].values
val=test.iloc[:,[1,2,3,4,5,6,8,9,10,11,12,13,14]]

submission=pd.read_csv("sample.csv")
Pred=test.iloc[:,0]


#Figuring out the Objects
train_x.dtypes

#Separating the Categorical Data from the Dataset
obj_train_x=train_x.select_dtypes(include=["object"].copy())
obj_val_x=val.select_dtypes(include=["object"].copy())
#Checkig Missing values
train_x1=train.iloc[:,[3,4,11,12,14]]
train_x1.isnull().sum()

test_x1=test.iloc[:,[3,4,11,12,14]]
val.isnull().sum()
#Separating the Numerical Data From the Dataset
train_x1=train.iloc[:,[3,4,11,12,14]].values
val=test.iloc[:,[3,4,11,12,14]].values


city=train.iloc[:,[10]]
city1=test.iloc[:,[10]]
#Taking care of missing data
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values="NaN",strategy="median",axis=0)  

imputer.fit(train_x1[:,2:3])
train_x1[:,2:3]=imputer.transform(train_x1[:,2:3])

imputer.fit(train_x1[:,4:5])
train_x1[:,4:5]=imputer.transform(train_x1[:,4:5])

imputer.fit(val[:,2:3])
val[:,2:3]=imputer.transform(val[:,2:3])

imputer.fit(val[:,4:5])
val[:,4:5]=imputer.transform(val[:,4:5])


#Changing the train_x1 to DataFrame as we need to concat the dummies
train_x1=pd.DataFrame(train_x1)
val=pd.DataFrame(val)
#encoding the categorical variables
from sklearn.preprocessing import LabelEncoder,LabelBinarizer

#LabelEncoding the Categorical Data 
labelEncoder_x=LabelEncoder()
obj_train_x=obj_train_x.apply(labelEncoder_x.fit_transform)

lb=LabelBinarizer()
city=lb.fit_transform(city)
city=pd.DataFrame(city)

#for Test Dataset
labelEncoder_x1=LabelEncoder()
obj_val_x=obj_val_x.apply(labelEncoder_x1.fit_transform)

lb=LabelBinarizer()
city1=lb.fit_transform(city1)
city1=pd.DataFrame(city1)

#Creating dummy variables for the Labal Encoded data 
dummies=pd.concat([pd.get_dummies(obj_train_x[col]) for col in obj_train_x], axis=1, keys=obj_train_x.columns)

dummies1=pd.concat([pd.get_dummies(obj_val_x[col]) for col in obj_val_x], axis=1, keys=obj_val_x.columns)

#To avoid dummy variable trap reducing the the dummies-1 of each segments
dummies=dummies.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,30,33,34,36,38,39,40,41,43]]
city=city.iloc[:,1:]

dummies1=dummies1.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,30,33,34,36,38,39,40,41,43]]
city1=city1.iloc[:,1:]

#Let's Now combine all the dummies in the train DataSet
#axis=1 , for cloumn binding 
#axis=0 , for row binding

train_x1=pd.concat([dummies,city,train_x1],axis=1)
val=pd.concat([dummies1,city1,val],axis=1)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
train_x1=sc_x.fit_transform(train_x1)
val=sc_x.fit_transform(val)


#splitting dataset into test and train set 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(train_x1,train_y,test_size=0.25,random_state=0)





#Applying KNN
from sklearn.naive_bayes import GaussianNB
clf=GaussianNB()
clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)


y_pred=pd.DataFrame(y_pred)

#Putting In the submission format
Pred=pd.concat([Pred,y_pred],axis=1)
Pred.to_csv("KNNPred.csv",index=False,header=True)




#Making Confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred) 




#Tunning the ANN

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense


def build_classifier(optimizer,loss,activation):
    classifier=Sequential()
    classifier.add(Dense(units=23,kernel_initializer="uniform",activation='relu',input_dim=44))
    classifier.add(Dense(units=23,kernel_initializer="uniform",activation='relu'))
    classifier.add(Dense(units=1,kernel_initializer="uniform",activation=activation))   
    classifier.compile(optimizer=optimizer,loss=loss,metrics=["accuracy"])
    return classifier

classifier=KerasClassifier(build_fn=build_classifier)
parameters={"batch_size":[25,32],
            "epochs":[100,500],
            "optimizer":["adam","rmsprop","sgd","adamax","nadam"],
            "loss":["binary_crossentropy"],
            "activation":['sigmoid','softmax']}

grid_search=GridSearchCV(estimator=classifier,
                         param_grid=parameters,
                         scoring="accuracy",
                         cv=10)
grid_search=grid_search.fit(x_train,y_train)
best_parameters=grid_search.best_params_
best_accuracy=grid_search.best_score_


#predicting on the Test set 
pred=grid_search.predict(val).round()

pred=pd.DataFrame(pred)

pred.to_csv("prediction.csv",index=False)

prediction=pd.read_csv("prediction.csv")

submit=submit.drop(submit.columns[1],axis=1)
id=pd.DataFrame(test["id"].values)

submit=pd.concat([id,prediction],axis=1)
submit.to_csv("TunnedANN.csv",index=False,header=True)









