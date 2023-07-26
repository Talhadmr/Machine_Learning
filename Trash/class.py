# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 00:49:24 2023

@author: Talha
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics

df = pd.read_excel("iris.xls")

print(df.head())

x = df.iloc[:, 0:4].values
y = df["iris"].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(x_train) 


#logistic class
#normalize edilmiş değerler ile çalıştırdığmda tahminler bozuldu ????
from sklearn.linear_model import LogisticRegression

lrg = LogisticRegression(random_state= 23 )
lrg.fit(x_train, y_train)

pred = lrg.predict(x_test)

confusion_matrix = metrics.confusion_matrix(y_test, pred)

print("log")
print(confusion_matrix)

#knn class

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=11)

knn.fit(x_train, y_train)
pred = knn.predict(x_test)

confusion_matrix = metrics.confusion_matrix(y_test, pred)
print("knn")
print(confusion_matrix)

#svm class

from sklearn.svm import SVC

svc = SVC(kernel = "poly")

svc.fit(x_train,y_train)

pred = svc.predict(x_test)

confusion_matrix = metrics.confusion_matrix(y_test, pred)
print("svm")
print(confusion_matrix)

from sklearn.naive_bayes import CategoricalNB

gnb = CategoricalNB()

gnb.fit(x_train,y_train)

pred = gnb.predict(x_test)

confusion_matrix = metrics.confusion_matrix(y_test, pred)
print("gnb")
print(confusion_matrix)
