# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 12:23:58 2023

@author: Talha
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

df = pd.read_csv('odev_salary.csv')



#veri setimdeki unvan_seviyesi, kidem, ve puan kolonları üzerinden\
#maaslar kolonunu ögrenmesini istiyorum. calısan id ve unvan kolonlarının
#modelim üzerine bir etkisinin olmayacağını düşündüğümden almadım

x = df.iloc[:, 2:5]
y = df.iloc[:, -1:]

#verilerin test ve split olarak ayrılması
from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(x,y , random_state=0, test_size=0.20)
"""
#verilerin olceklendirilmesi

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
"""

#multi-linear reg

from   sklearn.linear_model import LinearRegression

mlr = LinearRegression()

mlr.fit(x_train, y_train)

pred = mlr.predict(x_test)


#R^2 hesaplama
print("r2 linear")
print(r2_score(y_test, pred ))
#bu model sonucunda r2 negatif bir sayi cikti bu modelin hiç sağlıklı olmadığı anlamına gelir

#poly regression

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(x)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
print("r2 poly")
print(r2_score(y, lin_reg2.predict(poly_reg.fit_transform(x))))

"""

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(x_train)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y_train)

print(r2_score(y_train, lin_reg2.predict(poly_reg.fit_transform(x_train))))
"""
#Support vector regression

