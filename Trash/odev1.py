import pandas as pd
import numpy as np
df = pd.read_csv("odev_tenis.csv")


#encoding 

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

df['windy'] = le.fit_transform(df['windy'])

# Get one hot encoding of columns B
one_hot = pd.get_dummies(df['outlook'])
# Drop column B as it is now encoded
df = df.drop('outlook',axis = 1)
# Join the encoded df
df = df.join(one_hot)

le = preprocessing.LabelEncoder()

df['play'] = le.fit_transform(df['play'])

y = df['play']
x_l = df.iloc[:, 1:2]
x_r = df.iloc[:, 4:]


x = pd.concat([x_r,x_l], axis = 1)

from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(x,y , random_state=42, test_size=0.30)

from   sklearn.linear_model import LinearRegression

mlr = LinearRegression()

mlr.fit(x_train, y_train)

pred = mlr.predict(x_test)

import statsmodels.api as sm

X = np.append(arr = np.ones((14,1)).astype(int), values=x, axis=1)

X_l = x.iloc[:,[0,1,2,3]].values
X = np.array(X_l,dtype=float)
model = sm.OLS(y,X_l).fit()
print(model.summary())
