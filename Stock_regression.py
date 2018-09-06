import pandas as pd
import quandl
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import datetime

style.use('ggplot')

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, cross_validation, svm

df = quandl.get("EOD/HD",start_date="2000-01-01",end_date="2018-08-31", authtoken="gz9HbWBHnH4FrePQzTie")
df = df[["Adj_Close"]]
# print(df.tail())
forecast_out = int(30) # predicting 30 days into future
df['Prediction'] = df[['Adj_Close']].shift(-forecast_out)
# print(df.head())

X = np.array(df.drop(['Prediction'], 1))
X = preprocessing.scale(X)

X_forecast = X[-forecast_out:]
X = X[:-forecast_out]

y = np.array(df['Prediction'])
y = y[:-forecast_out]



X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)



# Training
clf = LinearRegression()
clf.fit(X_train,y_train)
# Testing
confidence = clf.score(X_test, y_test)
print("confidence: ", confidence)
forecast_prediction = clf.predict(X_forecast)
df['Forecast'] = np.nan
print(forecast_prediction)

for k in ['linear','poly','rbf','sigmoid']:
    clf = svm.SVR(kernel=k)
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print(k,confidence)

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_prediction:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

df['Adj_Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()