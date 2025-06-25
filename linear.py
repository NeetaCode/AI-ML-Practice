import matplotlib
matplotlib.use('TKAgg')

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import pandas as pd

df = pd.read_csv("C:\\Users\\neeta\\Downloads\\Housing.csv")


Y = df['price']
X = df['lotsize']

X = X.to_numpy().reshape(len(X), 1)
Y = Y.to_numpy().reshape(len(Y), 1)

X_train = X[:-250]
X_test = X[-250:]
Y_train = Y[:-250]
Y_test = Y[-250:]

plt.scatter(X_test, Y_test, color='blue')

plt.title('Test Data')
plt.xlabel('Size')
plt.ylabel('Price')
plt.xticks(())
plt.yticks(())

rgr = linear_model.LinearRegression()
rgr.fit(X_train, Y_train)

plt.plot(X_test, rgr.predict(X_test),color='Red', linewidth=3)
plt.show()

