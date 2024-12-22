## https://www.w3schools.com/python/python_ml_multiple_regression.asp
## https://machinelearningmastery.com/start-here/
'''

'''

import pandas
import numpy as np
from sklearn import linear_model

df = pandas.read_csv("data.csv")

X = df[['Weight', 'Volume']]
y = df['CO2']

regr = linear_model.LinearRegression()
regr.fit(X, y)

print(regr.coef_) 

predictedCO2 = regr.predict([[3300, 1300]])
print(predictedCO2) 
