## https://www.w3schools.com/python/python_ml_multiple_regression.asp
## https://machinelearningmastery.com/start-here/
'''
Multiple regression is like linear regression, but with more than one independent value, 
meaning that we try to predict a value based on two or more variables.

Take a look at the data set below, it contains some information about cars.

Car 		Model 		Volume 	Weight 	CO2 
	
Toyota 		Aygo 		1000 	790 	99
Mitsubishi 	Space Star 	1200 	1160 	95
Skoda 		Citigo 		1000 	929 	95
Fiat 		500 		900 	865 	90
Mini 		Cooper 		1500 	1140 	105
VW 			Up! 		1000 	929 	105
Skoda 		Fabia 		1400 	1109 	90
Mercedes 	A-Class 	1500 	1365 	92
Ford 		Fiesta 		1500 	1112 	98
Audi 		A1 			1600 	1150 	99
Hyundai 	I20 		1100 	980 	99
Suzuki 		Swift 		1300 	990 	101
Ford 		Fiesta 		1000 	1112 	99
Honda 		Civic 		1600 	1252 	94
Hundai 		I30 		1600 	1326 	97
Opel 		Astra 		1600 	1330 	97
BMW 		1	 		1600 	1365 	99
Mazda 		3	 		2200 	1280 	104
Skoda 		Rapid 		1600 	1119 	104
Ford 		Focus 		2000 	1328 	105
Ford 		Mondeo 		1600 	1584 	94
Opel 		Insignia	2000 	1428 	99
Mercedes 	C-Class 	2100 	1365 	99
Skoda 		Octavia 	1600 	1415 	99
Volvo 		S60 		2000 	1415 	99
Mercedes 	CLA 		1500 	1465 	102
Audi 		A4 			2000 	1490 	104
Audi 		A6 			2000 	1725 	114
Volvo 		V70 		1600 	1523 	109
BMW 		5	 		2000 	1705 	114
Mercedes 	E-Class 	2100 	1605 	115
Volvo 		XC70 		2000 	1746 	117
Ford 		B-Max 		1600 	1235 	104
BMW 		2 			1600 	1390 	108
Opel 		Zafira 		1600 	1405 	109
Mercedes 	SLK 		2500 	1395 	120

We can predict the CO2 emission of a car based on the size of the engine, 
but with multiple regression we can throw in more variables, 
like the weight of the car, to make the prediction more accurate.

The Pandas module allows us to read csv files and return a DataFrame object.

'''
import numpy as np
import pandas as pd
# ~ from scikit-learn import linear_model
from sklearn import linear_model

df = pd.read_csv("data.csv")

X = df[['Weight', 'Volume']]
y = df['CO2']

regr = linear_model.LinearRegression()
regr.fit(X, y)

#predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm3:
predictedCO2 = regr.predict([[2300, 1300]])

print(predictedCO2) 
'''

'''
