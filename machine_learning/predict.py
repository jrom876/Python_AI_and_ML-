## https://www.w3schools.com/python/python_ml_polynomial_regression.asp
## https://machinelearningmastery.com/start-here/
'''
Predict Future Values

Now we can use the information we have gathered to predict future values.

Example: Let us try to predict the speed of a car that passes the tollbooth at around the time 17:00:

To do so, we need the same mymodel array from the example above:

mymodel = numpy.poly1d(numpy.polyfit(x, y, 3)) 
'''

import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

## Good Fit
x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

## Bad Fit
# ~ x = [89,43,36,36,95,10,66,34,38,20,26,29,48,64,6,5,36,66,72,40]
# ~ y = [21,46,3,35,67,95,53,72,58,10,26,34,90,33,38,20,56,2,47,15]

## NumPy has a method that lets us make a polynomial model:
mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))

# ~ R-Squared
# ~ It is important to know how well the relationship between the 
# ~ values of the x- and y-axis is. 
# ~ If there is no relationship, the polynomial regression cannot be used to predict anything.
# ~ The relationship is measured with a value called the r-squared.
# ~ The r-squared value ranges from 0 to 1, 
# ~ where 0 means no relationship, and 1 means 100% related.
print("R squared\t", r2_score(y, mymodel(x)))

## Then specify how the line will display, we start at position 1, and end at position 22:
myline = numpy.linspace(1, 22, 100)

# ~ mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))
speed = mymodel(17)
print("speed\t",speed)

plt.scatter(x, y)
plt.plot(myline, mymodel(myline))
plt.show()
