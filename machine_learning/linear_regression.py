## https://www.w3schools.com/python/python_ml_mean_median_mode.asp
## https://machinelearningmastery.com/start-here/
'''
R for Relationship

It is important to know how the relationship between the values of the 
x-axis and the values of the y-axis is, if there are no relationship 
the linear regression can not be used to predict anything.

This relationship - the coefficient of correlation - is called r.

The r value ranges from -1 to 1, where 0 means no relationship, and 1 (and -1) means 100% related.

Python and the Scipy module will compute this value for you, all you have to do is feed it with the x and y values.

slope, intercept, r, p, std_err = stats.linregress(x, y)
'''
import matplotlib.pyplot as plt
from scipy import stats

## Good fit
x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

## Bad fit
# ~ x = [89,43,36,36,95,10,66,34,38,20,26,29,48,64,6,5,36,66,72,40]
# ~ y = [21,46,3,35,67,95,53,72,58,10,26,34,90,33,38,20,56,2,47,15]

slope, intercept, r, p, std_err = stats.linregress(x, y)

print("r = ", r)

def myfunc(x):
  return slope * x + intercept

speed = myfunc(12)
print(speed)

## Run each value of the x array through the function. 
## This will result in a new array with new values for the y-axis:
mymodel = list(map(myfunc, x))

## Draw the original scatter plot:
plt.scatter(x, y)

## Draw the line of linear regression:
plt.plot(x, mymodel)

## Display the diagram.
plt.show()
