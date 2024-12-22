## https://www.w3schools.com/python/python_ml_mean_median_mode.asp
## https://machinelearningmastery.com/start-here/

####### MEAN VALUE #######
'''
    Mean - The average value
    Median - The mid point value
    Mode - The most common value
    
The mean value is the average value.
To calculate the mean, find the sum of all values, and divide the sum by the number of values:
(99+86+87+88+111+86+103+87+94+78+77+85+86) / 13 = 89.77

The median value is the value in the middle, after you have sorted all the values:
It is important that the numbers are sorted before you can find the median.
77, 78, 85, 86, 86, 86, 87, 87, 88, 94, 99, 103, 111 ==> median = 87

The Mode value is the value that appears the most number of times:
99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86 = 86

'''

import numpy
from scipy import stats

speed = [99,86,87,88,86,103,87,94,78,77,85,86]

x = numpy.mean(speed)
y = numpy.median(speed)
z = stats.mode(speed)

print(x)
print(y)
print(z)
