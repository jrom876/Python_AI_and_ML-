## https://www.w3schools.com/python/python_ml_mean_median_mode.asp
## https://machinelearningmastery.com/start-here/

'''
Percentiles are used in statistics to give you a number that describes 
the value that a given percent of the values are lower than.

Example: Let's say we have an array that contains the ages of every person living on a street.

ages = [5,31,43,48,50,41,7,11,15,39,80,82,32,2,8,6,25,36,27,61,31]

What is the 75. percentile? 
The answer is 43, meaning that 75% of the people are 43 or younger.
'''

import numpy

ages = [5,31,43,48,50,41,7,11,15,39,80,82,32,2,8,6,25,36,27,61,31]

if __name__ == '__main__':
	    
    x = numpy.percentile(ages, 75) ## == 43.0
    print(x)
    x = numpy.percentile(ages, 90) ## == 61.0
    print(x)
