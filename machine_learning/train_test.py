## https://www.w3schools.com/python/python_ml_train_test.asp

'''
Evaluate Your Model

In Machine Learning we create models to predict the outcome of certain events, like in the previous chapter where we predicted the CO2 emission of a car when we knew the weight and engine size.

To measure if the model is good enough, we can use a method called Train/Test.

What is Train/Test

Train/Test is a method to measure the accuracy of your model.

It is called Train/Test because you split the data set into two sets: a training set and a testing set.

80% for training, and 20% for testing.

You train the model using the training set.

You test the model using the testing set.

Train the model means create the model.

Test the model means test the accuracy of the model.

'''

# ~ Start With a Data Set

# ~ Start with a data set you want to test.
# ~ Our data set illustrates 100 customers in a shop, and their shopping habits.

# ~ import numpy
# ~ import matplotlib.pyplot as plt
# ~ numpy.random.seed(2)

# ~ x = numpy.random.normal(3, 1, 100)
# ~ y = numpy.random.normal(150, 40, 100) / x

# ~ train_x = x[:80]
# ~ train_y = y[:80]

# ~ test_x = x[80:]
# ~ test_y = y[80:] 

## plt.scatter(x, y)
# ~ plt.scatter(test_x, test_y)
# ~ plt.show() 

###############################

import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
numpy.random.seed(2)

x = numpy.random.normal(3, 1, 100)
y = numpy.random.normal(150, 40, 100) / x

train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))

r2 = r2_score(train_y, mymodel(train_x))
print("train r2\t",r2)
r2 = r2_score(test_y, mymodel(test_x))
print("test r2 \t",r2)

## Example
## How much money will a buying customer spend, if she or he stays in the shop for 5 minutes?
print(mymodel(5))
myline = numpy.linspace(0, 6, 100)

plt.scatter(train_x, train_y)
plt.plot(myline, mymodel(myline))
plt.show() 
