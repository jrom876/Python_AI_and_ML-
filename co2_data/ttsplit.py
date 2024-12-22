## https://realpython.com/train-test-split-python-data/
## https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset

import numpy as np
import california_housing as chous
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

##################################################
### test data options and debug script

# ~ x = np.arange(1, 25).reshape(12, 2)
# ~ y = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0])

# ~ x = np.arange(20).reshape(-1, 1)
# ~ y = np.array([5, 12, 11, 19, 30, 29, 23, 40, 51, 54, 74,
			# ~ 62, 68, 73, 89, 84, 89, 101, 99, 106])
			
x, y = fetch_california_housing(return_X_y=True)

# ~ print(x)
# ~ array([[ 1,  2],
       # ~ [ 3,  4],
       # ~ [ 5,  6],
       # ~ [ 7,  8],
       # ~ [ 9, 10],
       # ~ [11, 12],
       # ~ [13, 14],
       # ~ [15, 16],
       # ~ [17, 18],
       # ~ [19, 20],
       # ~ [21, 22],
       # ~ [23, 24]])

# ~ print(y)
# ~ array([0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0])

chous.show_calif_hous_data(25)

##################################################
### train_test_split call options and debug script

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
# ~ x_train, x_test, y_train, y_test = train_test_split(x, y, 
													# ~ test_size=4, 
													# ~ random_state=0,
													# ~ stratify=y,
													# ~ shuffle=True)

print(x_train)
print(x_test)
print(y_train)
print(y_test)


####################################################
### Regressor function call options and debug script

print("\nLinear Regressor")
model = LinearRegression().fit(x_train, y_train)
print("intercept {0}".format(model.intercept_))
print("slope {0}".format(model.coef_))
print("score train {0}".format(model.score(x_train, y_train)))
print("score test {0}".format(model.score(x_test,y_test)))

print("\nGradient Boosting Regressor")
model = GradientBoostingRegressor(random_state=0).fit(x_train, y_train)
print(model.score(x_train, y_train))
print(model.score(x_test,y_test))

# ~ print("\nRandom Forest Regressor")
# ~ model = RandomForestRegressor(random_state=0).fit(x_train, y_train)
# ~ print(model.score(x_train, y_train))
# ~ print(model.score(x_test,y_test))


