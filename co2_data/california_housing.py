## https://stackoverflow.com/questions/53184361/how-to-load-sklearn-datasets-manually

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

### https://scikit-learn.org/stable/api/sklearn.datasets.html
def show_calif_hous_data(num):	
	data = fetch_california_housing()
	calf_hous_df = pd.DataFrame(data= data.data, columns=data.feature_names)    
	print(calf_hous_df.head(num))


if __name__ == "__main__":
	
	show_calif_hous_data(25)

	# ~ data = fetch_california_housing()
	# ~ calf_hous_df = pd.DataFrame(data= data.data, columns=data.feature_names)    
	# ~ print(calf_hous_df.head(15))

### OPTIONAL:	train_test_split california housing data
	# ~ x, y = fetch_california_housing(return_X_y=True)
	# ~ print(x)
	# ~ print(y)

	# ~ x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)

	# ~ print("{0}".format(x_train))
	# ~ print(x_test)
	# ~ print(y_train)
	# ~ print(y_test)
