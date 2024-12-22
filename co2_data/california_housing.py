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
