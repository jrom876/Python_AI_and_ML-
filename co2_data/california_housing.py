## https://stackoverflow.com/questions/53184361/how-to-load-sklearn-datasets-manually
### https://scikit-learn.org/stable/api/sklearn.datasets.html

import pandas as pd
from sklearn.datasets import fetch_california_housing

def show_calif_hous_data(num):	
	data = fetch_california_housing()
	calif_hous_df = pd.DataFrame(data= data.data, columns=data.feature_names)    
	print(calif_hous_df.head(num))

if __name__ == "__main__":
	
	show_calif_hous_data(25)
