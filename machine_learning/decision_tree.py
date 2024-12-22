## https://www.w3schools.com/python/python_ml_decision_tree.asp

'''
The Confusion Matrix created has four different quadrants:

True Negative (Top-Left Quadrant)
False Positive (Top-Right Quadrant)
False Negative (Bottom-Left Quadrant)
True Positive (Bottom-Right Quadrant)

True means that the values were accurately predicted, False means that there was an error or wrong prediction.
'''

##############################

#Three lines to make our compiler able to draw:
import sys
import tkinter as tk
from tkinter import *
import matplotlib 
matplotlib.use('QtAgg')

import matplotlib.pyplot as plt
import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

#############################################
############### TKINTER SETUP ###############
root = tk.Tk()
root.title("Decision Tree")
root.geometry("1200x900")

#############################################

if __name__ == '__main__':  
	
	df = pandas.read_csv("comedy_show.csv")
	# ~ print(df)

	d = {'UK': 0, 'USA': 1, 'N': 2}
	df['Nationality'] = df['Nationality'].map(d)
	d = {'YES': 1, 'NO': 0}
	df['Go'] = df['Go'].map(d)

	features = ['Age', 'Experience', 'Rank', 'Nationality']

	X = df[features]
	y = df['Go']

	# ~ print(X)
	# ~ print(y) 

	dtree = DecisionTreeClassifier()
	dtree = dtree.fit(X, y)
	print(dtree.predict([[40, 10, 7, 1]]))

	tree.plot_tree(dtree, feature_names=features) 
	plt.show()

	# ~ #Two  lines to make our compiler able to draw:
	plt.savefig(sys.stdout.buffer)
	sys.stdout.flush()

#NOTE:
#You will see that the Decision Tree gives you different results 
## if you run it enough times, even if you feed it with the same data.

#That is because the Decision Tree does not give us a 100% certain answer. 
## It is based on the probability of an outcome, and the answer will vary.


# ~ import pandas

# ~ df = pandas.read_csv("data.csv")

# ~ print(df)

# ~ d = {'UK': 0, 'USA': 1, 'N': 2}
# ~ df['Nationality'] = df['Nationality'].map(d)
# ~ d = {'YES': 1, 'NO': 0}
# ~ df['Go'] = df['Go'].map(d)

# ~ print(df)

# ~ features = ['Age', 'Experience', 'Rank', 'Nationality']

# ~ X = df[features]
# ~ y = df['Go']

# ~ print(X)
# ~ print(y) 
