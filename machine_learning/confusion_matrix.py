## https://www.w3schools.com/python/python_ml_confusion_matrix.asp

'''
What is a confusion matrix?

It is a table that is used in classification problems to assess where errors in the model were made.

The rows represent the actual classes the outcomes should have been. While the columns represent the predictions we have made. Using this table it is easy to see which predictions are wrong.
Creating a Confusion Matrix

Confusion matrixes can be created by predictions made from a logistic regression.

For now we will generate actual and predicted values by utilizing NumPy:

import numpy

Next we will need to generate the numbers for "actual" and "predicted" values.

actual = numpy.random.binomial(1, 0.9, size = 1000)
predicted = numpy.random.binomial(1, 0.9, size = 1000)

In order to create the confusion matrix we need to import metrics from the sklearn module.

from sklearn import metrics

Once metrics is imported we can use the confusion matrix function on our actual and predicted values.

confusion_matrix = metrics.confusion_matrix(actual, predicted)

To create a more interpretable visual display we need to convert the table into a confusion matrix display.

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])

Vizualizing the display requires that we import pyplot from matplotlib.

import matplotlib.pyplot as plt

Finally to display the plot we can use the functions plot() and show() from pyplot.

cm_display.plot()
plt.show()

See the whole example in action:
'''
### https://stackoverflow.com/questions/21688409/matplotlib-plt-show-isnt-showing-graph
### pip install PyQt6

#Three lines to make our compiler able to draw:
import sys

import tkinter as tk
from tkinter import *

import matplotlib 
# ~ matplotlib.use('Agg')
matplotlib.use('QtAgg')
# ~ matplotlib.use('TkAgg')
# ~ matplotlib.use('SVG')


import matplotlib.pyplot as plt
import numpy
from sklearn import metrics

#############################################
############### TKINTER SETUP ###############
root = tk.Tk()
root.title("Confusion Matrix")
# ~ root.geometry("1000x500")
# ~ root.geometry("1200x750")
root.geometry("1200x900")
# ~ column_size = 60
# ~ row_size = 25

#############################################
def get_accuracy(actual, predicted):
	Accuracy = metrics.accuracy_score(actual, predicted)
	print("Accuracy = {0}".format(Accuracy))
	return Accuracy
#############################################

#############################################
def get_precision(actual, predicted):
	Precision = metrics.precision_score(actual, predicted)
	print("Precision = {0}".format(Precision))
	return Precision
#############################################

#############################################
def get_sensitivity(actual, predicted):
	Sensitivity_recall = metrics.recall_score(actual, predicted)
	print("Sensitivity_recall = {0}".format(Sensitivity_recall))
	return Sensitivity_recall
#############################################

#############################################
def get_specificity(actual, predicted):
	Specificity = metrics.recall_score(actual, predicted, pos_label=0)
	print("Specificity = {0}".format(Specificity))
	return Specificity
#############################################

#############################################
def get_F1_score(actual, predicted):
	F1_score = metrics.f1_score(actual, predicted, pos_label=0)
	print("F1_score = {0}".format(F1_score))
	return F1_score
#############################################

actual = numpy.random.binomial(1,.9,size = 1000)
predicted = numpy.random.binomial(1,.9,size = 1000)

if __name__ == '__main__':  
	
	Accuracy = metrics.accuracy_score(actual, predicted)
	Precision = metrics.precision_score(actual, predicted)
	Sensitivity_recall = metrics.recall_score(actual, predicted)
	Specificity = metrics.recall_score(actual, predicted, pos_label=0)
	F1_score = metrics.f1_score(actual, predicted, pos_label=0)
	
	print({"Accuracy":Accuracy,"Precision":Precision,
			"Sensitivity_recall":Sensitivity_recall,
			"Specificity":Specificity,"F1_score":F1_score})
	confusion_matrix = metrics.confusion_matrix(actual, predicted)
	# ~ get_accuracy(actual, predicted)
	cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])

	cm_display.plot()
	plt.show()

	#Two  lines to make our compiler able to draw:
	plt.savefig(sys.stdout.buffer)
	sys.stdout.flush()
