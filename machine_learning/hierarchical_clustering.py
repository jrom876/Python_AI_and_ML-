## https://www.w3schools.com/python/python_ml_hierarchial_clustering.asp
import sys

import tkinter as tk
from tkinter import *

import matplotlib 
# ~ matplotlib.use('Agg')
matplotlib.use('QtAgg')
# ~ matplotlib.use('TkAgg')
# ~ matplotlib.use('SVG')

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering

#############################################
############### TKINTER SETUP ###############
root = tk.Tk()
root.title("Confusion Matrix")
# ~ root.geometry("1000x500")
# ~ root.geometry("1200x750")
root.geometry("1200x900")

x = [4, 5, 10, 4, 3, 11, 14 , 6, 10, 12]
y = [21, 19, 24, 17, 16, 25, 24, 22, 21, 21]

data = list(zip(x, y))

hierarchical_cluster = AgglomerativeClustering(n_clusters=2, linkage='ward')
labels = hierarchical_cluster.fit_predict(data)

plt.scatter(x, y, c=labels)
plt.show() 

# ~ #Two  lines to make our compiler able to draw:
plt.savefig(sys.stdout.buffer)
sys.stdout.flush()
