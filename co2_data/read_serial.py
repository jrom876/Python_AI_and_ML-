### Testing the /dev/tty/USBx inputs
## https://stackoverflow.com/questions/39176985/how-to-pipe-data-from-dev-ttyusb0-to-a-python-script
## https://en.wikiversity.org/wiki/Python/Serial_port/pySerial
## https://pypi.org/project/pyserial/
## https://scikit-learn.org/stable/index.html

## https://stackoverflow.com/questions/16077912/python-serial-how-to-use-the-read-or-readline-function-to-read-more-than-1-char
##** https://github.com/microsoft/ML-For-Beginners

## https://realpython.com/python-ai-neural-network/
## https://realpython.com/python-zip-function/
## https://realpython.com/python-sum-function/

import os
import math
import csv
from random import *
import random
import pandas as pd
import numpy as np
import neural_network as nn
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import serial
import neural_network as nn

## This reads the serial data being streamed by the Arduino on port ttyUSB0

## Serial port Short setup
# ~ ser = serial.Serial('/dev/ttyUSB0', 9600)

## Serial port Long setup
ser = serial.Serial(
        # Serial Port to read the data from
        port='/dev/ttyUSB0',
 
        #Rate at which the information is shared to the communication channel
        baudrate = 9600,
   
        #Applying Parity Checking (none in this case)
        parity=serial.PARITY_NONE,
 
       # Pattern of Bits to be read
        stopbits=serial.STOPBITS_ONE,
     
        # Total number of bits to be read
        bytesize=serial.EIGHTBITS,
 
        # Number of serial commands to accept before timing out
        timeout=1
)


bias =	np.array([-0.5]) ## DBTEST
error  = 0
count = 0	

### DEPRECATED: hard coding the weights_1 vector 
### weights_1 is now initialized by init_weights(num)
# ~ weights_1 =	np.array([[1.01, -0.01],
					# ~ [1.01, -0.01],
					# ~ [1.01, -0.01],]) ## DBTEST
	
################################################################
### Extracts CO2 value from serial.Serial('/dev/ttyUSB0', 9600) 
### serial data stream object using readline().decode('ascii')
### SAVE: This search and parsing algorithm is VERY useful!! ###
### Switch between return and yield: return == True, yield == False
  
def get_CO2_value(ret = False):
	while True:
		global cc
		cc = str("")
		data = ser.readline().decode('ascii')
		## ~ print("data = {0}".format(data)) ## DBPRINT
		if ("CO2" in data):
			cc=str(data)
			# ~ print(cc[11:][:-5]) ## DBPRINT
			result = cc[11:][:-5]
			# ~ print(data) ## DBPRINT
			result = int(result)
			print() ## DBPRINT
			print(result) ## DBPRINT
			
			### Switch between return and yield as needed
			if(ret): return (result)
			else: yield (result)
			
		# ~ if data:
			# ~ print(data) ## DBPRINT
#####################################################################

### Initializing vectors and arrays ###	

def init_iv_and_targets(num):
	global input_vectors
	input_vectors = np.array([[0,1.1]])		
	global targets
	targets = np.array([1])
	
	for j in range(num-1):
		## initialize input_vectors with num*[0,0]		
		input_vectors = np.append(input_vectors,[[0,1.1]], axis = 0)
		if (j == (num-2)): print ("input_vectors = {0}".format(input_vectors)) ## DBPRINT

		## initialize targets with num*[1]		
		targets = np.append(targets,[1])
		if (j == (num-2)): print ("\ntargets = {0}".format(targets)) ## DBPRINT
	return input_vectors, targets

def init_pred_and_errors(num):
	global pred
	pred = np.array([[],])		
	global errors_array
	errors_array = np.array([],)
	
	for j in range(num-1):
		## initialize predictions with num*[0]		
		pred = np.append(pred,[[]])
		if (j == (num-2)): print ("pred = {0}".format(pred)) ## DBPRINT

		## initialize errors with num*[1]		
		errors_array = np.append(errors_array,[])
		if (j == (num-2)): print ("errors = {0}".format(errors_array)) ## DBPRINT
	return pred, errors_array
	
def init_weights(num):
	global weights_1
	weights_1 = np.array([[0.95,-0.1],])
	
	for j in range(num-1):
		## initialize weights with num*[0]		
		weights_1 = np.append(weights_1,[[0.95,-0.1]], axis = 0)
		if (j == (num-2)): print ("weights_1 = {0}".format(weights_1)) ## DBPRINT

	return weights_1
		
def init_dot_products(num):
	global dot_product_1
	dot_product_1 = np.array([[0.0],])
	
	for j in range(num-1):
		## initialize weights with num*[0]		
		dot_product_1 = np.append(dot_product_1,[[0.0]], axis = 0)
		if (j == (num-2)): print ("dot_product_1 = {0}".format(dot_product_1)) ## DBPRINT

	return dot_product_1
#####################################################################

### Displaying and showing data plots ###	
	
def display_all():	
	# ~ print("\n##############<<<<<~>>>>>##############") ## DBPRINT
	# ~ print("\nCurrent value of all network vectors and arrays\n") ## DBPRINT
	print ("input_vectors = {0}".format(input_vectors))
	print ("\ntargets = {0}".format(targets))
	print ("weights_1 = {0}".format(weights_1))
	print("\nbias:\t{0}\n".format(bias))	
	# ~ print ("predictions = {0}".format(pred))
	# ~ print ("errors = {0}".format(errors_array))
	print ("dot_product_1 = {0}".format(dot_product_1))
	print("\n##############<<<<<~>>>>>###############\n") ## DBPRINT
		
def show_all(train_error):	
	plt.plot(train_error)
	plt.xlabel("Iterations")
	plt.ylabel("Error for all training instances")
	plt.savefig("cumulative_error.png")
	plt.show()
	

########################################

### Computing dot products ###	
			
def computeDotProduct_1(iv, wt):
    # Computing the dot product of input_vector and weights_1
	dot_product_1 = np.dot(iv, wt)
	print("dot_product_1: \t\t{0}".format(dot_product_1))
	return dot_product_1
			
def dot_product(x_vector, y_vector):
    # Computing the dot product of input_vector and weights_1
	if len(x_vector) != len(y_vector):
		raise ValueError("Vectors must have equal sizes")
	dot_product_1 = sum(x * y for x, y in zip(x_vector, y_vector))
	print("dot_product: \t\t{0}".format(dot_product_1))
	return dot_product_1
	
def computeDotProduct_array(num, iv, wt):
    # Computing the dot product of input_vector[] and weights_1[]
	for i in range(num):				
		dot_product_array = np.dot(iv[i], wt[i])
		print("dot_product_array: \t\t{0}".format(dot_product_array))
	return dot_product_array
		
#######################################
### https://numpy.org/doc/stable/reference/generated/numpy.append.html

### run_multiple_tests: Run tests with multiple input vectors ###
### This ties everything together, and will be used as the MAIN function call.

def run_multiple_tests(num, learning_rate = 0.01, iterations = 1000, show = False):
	
	print("\n##############<<<<<~>>>>>##############") ## DBPRINT
	print("\nInitializing network vectors and arrays\n") ## DBPRINT
	init_iv_and_targets(num)	
	print()
	init_weights(num)	
	print("\nbias:\t{0}\n".format(bias))
	pred, errors_array = init_pred_and_errors(num)
	init_dot_products(num)
	obj = get_CO2_value(False)
	neural_network = nn.NeuralNetwork(learning_rate)
	print("\n##############<<<<<~>>>>>###############\n") ## DBPRINT
	
	for i in range(num):
		# ~ input_vectors[i] = [next(obj), 0] ## DBTEST
		input_vectors[i][0] = next(obj)
		print("iv = {0}:{1}".format(input_vectors[i][0], input_vectors[i][1])) ## DBPRINT
		
		dot_product_1[i] = dot_product(input_vectors[i], weights_1[i])
		prediction = nn.make_prediction(input_vectors[i], weights_1[i], bias)
		pred = np.insert(pred, i, prediction)
		print ("pred = {0}".format(pred))
		# ~ print(f"The prediction result is: {prediction}") ## DBPRINT
		err = nn.get_Mean_squared_error(prediction, targets[i])
		errors_array = np.insert(errors_array, i, err)
		print ("errors_array = {0}".format(errors_array))

		prediction = nn.update_prediction(input_vectors[i], 
						nn.update_weights(weights_1[i], 
						nn.get_derivative(prediction, targets[i])), 
						bias)
		derror_dprediction = 2 * (prediction - targets[i])
		layer_1 = np.dot(input_vectors[i], weights_1[i]) + bias
		dprediction_dlayer1 = nn.sigmoid_deriv(layer_1)
		dlayer1_dbias = 1

		derror_dbias = nn.get_derror_dbias(derror_dprediction,dprediction_dlayer1,dlayer1_dbias)

	# ~ print("\nInput Vectors = {0}".format(input_vectors)) ## DBPRINT		
	print ("\npred = {0}\n".format(pred))	
	print ("errors_array = {0}\n".format(errors_array))
	display_all()
	
	print("\nEntering the class")
	
	global tr_error
	tr_error = neural_network.train(input_vectors[i], targets, iterations)
	print("tr_error = {0}\n".format(tr_error)) ## DBPRINT
	
	# ~ display_all()
	if (show): show_all(tr_error)
	
	### DEPRECATED: calling the plt.plot options 
	### All plt.plot functions below are called by show_all(tr_error)

	## ~ plt.plot(tr_error)
	## ~ plt.xlabel("Iterations")
	## ~ plt.ylabel("Error for all training instances")
	## ~ plt.savefig("cumulative_error.png")
	## ~ plt.show()
		

############################################
### Neural Network, ML, and AI functions ###
### https://www.w3schools.com/python/python_ml_standard_deviation.asp

def get_mean(my_array):
	mean = np.mean(my_array)
	print("mean = {0}".format(mean))
	return mean

def get_median(my_array):
	mean = np.median(my_array)
	print("median = {0}".format(median))
	return median

def get_mode(my_array):	## get most common number
	mean = stats.mode(my_array)
	print("mode = {0}".format(mode))
	return mode

def get_stdev(my_array):	## get standard deviation
	stdev = np.std(my_array)
	print("std = {0}".format(stdev))
	return stdev

def get_first_indexes_mult(iv_0, wt_0):
    # ~ first_indexes_mult = input_vector[0] * weights_1[0]
    first_indexes_mult = iv_0 * wt_0
    print("first_indexes_mult\t{0:.3}".format(first_indexes_mult))
    return first_indexes_mult
    
def get_second_indexes_mult(iv_1, wt_1):
    # ~ second_indexes_mult = input_vector[1] * weights_1[1]
    second_indexes_mult = iv_1 * wt_1
    print("second_indexes_mult\t{0:.3f}".format(second_indexes_mult))
    return second_indexes_mult
 
##################################################

#####################
### START OF MAIN ###	
#####################

# Driver program
if __name__ == "__main__":
    
########################################
### Test with multiple input vectors ###
### run_multiple_tests(num, learning_rate = 0.01, iterations = 1000, show = False)

	run_multiple_tests(3, 0.01, 1000, False)	

#####################################
### Test with single input vector ###	
	
	# ~ test_vector = np.array([[0,0]])
	
	# ~ computeDotProduct_1()    
	# ~ computeDotProduct_2()  
	# ~ prediction = nn.make_prediction(input_vector, weights_1, bias)
	# ~ print(f"The prediction result is: {prediction}")
	# ~ target = 1
	# ~ nn.get_Mean_squared_error(prediction, target)

	# ~ ## ~ prediction = nn.update_prediction(input_vector, nn.update_weights(weights_1, nn.get_derivative(prediction, target)), bias)
	# ~ derror_dprediction = 2 * (prediction - target)
	# ~ layer_1 = np.dot(input_vector, weights_1) + bias
	# ~ dprediction_dlayer1 = nn.sigmoid_deriv(layer_1)
	# ~ dlayer1_dbias = 1

	# ~ derror_dbias = nn.get_derror_dbias(derror_dprediction,dprediction_dlayer1,dlayer1_dbias)



	# ~ print("\nEntering the class")
	# ~ learning_rate = 0.01
	# ~ neural_network = NeuralNetwork(learning_rate)
	# ~ neural_network.predict(input_vector)
	# ~ training_error = neural_network.train(input_vectors, targets, 10)
	# ~ plt.plot(training_error)
	# ~ plt.xlabel("Iterations")
	# ~ plt.ylabel("Error for all training instances")
	# ~ plt.savefig("cumulative_error.png")
    # ~ plt.show()

###################
### END OF MAIN ###	
###################


### SAVE THIS: Very Useful!! ###
   
'''
while True:
	data = ser.readline().decode('ascii')
	## ~ print("data = {0}".format(data))
	if ("CO2" in data):
	## ~ if data.decode() == "CO2":
		# ~ cc=str(ser.readline())
		cc=str(data)
		print(cc[11:][:-5])
		# ~ print(data)
	
	# ~ if data:
		# ~ print(data)
		
	# ~ data = ser.read(1)
	# ~ if data.decode() == "CO2":
		# ~ if data:
			# ~ print("CO2")
			# ~ ## ~ print(data)
'''
	

### SAVE THIS: full serial port setup snippet ###
# ~ ser = serial.Serial(
        # ~ # Serial Port to read the data from
        # ~ port='/dev/ttyUSB0',
 
        # ~ #Rate at which the information is shared to the communication channel
        # ~ baudrate = 9600,
   
        # ~ #Applying Parity Checking (none in this case)
        # ~ parity=serial.PARITY_NONE,
 
       # ~ # Pattern of Bits to be read
        # ~ stopbits=serial.STOPBITS_ONE,
     
        # ~ # Total number of bits to be read
        # ~ bytesize=serial.EIGHTBITS,
 
        # ~ # Number of serial commands to accept before timing out
        # ~ timeout=1
# ~ )

### SAVE THIS: Parser Example snippet ###
# ~ while True:
	# ~ global cc
	# ~ cc = str("")
	# ~ data = ser.readline().decode('ascii')	
	# ~ cc=str(ser.readline())
	# ~ print(cc[2:][:-5])
	
	# ~ if ("CO2" in data):
			# ~ cc=str(data)
			# ~ ## ~ print(cc[11:][:-5]) ## DBPRINT
			# ~ result = cc[11:][:-5]
			# ~ ## ~ print(data) ## DBPRINT
			# ~ result = int(result)
			# ~ print(result) ## DBPRINT
			
			# ~ ### Switch between return and yield as needed
			# ~ ## return (result)
			# ~ yield (result)
			
### END OF SAVE THIS ###
