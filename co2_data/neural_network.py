## https://realpython.com/python-ai-neural-network/
## https://realpython.com/gradient-descent-algorithm-python/
## https://discovery.cs.illinois.edu/learn/Simulation-and-Distributions/Law-of-Large-Numbers/
## https://www.datacamp.com/tutorial/multilayer-perceptrons-in-machine-learning

import os
import csv
import math
from random import *
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

##########################################
##########################################
class NeuralNetwork:
    def __init__(self, learning_rate):
        self.weights = np.array([np.random.randn(), np.random.randn()])
        self.bias = np.random.randn()
        self.learning_rate = learning_rate

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_deriv(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def predict(self, input_vector):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2
        print("class prediction = {0}".format(prediction))
        return prediction

    def _compute_gradients(self, input_vector, target):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2

        derror_dprediction = 2 * (prediction - target)
        dprediction_dlayer1 = self._sigmoid_deriv(layer_1)
        dlayer1_dbias = 1
        dlayer1_dweights = (0 * self.weights) + (1 * input_vector)

        derror_dbias = (derror_dprediction * dprediction_dlayer1 * dlayer1_dbias)
        derror_dweights = (derror_dprediction * dprediction_dlayer1 * dlayer1_dweights)
		
        # ~ print("GRADIENTS:\tderror_dbias = {0}\tderror_dweights {1}".format(derror_dbias, derror_dweights))
        return derror_dbias, derror_dweights

    def _update_parameters(self, derror_dbias, derror_dweights):
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - (derror_dweights * self.learning_rate)
        
    def train(self, input_vectors, targets, iterations):
        cumulative_errors = []
        for current_iteration in range(iterations):			
            # Pick a data instance at random
            random_data_index = np.random.randint(len(input_vectors))
            input_vector = input_vectors[random_data_index]
            target = targets[random_data_index]
            # Compute the gradients and update the weights
            derror_dbias, derror_dweights = self._compute_gradients(
                input_vector, target)
            self._update_parameters(derror_dbias, derror_dweights)

            # Measure the cumulative error for all the instances
            if current_iteration % 100 == 0:
                cumulative_error = 0
                # Loop through all the instances to measure the error
                for data_instance_index in range(len(input_vectors)):
                    data_point = input_vectors[data_instance_index]
                    target = targets[data_instance_index]

                    prediction = self.predict(data_point)
                    error = np.square(prediction - target)

                    cumulative_error = cumulative_error + error
                cumulative_errors.append(cumulative_error)
                # ~ print("cumulative_errors = {0}".format(cumulative_errors)) ## DBPRINT

        return cumulative_errors


def get_first_indexes_mult(iv_0, wt_0):
    # ~ first_indexes_mult = input_vector[0] * weights_1[0] ## SAVE
    first_indexes_mult = iv_0 * wt_0
    print("first_indexes_mult\t{0}".format(first_indexes_mult))
    return first_indexes_mult
    
def get_second_indexes_mult(iv_1, wt_1):
    # ~ second_indexes_mult = input_vector[1] * weights_1[1] ## SAVE
    second_indexes_mult = iv_1 * wt_1
    print("second_indexes_mult\t{0}".format(second_indexes_mult))
    return second_indexes_mult
    
def computeDotProduct_1():
    # Computing the dot product of input_vector and weights_1
    get_first_indexes_mult(input_vectors[0], weights_1[0]) ## SAVE
    # ~ get_second_indexes_mult(input_vectors[1], weights_1[1]) ## SAVE
    
    # ~ dot_product_1 = (get_first_indexes_mult(input_vector[0], weights_1[0]) +
					# ~ get_second_indexes_mult(input_vector[1], weights_1[1])) ## SAVE
					
    dot_product_1 = np.dot(input_vectors, weights_1)
    print(f"dot_product1: \t\t{dot_product_1}")
    # ~ print("dot_product_1\t", dot_product_1)
    return dot_product_1
    
def computeDotProduct_2():
    # Computing the dot product of input_vector and weights_2
    get_first_indexes_mult(input_vectors[0], weights_2[0]) ## SAVE
    # ~ get_second_indexes_mult(input_vectors[1], weights_2[1]) ## SAVE
    
    # ~ dot_product_2 = (get_first_indexes_mult(input_vector[0], weights_2[0]) +
					# ~ get_second_indexes_mult(input_vector[1], weights_2[1])) ## SAVE
					
    dot_product_2 = np.dot(input_vectors, weights_2)
    print("dot_product_2\t\t{0}".format(dot_product_2))
    return dot_product_2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def make_prediction(input_vector, weights, bias):
    layer_1 = np.dot(input_vector, weights) + bias
    layer_2 = sigmoid(layer_1)
    return layer_2

def get_Mean_squared_error(prediction, target):
    mse = np.square(prediction - target)
    print(f"Prediction: {prediction}; Error: {mse}")
    return mse

def get_derivative(prediction, target):
    derivative = 2 * (prediction - target)
    print(f"derivative: {derivative}")
    return derivative

def update_weights(weights, derivative):
    weights_1 = weights - derivative
    return weights_1

def update_prediction(input_vector, weights_1, bias):
    prediction = make_prediction(input_vector, weights_1, bias)
    error = (prediction - targets) ** 2
    print(f"New Prediction: {prediction}; Error: {error}")
    return prediction

def sigmoid_deriv(x):
    return sigmoid(x) * (1-sigmoid(x))

## NOTE: derror/dbias = (derror/dprediction) * (dprediction/dlayer) * (dlayer/dbias)
## everything cancels down to:	derror/dbias

def get_derror_dbias(derror_dprediction,dprediction_dlayer1,dlayer1_dbias):	
    derror_dbias = (derror_dprediction * dprediction_dlayer1 * dlayer1_dbias)
    print("derror_dbias\t{0}".format(derror_dbias))
    return derror_dbias

####################################################
############ Example data from tutorial ############
### https://realpython.com/python-ai-neural-network/
####################################################
# EXAMPLE: Wrapping the vectors in NumPy arrays
## ~ input_vector = np.array([1.66, 1.56]) ## DBTEST

# ~ input_vector = np.array([2, 1.5]) ## DBTEST
weights_1 =	np.array([1.45, -0.66]) ## DBTEST
## ~ input_vector = [1.72, 1.23] ## DBTEST
## ~ weights_1 = [1.26, 0] ## DBTEST
weights_2 = np.array([2.17, 0.32]) ## DBTEST
bias =	np.array([0.0]) ## DBTEST
error  = 0
input_vectors = np.array([[3, 1.5],])

# ~ input_vectors = np.array([[3, 1.5],
                # ~ [2, 1],
                # ~ [4, 1.5],
                # ~ [3, 4],
                # ~ [3.5, 0.5],
                # ~ [2, 0.5],
                # ~ [5.5, 1],
                # ~ [1, 1],])

targets = np.array([0, 1, 0, 1, 0, 1, 1, 0]) ## DBTEST

##########################################

if __name__ == '__main__':   
    computeDotProduct_1()    
    computeDotProduct_2()  
    prediction = make_prediction(input_vectors, weights_1, bias)
    print(f"The prediction result is: {prediction}")
    target = 0
    get_Mean_squared_error(prediction, target)
    
    prediction = update_prediction(input_vectors, update_weights(weights_1, get_derivative(prediction, target)), bias)
    derror_dprediction = 2 * (prediction - target)
    layer_1 = np.dot(input_vectors, weights_1) + bias
    dprediction_dlayer1 = sigmoid_deriv(layer_1)
    dlayer1_dbias = 1
    
    derror_dbias = get_derror_dbias(derror_dprediction,dprediction_dlayer1,dlayer1_dbias)
    
    print("\nEntering the class")
    learning_rate = 0.01
    neural_network = NeuralNetwork(learning_rate)
    neural_network.predict(input_vectors)
    training_error = neural_network.train(input_vectors, targets, 1000)
    plt.plot(training_error)
    plt.xlabel("Iterations")
    plt.ylabel("Error for all training instances")
    plt.savefig("cumulative_error.png")
    # ~ plt.show()
    
## Out[8]: The dot product is: 2.1672
