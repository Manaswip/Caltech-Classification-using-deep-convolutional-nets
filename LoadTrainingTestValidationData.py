import scipy.io as sio
import os
import numpy as np

""" Return training data, testing data and validation data as a list
Validation data is used to choose suitable hyper parameters for the algorithm
"""

def load_data():

	os.chdir('/Users/manaswipodduturi/Documents/Research/MachineLearning/NeuralNets/Caltech')
	data = sio.loadmat('caltech_features.mat')
	input_features = data['input_features']
	input_labels = data['input_labels']
	test_features = data['test_features']
	test_labels = data['test_labels']
	input_labels = np.transpose(input_labels)
	test_labels = np.transpose(test_labels)
	test_labels = np.reshape(test_labels,(test_labels.shape[0],))
	input_labels = np.reshape(input_labels,(input_labels.shape[0],))

	training_data = (input_features,input_labels)
	testing_data = (test_features,test_labels)
	validation_data = (test_features,test_labels)
	return training_data,testing_data,validation_data

