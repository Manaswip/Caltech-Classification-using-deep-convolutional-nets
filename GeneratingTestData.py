import scipy.io as sio
import os
import numpy as np

"""  Selecting 10 percent of images as testing data from each category randomly. 
The mat file generated in this file is used when ever we run algorithm to train data set.
"""

def load_data():

#Navigate to the folder where artifically increased dataset mat file is located
	os.chdir('/Users/manaswipodduturi/Documents/Research/MachineLearning/NeuralNets/Caltech')
	data = sio.loadmat('caltech_data_increased.mat')
	labels=data['labels']
	labels = np.reshape(labels,(labels.shape[0],))
	training_x = data['features']
	test_images = np.empty((0,15000),dtype=np.uint8)
	test_labels = np.empty((0,1),dtype=np.uint8)

# Select 10 percent of images from each category and add to test_images and remove them from 
# training_images
	for i in xrange(0,101):
		where = np.argwhere(labels==i)
		size = where.shape[0]
		rand_num = size*10/100
		rand = np.arange(rand_num)
		random = where[rand]
		test_images = np.append(test_images,training_x[np.reshape(random,(random.shape[0],))],axis=0)
		test_labels = np.append(test_labels,labels[random])
		training_x=np.delete(training_x,random,0)
		labels=np.delete(labels,random,0)

# Save training images,labels and testing images, labels in separate mat file.
	sio.savemat('/Users/manaswipodduturi/Documents/Research/MachineLearning/NeuralNets/Caltech/caltech_features.mat',
        {'input_features':training_x,
          'input_labels' : labels,
          'test_features': test_images,
          'test_labels': test_labels})






