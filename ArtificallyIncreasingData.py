import os
from os.path import isfile,join
import cv2
import numpy as np 
import scipy.io as sio

""" Caltech dataset has around 8677 images for 101 categories which is pretty less. In order to avoid
overfitting of data and to get more accuracy its a good idea to artifically increase the size of data
by translating and rotationg each image.

In this file it can be seen that each image is transformed 10 times resulting in dataset size of 
86770 which is very high compared to original dataset size
 """

def transformation():

	os.chdir('/Users/manaswipodduturi/Documents/Research/MachineLearning/NeuralNets/Caltech')
#load features and labels stored in mat file 
	data = sio.loadmat('caltech_data.mat')
	labels=data['labels']
	labels=np.transpose(labels)
	labels = np.reshape(labels,(labels.shape[0],))
	training_x = data['features']
	dstImages = np.array(())
	all_images = np.array((training_x.shape[0]*10,15000),dtype=np.uint8)
	all_labels = np.array((training_x.shape[0]*10,1),dtype=np.uint8)
	cols=100
	rows = 150

	for i in xrange(0,training_x.shape[0]):
#translating image by x:-5  y:0
		M = np.float32([[1,0,-5],[0,1,0]])
		dst = cv2.warpAffine(np.uint8(training_x[i].reshape(150,100)),M,(cols,rows))
		all_images[10*i+0] = np.ravel(dst)
#translating image by x:0  y:5
		M = np.float32([[1,0,0],[0,1,5]])
		dst = cv2.warpAffine(np.uint8(training_x[i].reshape(150,100)),M,(cols,rows))
		all_images[10*i+1] = np.ravel(dst)		
#translating image by x:-5  y:5
		M = np.float32([[1,0,-5],[0,1,5]])
		dst = cv2.warpAffine(np.uint8(training_x[i].reshape(150,100)),M,(cols,rows))
		all_images[10*i+2] = np.ravel(dst)
#translating image by x:5  y:-5
		M = np.float32([[1,0,5],[0,1,-5]])
		dst = cv2.warpAffine(np.uint8(training_x[i].reshape(150,100)),M,(cols,rows))
		all_images[10*i+3] = np.ravel(dst)
#translating image by x:5  y:5
		M = np.float32([[1,0,5],[0,1,5]])
		dst = cv2.warpAffine(np.uint8(training_x[i].reshape(150,100)),M,(cols,rows))
		all_images[10*i+4] = np.ravel(dst)
#Rotating image by -5 degrees
		M = cv2.getRotationMatrix2D((cols/2,rows/2),-5,1)
		dst = cv2.warpAffine(np.uint8(training_x[i].reshape(150,100)),M,(cols,rows))
		all_images[10*i+5] = np.ravel(dst)
#Rotating image by 5 degrees		
		M = cv2.getRotationMatrix2D((cols/2,rows/2),5,1)
		dst = cv2.warpAffine(np.uint8(training_x[i].reshape(150,100)),M,(cols,rows))
		all_images[10*i+6] = np.ravel(dst)
#Rotating image by 10 degrees		
		M = cv2.getRotationMatrix2D((cols/2,rows/2),-10,1)
		dst = cv2.warpAffine(np.uint8(training_x[i].reshape(150,100)),M,(cols,rows))
		all_images[10*i+7] = np.ravel(dst)	
#Rotating image by 10 degrees	
		M = cv2.getRotationMatrix2D((cols/2,rows/2),10,1)
		dst = cv2.warpAffine(np.uint8(training_x[i].reshape(150,100)),M,(cols,rows))
		all_images[10*i+8] = np.ravel(dst)	
#Image without any transformations
		dst = np.uint8(training_x[i].reshape(150,100))
		all_images[10*i+9] = np.ravel(dst)	

		all_labels[10*i+0:10*i+10]=labels[i]

	sio.savemat('/Users/manaswipodduturi/Documents/Research/MachineLearning/NeuralNets/caltech_data_increased.mat',
        {'features':all_images,
         'labels' : all_labels})


	