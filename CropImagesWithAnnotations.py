import cv2
import scipy.io as sio
from os import listdir
from os.path import isfile,join,isdir
import numpy as np
from matplotlib.path import Path
import os

""" Caltech-101 dataset has images belonging to 101 different Categories. The dataset also provides us with annotation 
for each image which helps us to crop the foreground image and get rid of background in the image.

In this file the foreground/area of intereset in the image is cropped by ignoring the background
 and all the cropped out images are written to a folder and also values of each and every image is appended
 in array and stored in .mat file in order to avoid converting image to array every time we run the program
"""
def load_data_shared():

# Navigate to  main directory in which there are 101 subdirectories containing images.
    mypath = '/Users/manaswipodduturi/Documents/Research/MachineLearning/Data/Caltech/101_ObjectCategories'
# Process all the folders in the main directory
    folders = [folder for folder in listdir(mypath) if isdir(join(mypath,folder))]
    all_images = np.array([])
    all_labels = np.array([],dtype=np.uint8)
# Process all the images in each and every folder in the main directory.
    for j in xrange(0,len(folders)):
 	  files = [file for file in listdir(join(mypath,folders[j])) if (isfile(join(mypath,folders[j],file)) & bool(file!='.DS_Store')) ]
 	  images = np.empty(len(files),dtype=object)
 	  labels = np.empty(len(files),dtype=np.uint8)
#Convert each and every image to array and append array values of all the images
 	  for n in xrange(0,len(files)):
 	  	   images[n] = cv2.imread(join(mypath,folders[j],files[n]))
 	  	   labels.fill(j)
 	  all_images = np.append(all_images,images)
 	  all_labels = np.append(all_labels,labels)

# Navigate to  main directory in which there are 101 subdirectories containing annotation of each image.
    mypath = '/Users/manaswipodduturi/Documents/Research/MachineLearning/Data/Caltech/Annotations'
    folders = [folder for folder in listdir(mypath) if isdir(join(mypath,folder))]
    all_annotations = np.array([])
    for j in xrange(0,len(folders)):
        files = [file for file in listdir(join(mypath,folders[j])) if (isfile(join(mypath,folders[j],file)) & bool(file!='.DS_Store'))]
        annotations = np.empty(len(files),dtype=object)
# Get array of annotations for each image from .mat file and append annotations of all the images
        for n in xrange(0,len(files)):
            annotations[n] = sio.loadmat(join(mypath,folders[j],files[n]))
        all_annotations = np.append(all_annotations,annotations)

# Navigate to  directory where you want to write cropped images.
    os.chdir('/Users/manaswipodduturi/Documents/Research/MachineLearning/Data/Caltech/processed_images1')
    image_with_annotation = np.empty((all_images.shape[0],150*100),dtype=object)
    for i in xrange(0,all_images.shape[0]):
        image = all_images[i]
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# Contour is a rough coordinates, which help us find the area of interest in the image
        contour = all_annotations[i]['obj_contour']
#Box coordinates provides us with rough rectangular outline of our area of interest in the image
        box = all_annotations[i]['box_coord']
        contour = np.transpose(contour)
        contour[:,0]=contour[:,0]+box[:,2]
        contour[:,1]=contour[:,1]+box[:,0]
# Outline/sketch the ares of interest in the image and find all the points inside that region
        p= Path(contour[:-2])
        mask = np.zeros((image.shape),dtype=np.uint8)
        for y in xrange(box[0,0],box[0,1]+1):
            for x in xrange(box[0,2],box[0,3]+1):
                bool_value = p.contains_point([x,y])
                if bool_value:
                    mask[y-1,x-1]=1
        image = image*mask
#Write crooped out image to the subfolder
        #cv2.imwrite('image'+str(i+1)+'.jpg',image)
#append value of each image to an array
        image = cv2.resize(image, (100,150))
        image_with_annotation[i] = np.ravel(image)
    sio.savemat('/Users/manaswipodduturi/Documents/Research/MachineLearning/NeuralNets/Caltech/caltech_data.mat',
        {'features':image_with_annotation,
         'labels' : all_labels})        

    


