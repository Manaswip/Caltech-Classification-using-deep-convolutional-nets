<html>
<body>
<h2> Caltech-101 dataset classification using deep convolutional networks </h2>
Caltech 101 dataset contains images belonging to 101 different categories (for example camera,dolphin,elephant,faces, etc.). 
Images in this dataset have backgrounds, in order to crop region of interest from the image, preprocessing steps are performed
on the dataset using OpenCV. In each category there are 50 images on an average. </br>

Caltech dataset has around 8677 images for 101 categories which is pretty less. In order to avoid
overfitting of data and to get more accuracy its a good practice to artifically increase the size of data
by translating and rotationg each image.In this project each image is transformed 10 times resulting in dataset size of 
86770 which is very high compared to original dataset size.
</br>
Code supports different kind of layer types( Full connected layer, 
convoluations layer, max pooling layer, softmax layer) and different activation functions (sigmoid, rectified linear units,
tanh)

Code is built using Theano library so, this code can be run either on CPU or GPU, set GPU to true to run on GPU and set GPU to false to run on CPU

This program incorparates ideas and code from text book on <a href='http://neuralnetworksanddeeplearning.com/index.html'> Neural Networks and Deep learning from Michael Nielsen </a> and <a href='https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src'>Michael Nielsen's github</a> 

<h3> Project Layout </h3>
<p>
<b>ArtificallyIncreasingData.py:</b> In this program each image is transformed 10 times resulting in dataset size of 
86770 </br>
<b>Caltechraining_Theano.py:</b> Implementation of deep convolutional networks using theano giving an advantage of running the code either on CPU/GPU. In addition to that this code supports different cost functions and activation functions </br>
<b>CropImagesWithAnnotations.py:</b> In this program the foreground/area of intereset in the image is cropped using annotations provided along with dataset. All the cropped images are written to a folder and also array values of every image is appended in array and stored in .mat file in order to avoid converting image to array every time we run the program </br>
<b>GeneratingTestData.py:</b> 10 percent of images are selected as testing data from each category randomly and stored in mat file </br>
<b>LoadingTrainingTestValidationData.py:</b> Returns training data, testing data and validation data as a list
Validation data is used to choose suitable hyper parameters for the algorithm
<h3> Sample code to run the program </h3>
import cv2 </br> 
import Caltech </br> 
from Caltech import Network </br> 
from Caltech import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer </br> 
training_data, validation_data, test_data = Caltech.load_data_shared() </br> 
mini_batch_size = 10 </br> 
net = Network([ </br>
   &emsp;  &emsp;&emsp;&emsp;  ConvPoolLayer(image_shape=(mini_batch_size, 1, 150, 100), filter_shape=(20, 1, 5, 5),  </br>
    &emsp;  &emsp;&emsp;&emsp;                poolsize=(2, 2)), FullyConnectedLayer(n_in=20*73*48, n_out=50), </br>
     &emsp; &emsp;&emsp;&emsp;                SoftmaxLayer(n_in=50, n_out=10)], mini_batch_size) </br> 
net.SGD(training_data, 60, mini_batch_size, 0.1,validation_data, test_data)   </br>

</p>
</body>
</html>
