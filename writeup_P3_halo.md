# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output/distribution.png "Visualization"
[image2]: ./output/grayscale.png "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./output/test_images.png "test images"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distribution.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because most sign in the dataset is only related to the shape(no trafic sign).

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because this operation avoids the loss results becoming to large or to small.

I decided to generate additional data because the data number of each sign is not even as first image.
To add more data to the the data set, I used the following techniques because it can generate more fake but useful dataset.
*random noise
*image rotation(within 15 degree)
*image moving(lateral or vertial)
 >note:When I generated the fake data, I only use the random noise technique, the rotation and moving is appled when training process in function **process_batch()**


Here is an example of an original image and an augmented image(random noise):

![alt text][image3]

The difference between the original data set and the augmented data set is adding some random points(noise) in the off-line dataset.
When training the model, I feed the model with other augmention data by on-line caculating.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					| 2x2 stride, same padding,						|
| batch_normalization   |                                               |
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 	      			|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |    	
| RELU					| 2x2 stride, same padding,						|
| batch_normalization   |                                               |			
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 	      			|
| Fully connected		| input:400.   output:120     					|
| drop out              | keep probability: 0.5                           |
| Fully connected		| input:120.   output:84    					|
| drop out              | keep probability: 0.5                         |
| Fully connected		| input:84.   output:43    				    	|
| Softmax				| transfer logits to probabilities				|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the parameters:
| parameter         		|     decription	        					| 
|:---------------------:|:---------------------------------------------:| 
|learing rate  | 0.001|
|final epochs  |150   |
|Batch size    | 256  |
|mu            |0     |
|sigma         |0.1   |

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.962
* validation set accuracy of 0.942
* test set accuracy of 0.936

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
  
*I choosed the classical lenet-5 architecture*
* What were some problems with the initial architecture?
  
*Although the model worked well, but the accuracy is only near to 0.86 with augmention image*
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

1. First, I add the drop-out layers between full connected layers, this didn't help a lot, 
this is beacuse that I fogot to set the keep_prob to 1.0 when validation
2. Then I add the batch_normalization layers after the ativation layers, this help the model impoving more steadly.
* Which parameters were tuned? How were they adjusted and why?
1. batch_size set to 256, this can take more data in memoary, which makes training process faster
2. keep_out set to 0.5 to avoid over fitting.
3. learning rate set to 0.0008 to make sure the learing rate is pretty small.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
1. adding the drop out layer to avoiding over fitting
2. batch_normalization layers after the ativation layers, this help the model impoving more steadly.

### Test a Model on New Images

#### 1. Choose six German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] 

THe label of these images are [7,30,18,26,25,24] which means:
|label| meanings|
|:---------------------:|:---------------------------------------------:| 
|7|Speed limit (100km/h)|
|30|Beware of ice/snow|
|18|General caution|
|26|traffic signs|
|25|Road work|
|24|Road narrows on the right|
For the fourth image, it's very blur to figure out by eyes. Also the traffic signs is not only related to one sign, so it may be hard to figure out.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road work	      		| Road work					 				|
|Traffic signals        | Priority road|
| General caution		| General caution								|
| Beware of ice/snow    | Beware of ice/snow  		                	|
| Speed limit (100km/h) |  Speed limit(100km/h)					            	| 
| Road narrows on the right		| Road narrows on the right     							|


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83.3%. This compares favorably to the accuracy on the test set of 0.936

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

The probability are pretty even, and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99100804e-01	      		| Road work					 				|
|9.99001563e-01      | Priority road|
| 9.91068482e-01		| General caution								|
| 9.69919443e-01    | Beware of ice/snow  		                	|
| 4.40172762e-01 |  Speed limit(100km/h)					            	| 
| 9.91590738e-01	| Road narrows on the right     							||


For the second image , I think the algorithm may be right. Because it's quite blur..

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


