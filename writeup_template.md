# Term 1 Project 3 - Behavioral Cloning


## Project Overview

The objective of this project is to clone human driving behavior based on the current using a Deep Neural Network. 
 In order to achieve this, we are using a Car Simulator provided by Udacity, which has 2 modes Training and Autonomous.
During the training phase, we navigate our car inside the simulator using the keyboard. 
While we navigating the car the simulator records training images and respective steering angles. 
The simulated car has cameras in 3 different angles namely : left, center and right. 
We use all these images to train out model and make the car cross a lap on its own.
Then we use those recorded data to train our neural network. Trained model was tested on two tracks, namely training track and validation track. 


---

## Project Phases

Steps Involved

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality


####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py  - containing the script to create and train the model
* drive.py  - for driving the car in autonomous mode
* model.h5  - containing a trained convolution neural network 
* writeup_template.md/readme.md  - summarizing the results

####2. Submission includes functional code


### model.py
### drive.py
### model.h5
### writeup_report
### video.mp4



####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

Check Arc.png in images.

![image] (https://github.com/riaz/CarND-Behavioral-Cloning-P3/blob/master/images/Arc.png)

====================================================================================================


####1. An appropriate model architecture has been employed

We used the NVIDIA End-to-End Learning Architecture.



####2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. 
I used a combination of center lane driving, recovering from the left and right sides of the road.
I have also creating training data by driving the car in the opposite direction in both the tracks provided in the simulator.


###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture


We used the NVIDIA End-to-End Learning Architecture.


![alt text][https://github.com/riaz/CarND-Behavioral-Cloning-P3/blob/master/images/Arc.png]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

