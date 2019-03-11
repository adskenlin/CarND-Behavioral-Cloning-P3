# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)


[image1]: ./examples/center.jpg "normal"
[image2]: ./examples/recovery1.png "Recovery Image"
[image3]: ./examples/recovery2.png "Recovery Image"
[image4]: ./examples/center.jpg "Normal Image"
[image5]: ./examples/fliped.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* run1 containing the output images to record a video
* run1.mp4 recording the output of model as video

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5 run1
python video.py run1 --48FPS
```
and output a video with 48FPS.
#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My neural network model consists of a normalization layer(model.py line 88), a cropping layer(model.py line 90), 3 convolutional layers with 5x5 filter sizes and depths between 24 and 48, following valid pooling with (5,5) strides(model.py lines 92-94). Afterwards, the layers are connecting with convolution layers with 2 convolutional layers with 3x3 filter sizes and 64 depths.'ReLu' as activation function is used to introduce nonlinearity and added to every single Conv2D layer. Then nn will be flattened and add the fully-connected layers with the node numbers 1164, 100, 50, 10. Finally, get 1 output. 


#### 2. Attempts to reduce overfitting in the model
The methods I used to reduce the overfitting are collecting more data and data augmentation.
The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 110). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer with default values, so the learning rate was not tuned manually (model.py line 106).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, clock-wise driving(against former driving direction).

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to get a better cloning behavior for our cars, with reducing the underfitting and 'mse'(mean squared errors between predicton and ground truth)

At the beginning I build a very simple neural network to test my code and debug using the sampling driving data(got both poor training and validation loss). When I switch to my own dataset, it turned to be much underfitting.

My second step was to use a convolution neural network model similar to the model of nVida end2end deep learning CNN. I thought this model might be appropriate because it is enough deep and will have a good performance for large dataset(this time I uesed my own data). In order to gauge how well the model was working, I split my image and steering angle data into a 80% training and 20% validation set. At the end of training, the mse was in a so-so level but my car drived at a silly way and got off road. I found that the model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

I noticed that my dataset didn't contain enough data: The recovery data and the enough data to turn right(because in simulator the car drives anticlockwisely). To combat the overfitting, I used the simulator to collect more data, especially the data of recovery behavior and anticlockwise driving.

Then I got the acceptable low value of mean square error of training set and validation error. 

At the end of the process, the vehicle was able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:
```sh
# normalizing
model.add(Lambda(lambda x: x/127.5-1., input_shape = (row,col,ch), output_shape=(row,col,ch)))
# cropping
model.add(Cropping2D(cropping=((70,25), (0,0))))
# convolution
model.add(Conv2D(24,(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(36,(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(48,(5,5),strides=(2,2),activation="relu"))
model.add(Conv2D(64,(3,3),activation="relu"))
model.add(Conv2D(64,(3,3),activation="relu"))
# fully-connect
model.add(Flatten())
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
#output
model.add(Dense(1))
```

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to get back to the center when it is going to driving off the road. These images show what a recovery looks like starting from ... :
![alt text][image2]
![alt text][image3]
and get back to lane center.

Then I repeated a lap with anticlockwise driving to get more data points and make car learn to solve the right turn problem.

To augment the data sat, I also flipped images and angles thinking that this would collect more data and also benifit the right turn problem. For example, here is an image that has then been flipped:

![alt text][image4]
![alt text][image5]

After the collection process, I had 10007 number of data points. I then preprocessed this data by normalization and cropping layers at the beginning of neural network.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 as evidenced by watching when validation loss varies stable and is already low. I used an adam optimizer so that manually training the learning rate wasn't necessary.
