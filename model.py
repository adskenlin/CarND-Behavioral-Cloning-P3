import csv
from scipy import ndimage
import numpy as np
import cv2

# extract the samples from csv
samples = []
with open('/opt/Data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        samples.append(line)

# define a function to process the samples(parameter extracting, data augmentation)
def get_images_angles(samples):
    #placeholder for images and angles, which are extracted directly from csv sample
    car_images = []
    steering_angles = []
    for sample in samples:
        steering_center = float(sample[3])
        # add a correction to steering angle, can be used for corresponding images from left and right cameras
        correction = 0.2
        steering_left = steering_center + correction
        steering_right = steering_center - correction
        #extract the images of center, left, right cameras
        for i in range(3):
            source_path = sample[i]
            filename = source_path.split('/')[-1]
            current_path = '/opt/Data/IMG/'+ filename
            if i == 0:
                img_center = ndimage.imread(current_path)
            if i == 1:
                img_left = ndimage.imread(current_path)
            if i == 2:
                img_right = ndimage.imread(current_path)
        # append the images
        car_images.append(img_center)
        car_images.append(img_left)
        car_images.append(img_right)
        # append the angles
        steering_angles.append(steering_center)
        steering_angles.append(steering_left)
        steering_angles.append(steering_right)
    #data augmentation    
    augmented_images, augmented_steering_angles = [], []
    for image, steering_angle in zip(car_images, steering_angles):
        augmented_images.append(image)
        augmented_steering_angles.append(steering_angle)
        # flip the image and add to data
        augmented_images.append(cv2.flip(image,1))
        augmented_steering_angles.append(steering_angle*(-1.0))
    return augmented_images, augmented_steering_angles
    
# split the dataset to training set and validation set
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import sklearn
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from sklearn.utils import shuffle
#build a generator, make the neural network work better with large data
def generator(samples,batch_size=128):
    num_samples = len(samples)
    while True:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
#            for batch__sample in batch_samples:
            images, angles = get_images_angles(batch_samples)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)
# build up train generator and validation generator
train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)
# size of input image
row, col, ch = 160, 320, 3
# build the network
model = Sequential()
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
# compile the model
model.compile(loss='mse', optimizer='adam')
# set up batch_size
batch_size = 128
# fit the model
model.fit_generator(train_generator, steps_per_epoch = len(train_samples)/batch_size, 
                                     validation_data = validation_generator, 
                                     validation_steps = len(validation_samples)/batch_size,
                                     epochs=2,verbose=1)
# save the model
model.save('model.h5')
'''
import matplotlib.pyplot as plt

print(history_object.history.keys())
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc = 'upper right')
plt.show()
'''