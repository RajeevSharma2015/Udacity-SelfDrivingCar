# / ************************************************************************ /
#  File Name :  clone.py
#  Author    :  Rajeev Kumar Sharma
#  Date      :  24/12/2017
#  Input     :  Data file created by autonomous driving simulator i.e 
#               (1) driving_log.csv : recorded image files description
#               (2) IMG/*.jpg : images recorded for traning DNN
#  Output    :  A trained model for autonomous driving (on specific track)
#  Algorithm :  Generic functions for reading input training data, preprocessing
#               creating a DNN model and input data learned model outcome. KERAS
#               framework leveraged for this purpose.
#               Two models DNN/CNN LeNet and Nvidia are tried at different
#               instances to observe learning model performance. Nvidia 
#               self driving car learning model gives a better performace and
#               complete full track of driving smoothly without and drag.
# / ************************************************************************ /


#### Import of common packages
import os
import csv
import numpy as np
import cv2

#### Import of necessary KERAS libraries/functions 
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Conv2D, MaxPooling2D

#### Miscellenous Paramater's 
DATA_PATH = '../Data'             # training data inputs path
Model_FileName = 'model.h5'       # name of trained model file 
learn_iteration = 10              # model learning iterations
VAL_SPLIT_PERCENTAGE = .25        # validation split % of data

#### Data Retrieval and reading section
def getLines_loadImageMeasurement(dataPath):
    lines = []
    with open (dataPath + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
            
    images = []
    measurements = []
    for line in lines:
        for i in range(3):
            source_path = line[i]
            filename = source_path.split('/')[-1]
            current_path = dataPath + '/IMG/' + filename
            org_image = cv2.imread(current_path)
            image = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB) # Preprocess img
            images.append(image)
            measurement = float(line[3])
            measurements.append(measurement)

    print('preprocessed images')
    return np.array(images), np.array(measurements)

# image_flipped = np.fliplr(image)
# measurement_flipped = -measurement

#### creation of model preprocessing layer
def model_preprocessingLayer():
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0 - 0.3, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    return model

#### LeNet network for self driving car model
def leNetModel():
    model = createPreProcessingLayers()
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    return model

#### Nvidia Self Driving Car model definition
def Nvidia_Model():
    model = model_preprocessingLayer()
    model.add(Conv2D(24,5,5,subsample=(2,2),activation="relu"))
    model.add(Conv2D(36,5,5,subsample=(2,2),activation="relu"))
    model.add(Conv2D(48,5,5,subsample=(2,2),activation="relu"))
    model.add(Conv2D(64,3,3,activation="relu"))
    model.add(Conv2D(64,3,3,activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

#### train and save a modela
def train_saveModel(model, inputs, outputs, modelFile, learn_iteration):
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split = VAL_SPLIT_PERCENTAGE, shuffle=True, nb_epoch=learn_iteration)
    model.save('model.h5')
    print("Model savet in file " + modelFile)

####################### Section -2 clone.py #############################

print('Loading images')
X_train, y_train = getLines_loadImageMeasurement(dataPath=DATA_PATH)
print("size of Image =", X_train.shape[1:3])

model = Nvidia_Model()
print('Training model')
train_saveModel(model, X_train, y_train, Model_FileName, learn_iteration)

exit()

#####################
