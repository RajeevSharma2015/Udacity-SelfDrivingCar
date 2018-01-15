
# / ************************************************************************ /
#  File Name :  LaneDetection.py
#  Author    :  Rajeev Kumar Sharma
#  Date      :  05/01/2018
#  Input     :  Data file created by autonomous driving simulator i.e 
#               (1) calibration_test.jpg : recorded image files description
#               (2) IMG/*.jpg : images recorded for traning DNN
#  Output    :  A trained model for autonomous driving (on specific track)
#  Algorithm :  Generic functions for reading input training data, preprocessing
# / ************************************************************************ /

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#%matplotlib inline

# prepare object points
nx = 8     # TODO: enter the number of inside corners in x
ny = 6     # TODO: enter the number of inside corners in y

# Make a list of calibration images
fname = './camera_cal' + '/calibration1.jpg'
img = cv2.imread(fname)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.imwrite("Calibration-outgray.jpg", gray)

print("Image Chess board processing begin")
# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
if ret == True:
    ### Draw and display the corners
    cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
    cv2.imwrite("Calibration-final.jpg", img)
    # plt.imshow(img)
    print("Find Chess board corner successful")

# print("Find Chess board corner successful")

