###############################################################################
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 11:20:13 2018

@author: rajeev kumar sharma

Purpose : This file serves following functionalities 
            (a) import of all needful libraries, functions, packages & global variables
            (b) include functions defined in other files (lessons learned and global declaration)
            (c) read, visualise and preprocess data for classifier (vehicle and non vehicles)
            (d) initialize SVM classifier and train for vehicles/non-vehicles
            (e) test various functionalities defined - 
                i. all curriculam learned lessons- car identify, slide windows and draw boxes
                ii. prepare and teast - a pipeline to identify all vehicles in a frame
                iii. run pipeline on a video
            (d) fine tune pipeline for learning parametes
            (g) fine tune pipeline for elimination of fake identification (vehicle)
            (i) Intigrate lane line identification pipeline in program.
""" 
###############################################################################


### Import of STD library - package and misc functiona ########################
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import os
import pickle
import random
import warnings
from datetime import datetime
from scipy.ndimage.measurements import label

#### classifier libraries/function ############################################
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
#from sklearn.cross_validation import train_test_split

## user defined functions - learned in curriculam
from lesson_functions import *
#from hog_subsample import *
#from VD_SystemParam import *
from global_declarations import *
import laneline 

from moviepy.editor import VideoFileClip
from IPython.display import HTML

##### Global Variable 
global X_scaler
global svc

### System Parameters #########################################################
DEBUG_FLAG = 'False'
REDUCE_SAMPLE = 'True'
KITTI_DATA = 'False'
EXTRA_DATA = 'False'
sample_size = 2000           # Limiting number of samples
VEHICLES = ''               # Path for vehicle data
NOT_VEHICLES = ''           # Path for Not Vehicle data
projectDir1 = './vehicles/vehicles/GTI*/'             # GTI vehicles path
projectDir2 = './vehicles/vehicles/KITTI*/'           # KITTI vehicles path
projectDir3 = './non-vehicles/non-vehicles/GTI/'      # GTI non vehicle
projectDir4 =  './non-vehicles/non-vehicles/Extras/'  # Extra non vehicle
projectDir5 = './vehicles/vehicles/Whit*/'             # white vehicles path

OUTPUT_IMAGE = './Output/test_images'                      # Output path 
OUTPUT_FIND_CAR = './Output/find_cars/'          # Output path - additional
OUTPUT_VIDEO_PATH = './Output/processed_video/' # video output path

### Initialize variables
cars = []          # cars list
notcars = []       # non cars list

#### Classifier parameter decleration and initialization
color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, \
hist_bins, hist_range, spatial_feat, hist_feat, hog_feat, n_count = init_param()

print ("Initialized Paramaters:")
print("color_space :", color_space)
print("spatial_size :", spatial_size)
print("",)

#### Frames processing parameters
THRES, ALPHA, track_list, THRES_LEN, Y_MIN, heat_p, boxes_p  = init_car_find()

###############################################################################
### Method-1 :  Input of project Vehicle/Non-Vehicle and Misc Images
###############################################################################
projectFiles1 = glob.glob( os.path.join(projectDir1, '*.png') )
projectFiles2 = glob.glob( os.path.join(projectDir2, '*.png') )
projectFiles3 = glob.glob( os.path.join(projectDir3, '*.png') )
projectFiles4 = glob.glob( os.path.join(projectDir4, '*.png') )
projectFiles5 = glob.glob( os.path.join(projectDir5, '*.png') )

for image in projectFiles1:
    cars.append(image)
print("first set of vehicles read (GTI) : ", len(cars) ) # first set of vehicle list

if KITTI_DATA == 'True':
    for image in projectFiles2:
        cars.append(image)
    print("Second set of vehicles read (KTTI) & append : ", len(cars) ) # Second set of vehicle list & append

for image in projectFiles3:
    notcars.append(image)
print("First set of non vehicles read (GTI) : ", len(notcars) ) # First set (GTI) of non vehicle list

if EXTRA_DATA == 'True':       # Non-vehicle Data 
    for image in projectFiles4:
        notcars.append(image)
    print("Second set of non vehicles read (Extras) & append : ", len(notcars) ) # Second set of non vehicle list
    
print("",)
print("",)

## make REDUCE_SAMPLE False- if you don't want to reduce the sample size
if REDUCE_SAMPLE == 'True':
    #cars = cars[0:sample_size]
    #notcars = notcars[0:sample_size]
    cars = random.sample(cars, sample_size)  # randomize
    notcars = random.sample(notcars, sample_size) # randomize
    print("length of cars images : ", len(cars))
    print("length of non car images : ", len(notcars))
else:
    print("Complete set of images used")
    
#for image in projectFiles5:
#    image = cv2.resize(image  , (64 , 64))    # resize image into 64*64
#    cars.append(image)
#print("Additional white vehicles sample (Rajeev) : ", len(cars) ) # additional set of white vehicle

###############################################################################

################ Classifier Section ###########################################
# Define parameters for feature extraction
car_features = extract_features(cars, color_space=color_space, spatial_size=spatial_size,
                                hist_bins=hist_bins, hist_range=hist_range , orient=orient,
                                pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                                hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                hist_feat=hist_feat, hog_feat=hog_feat)
print ("Car samples (features) : ", len(car_features))

notcar_features = extract_features(notcars, color_space=color_space, spatial_size=spatial_size,
                                   hist_bins=hist_bins, hist_range=hist_range, orient=orient,
                                   pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                                   hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                   hist_feat=hist_feat, hog_feat=hog_feat)
print ("Notcar samples (feature)) : ", len(notcar_features))
print("", )

X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
X_scaler = StandardScaler(copy=True, with_mean=False, with_std=False).fit(X) # Fit a per-column scaler
scaled_X = X_scaler.transform(X) # Apply the scaler to X
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features)))) # Define the labels vector

# Split up data into randomized training and test sets
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=22)

print('Using:',orient,'orientations', pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
print("", )

svc = LinearSVC(loss='hinge') # Use a linear SVC 
t=time.time() # Check the training time for the SVC
svc.fit(X_train, y_train) # Train the classifier
t2 = time.time()

print(round(t2-t, 2), 'Seconds to train SVC...')
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4)) # Check the score of the SVC
print("", )    

###### classifier test on test_images #########################################
t=time.time() # Start time
for image_p in glob.glob('test_images/test*.jpg'):
    image = cv2.imread(image_p)
    draw_image = np.copy(image)
    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[400, 640], 
                    xy_window=(128, 128), xy_overlap=(0.85, 0.85))
    hot_windows = []
    hot_windows += (search_windows(image, windows, svc, X_scaler, 
                                   color_space=color_space, spatial_size=spatial_size, 
                                   hist_bins=hist_bins, hist_range=hist_range, orient=orient, 
                                   pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                                   hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                   hist_feat=hist_feat, hog_feat=hog_feat)) 
                      
    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)                    
    #show_img(window_img)
    fname = OUTPUT_IMAGE + image_p 
    cv2.imwrite(fname, window_img)
    print("draw box image processed (glob & test input) : ", fname)

print(round(time.time()-t, 2), 'Seconds to process test images')
print("", )

###############################################################################

####### Detect new cars - a few Samples #######################################
### One test image -2 
image = cv2.imread('test_images/test2.jpg')
windows = slide_window(image, x_start_stop=[930, None], y_start_stop=[420, 650], 
                    xy_window=(128, 128), xy_overlap=(0.75, 0.75))
windows += slide_window(image, x_start_stop=[0, 350], y_start_stop=[420, 650], 
                    xy_window=(128, 128), xy_overlap=(0.75, 0.75))
window_img = draw_boxes(image, windows, color=(0, 0, 255), thick=6) 
windows = slide_window(image, x_start_stop=[400, 880], y_start_stop=[400, 470], 
                    xy_window=(48, 48), xy_overlap=(0.75, 0.75))
window_img = draw_boxes(window_img, windows, color=(255, 0, 0), thick=6)

fname = OUTPUT_IMAGE + 'draw_box_test2.jpg'
cv2.imwrite(fname, window_img)
print("draw box image processed : ", fname)
print("", )

#### One test Image -5
image = cv2.imread('test_images/test5.jpg')
track = (890, 450)
w_size = 80
windows = slide_window(image, x_start_stop=[track[0]-w_size,track[0]+w_size], 
                       y_start_stop=[track[1]-w_size,track[1]+w_size], 
                       xy_window=(115, 100), xy_overlap=(0.75, 0.75))
window_img = draw_boxes(image, windows, color=(0, 0, 255), thick=4)

track = (350, 420)
w_size = 30
windows = slide_window(image, x_start_stop=[track[0]-w_size,track[0]+w_size], 
                       y_start_stop=[track[1]-int(w_size),track[1]+int(w_size)], 
                       xy_window=(45, 45), xy_overlap=(0.75, 0.75))
window_img = draw_boxes(window_img, windows, color=(0, 255, 0), thick=4) 

track = (1180, 470)
w_size = 100
windows = slide_window(image, x_start_stop=[track[0]-w_size,track[0]+w_size], 
                       y_start_stop=[track[1]-int(w_size),track[1]+int(w_size)], 
                       xy_window=(150, 140), xy_overlap=(0.75, 0.75))
window_img = draw_boxes(window_img, windows, color=(255, 0, 0), thick=4) 

track = (800, 400)
w_size = 20
windows = slide_window(image, x_start_stop=[track[0]-w_size,track[0]+w_size], 
                       y_start_stop=[track[1]-int(w_size),track[1]+int(w_size)], 
                       xy_window=(30, 30), xy_overlap=(0.75, 0.75))
window_img = draw_boxes(window_img, windows, color=(255, 255, 0), thick=4)

                  
fname = OUTPUT_IMAGE + 'slide_window_test5.jpg'
cv2.imwrite(fname, window_img)
print("Slide window image processed : ", fname)
print("", )


######## Define a process frame & apply on a image
image = cv2.imread('test_images/test6.jpg')

out_image = find_car_process(image)

fname = OUTPUT_FIND_CAR + 'find_car_process_test6.jpg'
cv2.imwrite(fname, out_image)
print("find_car_process() test6 image outcome : ", fname)
print("", )

############### Lane lines identification - function call #####################
for image in glob.glob('test_images/test4.jpg'):
    img = laneline.draw_lane(laneline.draw_lane_img_p(image))
    
    if len(img.shape)==3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        print("Need to write gray image")
    
    # show_img(laneline.draw_lane(laneline.draw_lane_img_p(image)))
    fname = OUTPUT_IMAGE + 'lane_line_test5.jpg'
    cv2.imwrite(fname, img)
    print("Image processed - lane line identified : ", fname)
    print("", )
###############################################################################

###################### Frames Processing - Class Section ######################
if DEBUG_FLAG == 'True':
    color_space, orient, pix_per_cell, cell_per_block, hog_channel, \
    spatial_size, hist_bins, hist_range, spatial_feat, hist_feat, \
    hog_feat,n_count = init_param()

####### find a car in images - hog samples processing #########################
    
    #### First tried car find methodology - taught in curriculam ##############
    #####
    # This approach took input though PICKLE
    # Function used in as is explained in curriculam
    # feature analysis parameters are globally declared and initialized
    #####
    
    dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
    svc = dist_pickle["svc"]
    X_scaler = dist_pickle["scaler"]
    orient = dist_pickle["orient"]
    pix_per_cell = dist_pickle["pix_per_cell"]
    cell_per_block = dist_pickle["cell_per_block"]
    spatial_size = dist_pickle["spatial_size"]
    hist_bins = dist_pickle["hist_bins"]
    
    # img = mpimg.imread('test_image.jpg')
    ystart = 400
    ystop = 656
    scale = 1.02
    
    image = cv2.imread('test_images/test5.jpg') 
    out_img = find_cars_cu(image, ystart, ystop, scale, svc, X_scaler, \
                           color_space, orient, pix_per_cell, \
                           cell_per_block, spatial_size, hist_bins, hist_range)
    # plt.imshow(out_img)
    # img = frame_proc(image, lane=True, vis=False)
    
    fname = OUTPUT_FIND_CAR + 'find_car_cu_test5.jpg'
    cv2.imwrite(fname, out_img)
    
    print("find_car_test5.jpg - find car function outcome : ", fname)
    print("", )
###############################################################################

###### Pipeline Execution : On test images 
test_images = glob.glob('./test_images/test*.jpg')

for i, im in enumerate(test_images):
    image = cv2.imread(im)
    out_image = find_car_process(image)
    fname = OUTPUT_FIND_CAR + 'find_car_process_pipeline_test' + str(i+1) + '.jpg'
    cv2.imwrite(fname, out_image)
    print("find_car_process() test* images outcome : ", fname)
    
print("", )   

################ Video process - Section A ####################################
VIDEO_PROCE = False
if VIDEO_PROCE != False:
    #### video processing - initial
    test_out_file = OUTPUT_VIDEO_PATH  + str(datetime.now()) + 'test_video_out.mp4'
    clip_test = VideoFileClip('test_video.mp4')
    #clip1 = VideoFileClip("project_video.mp4")
    clip_test_out = clip_test.fl_image(find_car_process)
    clip_test_out.write_videofile(test_out_file, audio=False)   
    
    
    ###### Pipeline - Improvisation (keeping previous rectangle)
    ## a class defined (keep previous rects)
    ## Update new location if len(rect) >15   
    test_out_file2 = OUTPUT_VIDEO_PATH  + str(datetime.now()) + 'test_video_out2.mp4'
    clip_test = VideoFileClip('test_video.mp4')
    clip_test_out2 = clip_test.fl_image(find_car_process_video)
    clip_test_out2.write_videofile(test_out_file2, audio=False)    
        
    ### apply pipeline on project video 
    proj_out_file = OUTPUT_VIDEO_PATH  + str(datetime.now()) + 'project_video_out.mp4'
    clip_proj = VideoFileClip('project_video.mp4') 
    clip_proj_out = clip_proj.fl_image(find_car_process_video)
    clip_proj_out.write_videofile(proj_out_file, audio=False)
###############################################################################                               
                               
######## Alternative Section : Curriculam functions trial #####################      
###################### Frames Processing - Class Section ######################
if DEBUG_FLAG == True :
    color_space, orient, pix_per_cell, cell_per_block, hog_channel, \
    spatial_size, hist_bins, hist_range, spatial_feat, hist_feat, \
    hog_feat,n_count = init_param()
    
    ####### find a car in images - hog samples processing #####################
    
    #### First tried car find methodology - taught in curriculam ##############
    #####
    # This approach took input though PICKLE
    # Function used in as is explained in curriculam
    # feature analysis parameters are globally declared and initialized
    #####
    
    dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
    svc = dist_pickle["svc"]
    X_scaler = dist_pickle["scaler"]
    orient = dist_pickle["orient"]
    pix_per_cell = dist_pickle["pix_per_cell"]
    cell_per_block = dist_pickle["cell_per_block"]
    spatial_size = dist_pickle["spatial_size"]
    hist_bins = dist_pickle["hist_bins"]
    
    # img = mpimg.imread('test_image.jpg')
    ystart = 400
    ystop = 656
    scale = 1.02
    
    image = cv2.imread('test_images/test5.jpg') 
    out_img = find_cars_cu(image, ystart, ystop, scale, svc, X_scaler, \
                           color_space, orient, pix_per_cell, \
                           cell_per_block, spatial_size, hist_bins, hist_range)
    # plt.imshow(out_img)
    # img = frame_proc(image, lane=True, vis=False)
    
    fname = OUTPUT_FIND_CAR + 'find_car_test5.jpg'
    cv2.imwrite(fname, out_img)
    
    print("find_car_test5.jpg - find car function outcome : ", fname)
    print("", )
###############################################################################


############   SECTION - Alternative function trials ##########################
################ find cars - section testing ##################################
#### Outcome of this section saved in folder: find_cars #######################
    
########## Section -1 : Curriculam taught "find_car" outcome trial
image = cv2.imread('test_images/test4.jpg') 

ystart = 400
ystop = 656
scale = 2.8
colorspace = color_space

out_img = find_cars_cu(image, ystart, ystop, scale, svc, X_scaler, \
                           color_space, orient, pix_per_cell, \
                           cell_per_block, spatial_size, hist_bins, hist_range)

fname = OUTPUT_FIND_CAR + 'cu_test4.jpg'
cv2.imwrite(fname, out_img)
print("find car (curriculam) function outcome : ", fname)
print("", )

##### Section-2 : Identification of "find_car" boxes or rectangle function ####
image = cv2.imread('test_images/test6.jpg')

ystart = 400
ystop = 656
scale = 3.5
colorspace = color_space

car_boxes = find_cars(image, ystart, ystop, scale, colorspace, hog_channel, 
                      svc, X_scaler, orient, pix_per_cell, cell_per_block, 
                      spatial_size, hist_bins, hist_range, show_all_rectangles=False)

print(len(car_boxes), 'car_boxes (rectangles) found in image (test6)')
print("", )

### draw boxes in image 
out_img = draw_boxes(image, car_boxes, color=(0,0,255), thick=6)
fname = OUTPUT_FIND_CAR + 'cu_test6.jpg'
cv2.imwrite(fname, out_img)
print("find car boxes values - draw outcome : ", fname)
print('...')


####################### cover multiple potential area of a image ##############
image = cv2.imread('test_images/test1.jpg')
scale = 3.0
colorspace = color_space
boxes = []

boxes.append(find_cars(image, 405, 650, scale, colorspace, hog_channel, svc, 
                       X_scaler, orient, pix_per_cell, cell_per_block, 
                       spatial_size, hist_bins, hist_range, show_all_rectangles=False))
boxes.append(find_cars(image, 415, 660, scale, colorspace, hog_channel, svc, 
                       X_scaler, orient, pix_per_cell, cell_per_block, 
                       spatial_size, hist_bins, hist_range, show_all_rectangles=False))
boxes.append(find_cars(image, 425, 670, scale, colorspace, hog_channel, svc, 
                       X_scaler, orient, pix_per_cell, cell_per_block, 
                       spatial_size, hist_bins, hist_range, show_all_rectangles=False))

print(len(car_boxes), 'boxes found (rect) in multi-potential area image (test1)')
print("", )

### draw boxes in image 
rectangles = [item for sublist in boxes for item in sublist] 
out_img = draw_boxes(image, rectangles, color='random', thick=3)
print('Number of boxes: ', len(rectangles))

fname = OUTPUT_FIND_CAR + 'test1.jpg'
cv2.imwrite(fname, out_img)
print("find car boxes values (multi) - draw outcome : ", fname)
print('...')
###############################################################################


def frame_proc(img, lane, video, vis):
    ######## specific parameters defition #############  
    global heat_p, boxes_p
    #n_count = 0
    
    # Create deque for caching 10 frames
    from collections import deque
    cache = deque(maxlen=5)
    
    # if (video and n_count%2==0) or not video:  ## skip alternative frame
    if video == True: # Skip every second video frame
        heat = np.zeros_like(img[:,:,0]).astype(np.float)
        boxes = []
        boxes = find_car_slide_win(img)   ### find car's through slide window
        
        heat = add_heat(heat, boxes)
        # Add current heatmap to cache
        heat = cache.append(heat)
        # Accumulate heatmaps for thresholding, might use average as well
        heat = np.sum(cache, axis=0)
        heat = apply_threshold(heat,THRES) # Apply threshold to help remove false positives
        
        # Visualize the heatmap when displaying    
        heatmap = np.clip(heat, 0, 255)
        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        #print((labels[0]))
        cars_boxes = draw_labeled_bboxes(labels)
        boxes_p = cars_boxes 
    else:
        cars_boxes = boxes_p
        
        
    if lane: #If we was asked to draw the lane line, do it
        if video:
            img = laneline.draw_lane(img, True)
        else:
            img = laneline.draw_lane(img, False)
            
    imp = draw_boxes(np.copy(img), cars_boxes, color=(0, 0, 255), thick=6)
    
    if vis:
        imp = draw_boxes(imp, boxes, color=(0, 255, 255), thick=2)
        for track in track_list:
            cv2.circle(imp, (int(track[0]), int(track[1])), 5, color=(255, 0, 255), thickness=4)
            
    #n_count += 1
    return imp


def find_car_slide_win(img): 
    boxes = []
    #### Prepare box list for different position ##########################
    #find_cars_step(img, ystart, ystop, xstart, xstop, scale, step)
    boxes = find_cars_step(img, 420, 680, 950, 1280, 1, 2)  # Track-1
    boxes += find_cars_step(img, 420, 680, 950, 1280, .75, 2)
    boxes += find_cars_step(img, 420, 680, 950, 1280, .5, 2)
    boxes += find_cars_step(img, 420, 680, 950, 1280, .25, 2)     
             
    boxes += find_cars_step(img, 420, 670, 680, 1000, 1, 2) # Track-2
    boxes += find_cars_step(img, 420, 670, 680, 1000, .75, 2)
    boxes += find_cars_step(img, 420, 670, 680, 1000, .5, 2)
    boxes += find_cars_step(img, 420, 670, 680, 1000, .25, 2)
    #######################################################################
    return boxes

def find_cars_step(img, ystart, ystop, xstart, xstop, scale, step):
    
    color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, \
    hist_bins, hist_range, spatial_feat, hist_feat, hog_feat,n_count = init_param()
    
    boxes = []  # identified rectangles variable
    
    draw_img = np.zeros_like(img)   
    img_tosearch = img[ystart:ystop,xstart:xstop,:]
    ctrans_tosearch = convert_color(img_tosearch, color_space)
    
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    if hog_channel == 'ALL':
        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]
    else: 
        ch1 = ctrans_tosearch[:,:,hog_channel]    
    
    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) -1
    cells_per_step = step  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)   
    if hog_channel == 'ALL':
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
       
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            
            # Extract HOG for this patch           
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            if hog_channel == 'ALL':
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            else:
                hog_features = hog_feat1
            
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell
            
            # Extract the image patch
            subimg = ctrans_tosearch[ytop:ytop+window, xleft:xleft+window]           
            # Get color features
            spatial_features = bin_spatial(subimg, spatial_size)
            hist_features = color_hist(subimg, hist_bins, hist_range)
            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)) 
            test_prediction = svc.predict(test_features)
            #test_features = np.hstack(hog1).reshape(1, -1)
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)+xstart
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                boxes.append(((int(xbox_left), int(ytop_draw+ystart)),
                              (int(xbox_left+win_draw),
                               int(ytop_draw+win_draw+ystart))))
    return boxes

############ Implement & Test - Video Processing ##############################

###### Project video - to apply pipeline
from moviepy.editor import VideoFileClip
#global n_count
laneline.init_params(0.0)
#n_count = 0

def process_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(frame_proc(image, lane=True, video=True, vis=False), \
                        cv2.COLOR_BGR2RGB)

#### Project output
output_p = OUTPUT_VIDEO_PATH + 'project_video_processed' + str(datetime.now()) + '.mp4'
clip1 = VideoFileClip("project_video.mp4")
#clip1 = VideoFileClip("project_video.mp4").subclip(18,25)
clip = clip1.fl_image(process_image)
clip.write_videofile(output_p, audio=False)

#### Test output -1 
output_v1 = OUTPUT_VIDEO_PATH + 'test_video_processed' + str(datetime.now()) + '.mp4'
clip2 = VideoFileClip("test_video.mp4")
clip_o = clip2.fl_image(process_image)
clip_o.write_videofile(output_v1, audio=False)

#################### end of frame - process pipeline ##########################
