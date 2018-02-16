 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 10:31:24 2018

@author: rajeev kumar sharma

Purpose : This file defines all function, method and classes used in vehicle 
          detection program. It also initialises system parameters used in 
          this module. 
"""
import numpy as np
import cv2
from skimage.feature import hog
###############################################################################

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 11:20:13 2018

@author: rajeev
"""
###############################################################################


import matplotlib.image as mpimg
from scipy.ndimage.measurements import label
from global_declarations import *

# import VD_SystemParam


def init_param():
    #color_space = 'YCrCb'
    color_space = 'YCrCb'           # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 11                    # HOG orientations
    pix_per_cell = 8              # HOG pixels per cell
    #cell_per_block = 3            # HOG cells per block
    cell_per_block = 2            # HOG cells per block
    hog_channel = 'ALL'           # Can be 0, 1, 2, or "ALL"
    #hog_channel = 0               # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16)       # Spatial binning dimensions
    hist_bins = 32                # Number of histogram bins
    hist_range = (0,256)          # need to change for PNG
    spatial_feat = True           # Spatial features on or off
    hist_feat = True              # Histogram features on or off
    hog_feat = True               # HOG features on or off
    n_count = 0                   # Frame counter
    return color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, hist_range, spatial_feat, hist_feat, hog_feat, n_count

def init_car_find():
    THRES = 10          # Minimal overlapping boxes - 3
    ALPHA = 0.75       # Filter parameter, weight of the previous measurements - .75
    track_list = []    #[np.array([880, 440, 76, 76])]
    # track_list += [np.array([1200, 480, 124, 124])]
    THRES_LEN = 32                   # Thresold - 32
    Y_MIN = 440                      # 440 - base value
    heat_p = np.zeros((720, 1280))   # Store prev heat image
    boxes_p = []                     # Store prev car boxes
    return THRES, ALPHA, track_list, THRES_LEN, Y_MIN, heat_p, boxes_p 

########## Section-A Lessions Learned functions ###############################
# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis, feature_vec):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to compute binned color features  
def bin_spatial(img, spatial_size):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, spatial_size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins, hist_range):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=hist_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=hist_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=hist_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space, spatial_size,hist_bins, hist_range, 
                     orient, pix_per_cell, cell_per_block, hog_channel, 
                     spatial_feat, hist_feat, hog_feat):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, spatial_size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, hist_bins, hist_range)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features
    
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop, y_start_stop, 
                    xy_window, xy_overlap):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color, thick):
    # Make a copy of the image
    imcopy = np.copy(img)
    random_color = False
    # Iterate through the bounding boxes
    for bbox in bboxes:
        if color == 'random' or random_color:
            color = (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
            random_color = True
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
    
def draw_labeled_bboxes_n(img, labels):
    # Iterate through all detected cars
    rects = []
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        rects.append(bbox)
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image and final rectangles
    return img, rects


def search_windows(img, windows, clf, scaler, color_space, spatial_size, 
                   hist_bins, hist_range, orient, pix_per_cell, cell_per_block,
                   hog_channel, spatial_feat, hist_feat, hog_feat):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, hist_range=hist_range,
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

def single_img_features(img, color_space, spatial_size,
                        hist_bins, hist_range, orient, 
                        pix_per_cell, cell_per_block, hog_channel,
                        spatial_feat, hist_feat, hog_feat):    

    img_features = [] # Define an empty list to receive features
    feature_image = convert_color(img, color_space) # col conversion
    if spatial_feat == True: # Compute spatial features if flag is set
        spatial_features = bin_spatial(feature_image, spatial_size=spatial_size)
        img_features.append(spatial_features) # Append features to list
    if hist_feat == True:# Compute histogram features if flag is set
        hist_features = color_hist(feature_image, hist_bins, hist_range)
        img_features.append(hist_features) #6Append features to list
    
    if hog_feat == True:  # Compute HOG features if flag is set
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
       
        img_features.append(hog_features)  # Append features to list
    return np.concatenate(img_features) # Return concatenated array of features


def convert_color(img, conv):
    if conv != 'RGB':
        if conv == 'LUV':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif conv == 'YCrCb':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        elif conv == 'HSV':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif conv == 'YUV':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif conv == 'HLS':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    else: 
        img = np.copy(img)
        
    return img


######### Section-B Lesson learned function ###################################
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    return heatmap # Return updated heatmap
    
def apply_threshold(heatmap, threshold): # Zero out pixels below the threshold in the heatmap
    heatmap[heatmap < threshold] = 0 
    return heatmap 

def filt(a,b,alpha): # Smooth the car boxes
    return a*alpha+(1.0-alpha)*b

def len_points(p1, p2): # Distance beetween two points
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def track_to_box(p): # Create box coordinates out of its center and span
    return ((int(p[0]-p[2]),int(p[1]-p[3])),(int(p[0]+p[2]), int(p[1]+p[3])))


def draw_labeled_bboxes(labels):
    
    THRES, ALPHA, track_list, THRES_LEN, Y_MIN, heat_p, boxes_p  = init_car_find()
    
    global track_list
    track_list_l = []
     
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        #img = draw_boxes(np.copy(img), [bbox], color=(255,0,255), thick=3)
        size_x = (bbox[1][0]-bbox[0][0])/2.0 #Size of the found box
        size_y = (bbox[1][1]-bbox[0][1])/2.0
        asp_d = size_x / size_y
        size_m = (size_x + size_y)/2
        x = size_x+bbox[0][0]
        y = size_y+bbox[0][1]
        asp = (y-Y_MIN)/130.0+1.2 # Best rectangle aspect ratio for the box (coefficients from perspectieve measurements and experiments)
        if x>1050 or x<230:
            asp*=1.4
        asp = max(asp, asp_d) # for several cars chunk
        size_ya = np.sqrt(size_x*size_y/asp)
        size_xa = int(size_ya*asp)
        size_ya = int(size_ya)
        
        if x > (-3.049*y+1809): #If the rectangle on the road, coordinates estimated from a test image
            track_list_l.append(np.array([x, y, size_xa, size_ya]))
            if len(track_list) > 0:
                track_l = track_list_l[-1]
                dist = []
                for track in track_list:
                    dist.append(len_points(track, track_l))
                min_d = min(dist)
                if min_d < THRES_LEN:
                    ind = dist.index(min_d)
                    track_list_l[-1] = filt(track_list[ind], track_list_l[-1], ALPHA)
    track_list = track_list_l
    boxes = []
    for track in track_list_l:
        #print(track_to_box(track))
        boxes.append(track_to_box(track))
    return boxes


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars_cu(img, ystart, ystop, scale, svc, X_scaler, col_space, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, hist_range):
    
    ########## local variables defined for video frames processing
    #color_space = 'LUV'           # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    #orient = 8                    # HOG orientations
    #pix_per_cell = 8              # HOG pixels per cell
    #cell_per_block = 2            # HOG cells per block
    #hog_channel = 0               # Can be 0, 1, 2, or "ALL"
    #spatial_size = (16, 16)       # Spatial binning dimensions
    #hist_bins = 32                # Number of histogram bins
    #spatial_feat = True           # Spatial features on or off
    #hist_feat = True              # Histogram features on or off
    #hog_feat = True               # HOG features on or off

    print("",)
    print("pix_per_cell value passed : ", pix_per_cell)
    print("spatial_size value passed : ", spatial_size)
    print("hist_bins value passed : ", hist_bins)
    
    #pix_per_cell = 8
    #spatial_size = (16, 16)
    #hist_bins = 32
    
    draw_img = np.copy(img)
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, col_space)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, spatial_size)
            hist_features = color_hist(subimg, hist_bins, hist_range)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                #print('test prediction TRUE')
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                
    return draw_img

########## Define a single function - identify cars and visualize #############
### Objective - extract features using hog sub-sampling 
### Also make predictions
### Provide list of identified car or rectangles
    
def find_cars(img, ystart, ystop, scale, cspace, hog_channel, svc, X_scaler, 
              orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, hist_range, 
              show_all_rectangles=False):
    
    ## Initialize variable and pre-process (if colour other than RGB)
    rectangles = []  # array of rectangles where cars were detected
    img = img.astype(np.float32)/255
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch,cspace) # apply color conversion
    # rescale image if other than 1.0 scale
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, 
                                     (np.int(imshape[1]/scale), 
                                      np.int(imshape[0]/scale)))
    # select colorspace channel for HOG 
    if hog_channel == 'ALL':
        ch1 = ctrans_tosearch[:,:,0]
        ch2 = ctrans_tosearch[:,:,1]
        ch3 = ctrans_tosearch[:,:,2]
    else: 
        ch1 = ctrans_tosearch[:,:,hog_channel]


    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)+1  #-1
    nyblocks = (ch1.shape[0] // pix_per_cell)+1  #-1 
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    cells_per_step = 2  # Instead of overlap, define how many cells to step
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
            
            if X_scaler != None :
                ############## ONLY FOR BIN_SPATIAL AND COLOR_HIST ################
                #### Section commented - due to OpenCv error i.e. resize bug ######
                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
                # Get color features
                spatial_features = bin_spatial(subimg, spatial_size)
                hist_features = color_hist(subimg, hist_bins, hist_range)

                # Scale features and make a prediction
                # test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
                #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshapespatial_features(1, -1))    
                #test_prediction = svc.predict(test_features) 
                ###################################################################
            
                #test_features = X_scaler.transform(np.hstack((hist_features, hog_features)).reshape(1, -1))
                # test_prediction = svc.predict(hog_features)
                #hog_features = np.hstack(hog_features).reshape(-1, 1)
                #spatial_features = np.hstack(spatial_features).reshape(-1, 1)
                #hist_features = np.hstack(hist_features).reshape(-1, 1)
            
                #print(np.size(hog_features), 'size of hog features', 
                #      np.size(spatial_features), 'size of spatial features', 
                #      np.size(hist_features), 'size of histogram feature')
                #print("",)
            
                #print(len(hog_features), 'number of hog features', 
                #      len(spatial_features), 'number of spatial features', 
                #      len(hist_features), 'number of histogram feature')
                #print("",)
            
                #X_feature = np.concatenate(spatial_features, hist_features, hog_features)
                ###################################################################
            
                test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                # test_features = X_scaler.transform(np.hstack(hog_features).reshape(1, -1))
                test_prediction = svc.predict(test_features)
            else:
                print(np.shape(hog_features), 'hog feature shape')
                test_features = np.hstack(hog_features).reshape(1, -1)
                #test_features = hog_features
                test_prediction = svc.predict(test_features)
            
            if test_prediction == 1 or show_all_rectangles:
               # print('Test prediction TRUE')
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                rectangles.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
              
    return rectangles

#
#
# print(np.shape(hog_features), 'size of hog features', 
#                  np.shape(spatial_features), 'size of spatial features', 
#                  np.shape(hist_features), 'size of histogram feature')
#            print("",)
#            X_feature = []
#            X_feature = np.concatenate((spatial_features, hist_features, hog_features), axis=0)
#            X_feature = np.reshape(X_feature, (1, -1))  
# print(np.shape(X_feature), ' Size of concatenated features')            
#            test_features = X_scaler.transform(X_feature)
  

def find_car_process(image):
    ######## specific parameters defition #############    
    boxes = [] 
    
    track = (890, 450)
    w_size = 80
    boxes.append(slide_window(image, x_start_stop=[track[0]-w_size,track[0]+w_size], 
                              y_start_stop=[track[1]-w_size,track[1]+w_size], 
                              xy_window=(115, 100), xy_overlap=(0.75, 0.75)))

    track = (350, 420)
    w_size = 30
    boxes.append(slide_window(image, x_start_stop=[track[0]-w_size,track[0]+w_size], 
                              y_start_stop=[track[1]-int(w_size),track[1]+int(w_size)], 
                              xy_window=(45, 45), xy_overlap=(0.75, 0.75)))

    track = (1180, 470)
    w_size = 100
    boxes.append(slide_window(image, x_start_stop=[track[0]-w_size,track[0]+w_size], 
                              y_start_stop=[track[1]-int(w_size),track[1]+int(w_size)], 
                              xy_window=(150, 140), xy_overlap=(0.75, 0.75)))

    track = (800, 400)
    w_size = 20
    boxes.append(slide_window(image, x_start_stop=[track[0]-w_size,track[0]+w_size], 
                              y_start_stop=[track[1]-int(w_size),track[1]+int(w_size)], 
                              xy_window=(30, 30), xy_overlap=(0.75, 0.75)))
    
    boxes = [item for sublist in boxes for item in sublist] 
    
    heatmap_img = np.zeros_like(image[:,:,0])
    heatmap_img = add_heat(heatmap_img, boxes)
    heatmap_img = apply_threshold(heatmap_img, 1)
    
    labels = label(heatmap_img)
    img = np.copy(image)
    
    draw_img, rects = draw_labeled_bboxes_n(img, labels)
    
    return draw_img

# Define a class to store data from video
class Vehicle_Detect():
    def __init__(self):
        # history of rectangles previous n frames
        self.prev_rects = [] 
        
    def add_rects(self, rects):
        self.prev_rects.append(rects)
        if len(self.prev_rects) > 15:
            # throw out oldest rectangle set(s)
            self.prev_rects = self.prev_rects[len(self.prev_rects)-15:]
            
def find_car_process_video(image):
    ######## specific parameters defition #############
    det = Vehicle_Detect()
    boxes = [] 
    
    track = (890, 450)
    w_size = 80
    boxes.append(slide_window(image, x_start_stop=[track[0]-w_size,track[0]+w_size], 
                              y_start_stop=[track[1]-w_size,track[1]+w_size], 
                              xy_window=(115, 100), xy_overlap=(0.75, 0.75)))

    track = (350, 420)
    w_size = 30
    boxes.append(slide_window(image, x_start_stop=[track[0]-w_size,track[0]+w_size], 
                              y_start_stop=[track[1]-int(w_size),track[1]+int(w_size)], 
                              xy_window=(45, 45), xy_overlap=(0.75, 0.75)))

    track = (1180, 470)
    w_size = 100
    boxes.append(slide_window(image, x_start_stop=[track[0]-w_size,track[0]+w_size], 
                              y_start_stop=[track[1]-int(w_size),track[1]+int(w_size)], 
                              xy_window=(150, 140), xy_overlap=(0.75, 0.75)))

    track = (800, 400)
    w_size = 20
    boxes.append(slide_window(image, x_start_stop=[track[0]-w_size,track[0]+w_size], 
                              y_start_stop=[track[1]-int(w_size),track[1]+int(w_size)], 
                              xy_window=(30, 30), xy_overlap=(0.75, 0.75)))
    
    boxes = [item for sublist in boxes for item in sublist]
    if len(boxes) > 0:
        det.add_rects(boxes)
    
    heatmap_img = np.zeros_like(image[:,:,0])
    for rect_set in det.prev_rects:
        heatmap_img = add_heat(heatmap_img, rect_set)
    heatmap_img = apply_threshold(heatmap_img, 1 + len(det.prev_rects)//2)
     
    labels = label(heatmap_img)
    img = np.copy(image)
    draw_img, rect = draw_labeled_bboxes_n(img, labels)
    
    return draw_img
