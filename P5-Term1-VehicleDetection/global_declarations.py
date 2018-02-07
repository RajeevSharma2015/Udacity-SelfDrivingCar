#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 11:20:13 2018

@author: rajeev
"""

###### Global variables declarations ##########################################
#### Classifier parameter decleration and initialization
global color_space           # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
global orient                # HOG orientations
global pix_per_cell          # HOG pixels per cell
global cell_per_block        # HOG cells per block
global hog_channel           # Can be 0, 1, 2, or "ALL"
global spatial_size          # Spatial binning dimensions
global hist_bins             # Number of histogram bins
global bins_range            # Dependent on PNG or file reading
global spatial_feat          # Spatial features on or off
global hist_feat             # Histogram features on or off
global hog_feat              # HOG features on or off

# global n_count               # Frame counter
  

#### Frames processing parameters
global THRES          # Minimal overlapping boxes
global ALPHA          # Filter parameter, weight of the previous measurements
global track_list     #[np.array([880, 440, 76, 76])]
# track_list += [np.array([1200, 480, 124, 124])]
global THRES_LEN                   # Thresold
global Y_MIN
global heat_p         # Store prev heat image
global boxes_p        # Store prev car boxes
global n_count          # Frame counter

#### Declaration for classifier - global, initialized in main function 
global X_scaler
global svc

global THRES            # Minimal overlapping boxes
global ALPHA            # Filter parameter, weight of the previous measurements
global track_list       #[np.array([880, 440, 76, 76])]
# track_list += [np.array([1200, 480, 124, 124])]
global THRES_LEN        # Thresold
global Y_MIN
global heat_p           # Store prev heat image
global boxes_p          # Store prev car boxes