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
  