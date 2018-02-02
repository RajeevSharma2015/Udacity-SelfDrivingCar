

### Import of STD library - package and misc functiona ########################
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import os
import pickle
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


### System Parameters #########################################################
DEBUG_FLAG = 'False'
REDUCE_SAMPLE = 'True'
sample_size = 500   # Limiting number of samples
VEHICLES = ''       # Path for vehicle data
NOT_VEHICLES = ''   # Path for Not Vehicle data
projectDir1 = './vehicles/vehicles/GTI*/'             # GTI vehicles path
projectDir2 = './vehicles/vehicles/KITTI*/'           # KITTI vehicles path
projectDir3 = './non-vehicles/non-vehicles/GTI/'      # GTI non vehicle
projectDir4 =  './non-vehicles/non-vehicles/Extras/'  # Extra non vehicle
OUTPUT_PATH = './output_images/'                      # Output path 
OUTPUT_PATH1 = './output_images/Additional/'          # Output path - additional
OUTPUT_VIDEO_PATH = './output_images/processed_video/' # video output path

### Initialize variables
cars = []          # cars list
notcars = []       # non cars list

#### Classifier parameter decleration and initialization
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

color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, \
hist_bins, hist_range, spatial_feat, hist_feat, hog_feat,n_count = init_param()

print ("Initialized Paramaters:")
print("color_space :", color_space)
print("spatial_size :", spatial_size)
print("",)

#### Frames processing parameters
THRES = 3          # Minimal overlapping boxes
ALPHA = 0.75       # Filter parameter, weight of the previous measurements
track_list = []    #[np.array([880, 440, 76, 76])]
# track_list += [np.array([1200, 480, 124, 124])]
THRES_LEN = 32                   # Thresold
Y_MIN = 440
heat_p = np.zeros((720, 1280))   # Store prev heat image
boxes_p = []                     # Store prev car boxes
#n_count = 0                      # Frame counter

###############################################################################
### Method-1 :  Input of project Vehicle/Non-Vehicle and Misc Images
###############################################################################
projectFiles1 = glob.glob( os.path.join(projectDir1, '*.png') )
projectFiles2 = glob.glob( os.path.join(projectDir2, '*.png') )
projectFiles3 = glob.glob( os.path.join(projectDir3, '*.png') )
projectFiles4 = glob.glob( os.path.join(projectDir4, '*.png') )

for image in projectFiles1:
    cars.append(image)
print("first set of vehicles read (GTI) : ", len(cars) ) # first set of vehicle list

for image in projectFiles2:
    cars.append(image)
print("Second set of vehicles read (KTTI) & append : ", len(cars) ) # Second set of vehicle list & append

for image in projectFiles3:
    notcars.append(image)
print("First set of non vehicles read (GTI) : ", len(notcars) ) # First set (GTI) of non vehicle list

for image in projectFiles4:
    notcars.append(image)
print("Second set of non vehicles read (Extras) & append : ", len(notcars) ) # Second set of non vehicle list
print("",)
print("",)

## make REDUCE_SAMPLE False- if you don't want to reduce the sample size
if REDUCE_SAMPLE == 'True':
    cars = cars[0:sample_size]
    notcars = notcars[0:sample_size]
    print("length of cars images : ", len(cars))
    print("length of non car images : ", len(notcars))
else:
    print("Complete set of images used")
###############################################################################


###############################################################################
### Method-2 : Input of cars and notcars
###############################################################################
#images_car = glob.glob('vehicles/vehicles/GTI*/*.png')
#for image in images_car:
#    cars.append(image)
## Uncomment if you need to reduce the sample size
# cars = cars[0:sample_size]
#print(len(cars))

### Read in notcars
#images_notcar = glob.glob('non_vehicles/non_vehicles/GTI/*.png')
#for image in images_notcar:
#    notcars.append(image)
## Uncomment if you need to reduce the sample size
# notcars = notcars[0:sample_size]
#print(len(notcars))
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
    fname = OUTPUT_PATH + image_p 
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

fname = OUTPUT_PATH + 'test_images/draw_box_test2.jpg'
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

                  
fname = OUTPUT_PATH + 'test_images/slide_window_test5.jpg'
cv2.imwrite(fname, window_img)
print("Slide window image processed : ", fname)
print("", )


######## Define a process frame & apply on a image
image = cv2.imread('test_images/test6.jpg')

out_image = find_car_process(image)

fname = OUTPUT_PATH + 'test_images/find_car_process_test6.jpg'
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
    fname = OUTPUT_PATH + 'test_images/lane_line_test5.jpg'
    cv2.imwrite(fname, img)
    print("Image processed - lane line identified : ", fname)
    print("", )
###############################################################################

###### Pipeline Execution : On test images 
test_images = glob.glob('./test_images/test*.jpg')

for i, im in enumerate(test_images):
    image = cv2.imread(im)
    out_image = find_car_process(image)
    fname = OUTPUT_PATH + 'test_images/' + 'find_car_process_pipeline_test' + str(i+1) + '.jpg'
    cv2.imwrite(fname, out_image)
    print("find_car_process() test* images outcome : ", fname)
    
print("", )   

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
                               
                               
######## Alternative Section : Curriculam functions trial #####################    
    
###################### Frames Processing - Class Section ######################
if DEBUG_FLAG == True :
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
    
    fname = OUTPUT_PATH + 'test_images/find_car_test5.jpg'
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
scale = 1.5
colorspace = color_space

out_img = find_cars_cu(image, ystart, ystop, scale, svc, X_scaler, \
                           color_space, orient, pix_per_cell, \
                           cell_per_block, spatial_size, hist_bins, hist_range)

fname = OUTPUT_PATH + 'find_cars/test4.jpg'
cv2.imwrite(fname, out_img)
print("find car (curriculam) function outcome : ", fname)
print("", )

##### Section-2 : Identification of "find_car" boxes or rectangle function ####
image = cv2.imread('test_images/test6.jpg')

ystart = 400
ystop = 656
scale = 1.5
colorspace = color_space

car_boxes = find_cars(image, ystart, ystop, scale, colorspace, hog_channel, 
                      svc, X_scaler, orient, pix_per_cell, cell_per_block, 
                      spatial_size, hist_bins, hist_range, show_all_rectangles=False)

print(len(car_boxes), 'car_boxes (rectangles) found in image (test6)')
print("", )

### draw boxes in image 
out_img = draw_boxes(image, car_boxes, color=(0,0,255), thick=6)
fname = OUTPUT_PATH + 'find_cars/test6.jpg'
cv2.imwrite(fname, out_img)
print("find car boxes values - draw outcome : ", fname)
print('...')


####################### cover multiple potential area of a image ####

image = cv2.imread('test_images/test1.jpg')
scale = 2.0
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

fname = OUTPUT_PATH + 'find_cars/test1.jpg'
cv2.imwrite(fname, out_img)
print("find car boxes values (multi) - draw outcome : ", fname)
print('...')


############ Define Video Processing ##########################################
from moviepy.editor import VideoFileClip
n_count = 0
laneline.init_params(0.0)
def process_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(frame_proc(image, lane=True, video=True, vis=False), \
                        cv2.COLOR_BGR2RGB)

output_v = 'project_video_processed' + str(datetime.now()) + '.mp4'
clip1 = VideoFileClip("project_video.mp4")
#clip = clip1.fl_image(process_image)
#clip.write_videofile(output_v, audio=False)

