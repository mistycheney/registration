'''
Configuration variables.
'''

import cv2
import os
import numpy as np

if 'HOME' in os.environ and os.environ['HOME'] == '/Users/yuncong':
    DATA_FOLDER = '/Users/yuncong/Documents/brain images/'
    PROJ_FOLDER = '/Users/yuncong/Documents/workspace/Registration/'
elif 'HOMEPATH' in os.environ and os.environ['HOMEPATH'] == '\\Users\\velu':
    DATA_FOLDER = 'C:/Users/velu/Desktop/brain images/'
    PROJ_FOLDER = 'C:/Users/velu/Desktop/registration/'
else:
    DATA_FOLDER = '/oasis/scratch/csd181/yuncong/'
    PROJ_FOLDER = '/oasis/scratch/csd181/yuncong/registration/'
SLIDE_FOLDER = DATA_FOLDER + 'Slide/'
SECTION_FOLDER = DATA_FOLDER + 'Section/'
ALLEN_FOLDER = DATA_FOLDER + 'Allen/'

SCORES_FOLDER = PROJ_FOLDER + 'scores/' 
SRC_FOLDER = PROJ_FOLDER + 'src/registration/'
PICKLE_FOLDER = PROJ_FOLDER + 'pickle/'
TMP_FOLDER = PROJ_FOLDER + 'tmp/'

ZOOM = 5
SUBJ_TO_ATLAS_SCALING = 29.248 / 15.2

TX_RANGE =  np.arange(-100, 100, 10)
TY_RANGE = np.arange(-100, 100, 10)
THETA_RANGE = np.arange(-np.pi / 6, np.pi / 6, np.pi / 6 / 10)

#TX_RANGE =  np.arange(-100, 100, 5)
#TY_RANGE = np.arange(-100, 100, 5)
#THETA_RANGE = np.arange(-np.pi / 5, np.pi / 5, np.pi / 90)

# ndpi thresholds
MIN_AREA = 2000
SMOOTH_SIZE = 5
KERNEL_SIGMA = 5
#FAR_THRESH = 110
#FAR_THRESH = 40
SPLIT_STOP_STD = 7000
MORPH_ELEMENT_SIZE = 5
STRUCTURE_ELEMENT = cv2.MORPH_CROSS

# Debug options:
#SHOW_DETECT = True
SHOW_DETECT = False
#SHOW_MATCH = True
SHOW_MATCH = False
# preprocessing
DEBUG = False

#SIFT detect:
EDGE_THRESH = 50       # how large is the gradient; larger means more points, valid [1,+inf]
PEAK_THRESH = 0.5      # how high is the extrema in scale space; smaller means more points

#SIFT matching:
DIST_RATIO = 0.85       # the ratio of shortest vs. second descriptor distance, higher more
DIST_THRESH = 400       # descriptor distance threshold

#RANSAC parameters:
class TRANSFORM:
    RIGID, RIGID_SCALING, AFFINE = range(3)

MODEL = TRANSFORM.RIGID_SCALING
#MODEL = TRANSFORM.RIGID
MIN_MODEL_SAMPLES = 4     # the number of samples needed to computer the model
MAX_ITER = 2000            # maximum RANSAC iteration
INLIER_DIST_THRESH = 15   # the distance within which a point is counted as an inlier of a model
MIN_CONSENSUS_SIZE = 8    # the minimum number of points in consensus to output the model
#VAR_THRESH = 40             # deprecated, use VAR_THRESH_ADAPT instead
#ERR_THRESH = 4    # for Affine
#ERR_THRESH = 9    # for rigid
ERR_THRESH = 40     # for rigid, Allen vs. Harvey data 
TOTAL_VAR_RATIO = .75
TOTAL_CONSENSUS_RATIO = .5

#Relevant sections:
#RELEVANCY_THRESH = 5           # threshold of the mean error to identify the relevant reference section images
RELEVANCY_THRESH = 30           # for rigid
