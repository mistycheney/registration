import cv2

DATA_FOLDER = '/Users/yuncong/Documents/medical_images/'
SLIDE_FOLDER = DATA_FOLDER + 'Slide/'
SECTION_FOLDER = DATA_FOLDER + 'Section/'
ALLEN_FOLDER = DATA_FOLDER + 'Allen/'

PROJ_FOLDER = '/Users/yuncong/Documents/workspace/Registration/'
SRC_FOLDER = PROJ_FOLDER + 'src/registration/'
PICKLE_FOLDER = PROJ_FOLDER + 'pickle/'
TMP_FOLDER = PROJ_FOLDER + 'tmp/'

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
