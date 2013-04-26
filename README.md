Mouse Brain Registration
============
This is the repo for "high resolution mouse brain registration" project.
Current implementation is based on section contour matching.

Wiki Page
--------
http://seed.ucsd.edu/mediawiki/index.php/High-resolution_Mouse_Brain_Section_Images_Registration

Evernote
--------
https://www.evernote.com/pub/mistycheney/brain

Reports/Presentations
----------------------
https://www.dropbox.com/sh/quhklvnr8o5m0qk/p-UiWtPVLq

Dataset
--------
Decompressed images are stored on Gordon `/oasis/scratch/csd181/yuncong/tif`

Github Repo
-----------
https://github.com/mistycheney/registration

* **data**:           contains the atlas images and sample subject images
 * **Allen**:         atlas images
 * **Section**:       subject images
* **gordon_scripts**: contains bash scripts used on the cluster (for running hadoop streaming)
 * **runSegmentation.sh** hadoop streaming script for segmenting multi-section slides
 * **runAlign.sh**        hadoop streaming script for doing the alignment
* **local_scripts**:  contains bash scripts used in local machine
 * **downloadScores.sh**  bash script for downloading scores pickle files from Gordon
* **pickle**:         stores pickle files
* **epydoc**:         api documentation generated by epydoc (see index.html)
* **java_classes**:   customized InputFormat classes
* **src/registration**: source code for the registration module
 * **aligner.py**:    Aligner class. Main class for registration, provides method for managing the optimization workflow.
 * **allen.py**:      Functions for downloading and querying the Allen atlas images
 * **config.py**:     Configuration variables
 * **contour.py**:    Functions for extracting contours and applying image transformations
 * **preprocess.py**: Functions for segmenting multi-section slides
 * **scoring.py**:    ScoreReader Class, with function for computing Hessians
 * **util.py**:       Utility functions for plotting, etc.
 * **viewer.py**:     Various viewer classes.
 * **run.py**:     	  Main function for doing the alignment locally.
 * **run_map.py**:    Mapper for doing the alignment using hadoop streaming on the cluster.
 * **segment_map.py**:    Mapper for segmenting multi-section slides using hadoop streaming on the cluster.

* **slide_seg_info.txt**: text file containing information for multi-section slide segmentation, each row specifies the x,y,width,height of the bounding box for a section. This is created manually.



