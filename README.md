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
* **gordon_scripts**: contains bash scripts used on the cluster (for running hadoop streaming)
* **local_scripts**:  contains bash scripts used in local machine
* **pickle**:         stores pickle files
* **epydoc**:         api documentation generated by epydoc
* **src**:            source code for the registration module
* **slide_seg_info.txt**: text file containing information for multi-section slide segmentation, each row specifies the x,y,width,height of the bounding box for a section. This is created manually.



