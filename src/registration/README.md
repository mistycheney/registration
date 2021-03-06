**src/registration**: source code for the registration module

* **aligner.py**: Aligner class. Main class for registration, provides method for managing the optimization workflow.
* **allen.py**: Functions for downloading and querying the Allen atlas images
* **config.py**: Configuration variables
* **contour.py**: Functions for extracting contours and applying image transformations
* **preprocess.py**: Functions for segmenting multi-section slides
* **scoring.py**: ScoreReader Class, with function for computing Hessians
* **util.py**: Utility functions for plotting, etc.
* **viewer.py**: Various viewer classes.
* **run.py**: Main function for doing the alignment locally.
* **run_map.py**: Mapper for doing the alignment using hadoop streaming on the cluster.
* **segment_map.py**: Mapper for segmenting multi-section slides using hadoop streaming on the cluster.
