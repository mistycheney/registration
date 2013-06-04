#! /bin/bash

PROJ=/home/yuncong/workspace/registration
SRC=$PROJ/src/registration
#OBJS=$SRC/aligner.py $SRC/allen.py $SRC/config.py $SRC/contour.py $SRC/scoring/py $SRC/util.py $SRC/viewer.py $SRC/preprocess.py

epydoc -v --html -o $PROJ/epydoc --name registration --url https://github.com/mistycheney/registration.git --docformat epytext --graph all --inheritance grouped $SRC
