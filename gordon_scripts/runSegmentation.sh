#!/bin/bash

shopt -s expand_aliases

INPUT=$1
OUTPUT=$2
MAPPER=$3

hdfs -rmr $OUTPUT

PROJ=/oasis/scratch/csd181/yuncong/registration/src/registration/
rm -f $PROJ/../../scores/*.*

had jar /opt/hadoop/contrib/streaming/hadoop-*streaming*.jar \
	-libjars /oasis/scratch/csd181/yuncong/segmentation/ProcessImage.jar \
	-files $PROJ/allen_match_id_stack_best.p -files $PROJ/allen_cnt_stack.p\
	-D mapred.map.tasks=100 \
	-D mapred.reduce.tasks=1 \
	-D mapred.child.java.opts=-Xmx1800m \
	-D mapred.reduce.slowstart.completed.maps=0.2 \
	-D mapred.map.tasks.speculative.execution=true \
	-D mapred.map.max.attempts=3 \
	-D mapred.reduce.max.attempts=3 \
	-file $MAPPER \
	-mapper $MAPPER \
	-input $INPUT -inputformat yuncong.WholeFileInputFormatOld \
	-output $OUTPUT \
	-cmdenv PYTHONPATH=/home/yuncong/opencv/release/lib:$PYTHONPATH \
	-cmdenv PATH=/oasis/scratch/csd181/yuncong/myVirtualEnv/bin:$PATH \
	-cmdenv LD_LIBRARY_PATH=/home/yuncong/opencv/release/lib:$LD_LIBRARY_PATH
