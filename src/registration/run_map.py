#!/usr/bin/env python

'''
The main function for running the program on hadoop server
@author: yuncong
'''

import os, sys
from registration import aligner
import base64
import cv2
import numpy as np

def extract_index(filename):
    i = int(filename.split('/')[-1][2:-4])
    return i

def parse_line(line):
    line_split = line.split('\t',1)
    filename = line_split[0]
    
    content64 = line_split[1]
    content = base64.b64decode(content64,'-_')
    content_bytes = np.asarray(bytearray(content), dtype=np.uint8)
    img = cv2.imdecode(content_bytes, 0)
    return filename, img
    
if __name__ == '__main__':
    alnr = aligner.Aligner('4','p56_coronal')
    alnr.prepare_allen()
    alnr.initial_shift()

    for line in sys.stdin:    
        filename, img = parse_line(line)
        i = extract_index(filename)
        
        alnr.prepare_subject_each_img(i, img)
        alnr.optimize_atlas_map(i)
        if i > 1:
            alnr.optimize_neighbor_map(i)

    
