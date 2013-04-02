#!/usr/bin/env python

import os, sys
import base64
import cv2
import numpy as np
from registration import config

def extract_index(filename):
    i = int(filename.split('/')[-1][2:-4])
    return i

def str2im(content64):
    content = base64.b64decode(content64,'-_')
    content_bytes = np.asarray(bytearray(content), dtype=np.uint8)
    img = cv2.imdecode(content_bytes, -1)
    return img

def im2str(img):
    retval, buf = cv2.imencode('.tif', img)
    s = bytearray(buf)
    s64 = base64.b64encode(s, '-_')
    return s64

def parse_line(line):
    line_split = line.split('\t',1)
    filename = line_split[0]
    img = str2im(line_split[1])
    return filename, img

def txt2dict():
    seg_name = config.DATA_FOLDER +'slide_seg_info.txt'
    seg_file = open(seg_name,'r')
    seg_lines = seg_file.readlines()
    seg_file.close()
    seg_info = dict([])
    for l in seg_lines:
        fn,x,y,w,h,i = l.strip().split()
        if fn not in seg_info:
            seg_info[fn] = dict([])
        else:
            seg_info[fn][int(i)-1] = (int(x),int(y),int(w),int(h))
    return seg_info

if __name__ == '__main__':
    for line in sys.stdin:    
        filename, img = parse_line(line)
        seg_info = txt2dict()
        fn = filename.split('/')[-1][:-4]
        subj, stain, slide, zoom, z0tif = fn.split('_')
        for i, (x,y,w,h) in seg_info[fn].iteritems():
            fn_out = '_'.join([subj, stain, zoom, str(i)])
#        sys.stderr.write(','.join([x,y,h,w,str(img.shape[0]),str(img.shape[1])]))

            scaling = config.ZOOM/5
            y = int(scaling*y)
            x = int(scaling*x)
            w = int(scaling*w)
            h = int(scaling*h)

            sub_im = img[y:y+h, x:x+w, :]
            cv2.imwrite('/oasis/scratch/csd181/yuncong/tif_x5_seg/' + 
                    fn_out+'.tif', sub_im)
#            print '%s\t%s'%(fn_out,im2str(sub_im))
        
