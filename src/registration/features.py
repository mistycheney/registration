import os, sys
from pylab import * 
import cv2

if __name__ == '__main__':
    img = imread('/Users/yuncong/Documents/medical_images/Section/4/4_10.tif')
    img_gaussian = cv2.GaussianBlur(img, (5,5), 1)
    img_log = cv2.Laplacian(img_gaussian, 5)
    img_min = img_log.min().min()
    img_max = img_log.max().max()
    img_log = (img_log - img_min)*255/(img_max-img_min)
    print uint8(img_log[1])

    thresh = 5
    img_zero = 255*uint8((img_log > 127-thresh) & (img_log < 127+thresh))
    print img_zero[1]

    cv2.imshow('image', uint8(img_log))
    cv2.imshow('image_zero', img_zero)
    # cv2.imshow('image_log', img_log)
    
    cv2.waitKey()
