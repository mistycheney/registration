'''
Created on Mar 21, 2013

@author: yuncong
'''
import os, sys
import cPickle as pickle

from registration import config
import cv2
import numpy as np
from matplotlib import mlab
import scipy.spatial

def find_large_contours(thresh_image, min_area=config.MIN_AREA):
    '''
    return only the "large" contours.
    '''
    thresh_image[:, 0] = 0
    thresh_image[:, -1] = 0
    thresh_image[0, :] = 0
    thresh_image[-1, :] = 0
    
    h, w = tuple(thresh_image.shape[:2])
    contours, hierarchy = cv2.findContours(thresh_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    all_bbox = [cv2.boundingRect(i) for i in contours]
    all_area = np.array([cv2.contourArea(c) for c in contours])
    all_aspect_ratio = np.array([float(b[2]) / b[3] for b in all_bbox])
    large = mlab.find((all_area > min_area) & (all_area < h * w * 0.8) & (all_aspect_ratio < 10))
    large_contours = [contours[i] for i in large]
        
#    print len(large_contours), 'large_contours'

    mask = np.zeros((h, w, 1), np.uint8)
    cv2.drawContours(mask, large_contours, -1, 255, -1)
#        cv2.imshow('', mask[:, :, 0])
#        cv2.waitKey()
    return large_contours, mask[:, :, 0]

def clean(img, bb=False, white_bg=True):
    '''
    clean a slide image, so that only the interesting parts are kept
    '''
#        from scipy import ndimage
    
#    cv2.imshow('',img)
#    cv2.waitKey()
    img_smooth = cv2.medianBlur(img, 5)
    img_smooth = cv2.GaussianBlur(img_smooth, (3, 3), 3)
    
    h, w = img.shape
    markers = np.zeros_like(img).astype(np.int16)
    markers[5, 5] = 1
    
    flooded = img_smooth.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    connectivity = 8
#    flags = connectivity | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY
    flags = connectivity | cv2.FLOODFILL_FIXED_RANGE
    seed_pt = 5, 5
    loDiff = 5
    hiDiff = 255
    retval, rect = cv2.floodFill(flooded, mask, seed_pt, (255, 255, 255), loDiff, hiDiff, flags)
    mask = 1 - mask[1:-1, 1:-1]
    large_contours, mask = find_large_contours(mask)
    
    if bb:
        rows, cols = np.nonzero(mask)
        left = min(cols); right = max(cols); top = min(rows); bottom = max(rows)
        mask = mask[top:bottom, left:right]
        img = img[top:bottom, left:right]
        large_contours, mask = find_large_contours(mask * 255)
    
    if white_bg:
        white = 255 * np.ones(mask.shape, np.uint8)
        img_clean = (white - mask) + (img & mask)
    else:
        img_clean = img & mask
#    cv2.imshow('img_clean', img_clean)
#    cv2.waitKey()
    return img_clean, large_contours

    
def draw_contours(cnts, bg_size, fill=False, show=False, title='', color=255):
    """
    A utility function that returns a white-background image showing the contour given by cnts.
    """
    if cnts.ndim < 3:
        cnts = np.resize(cnts, (cnts.shape[0], 1, 2))
    
    h = bg_size[1]
    w = bg_size[0]
#    if len(color) == 3:
    vis = np.zeros((h, w, 3), np.uint8)
#    else:
#        vis = np.zeros((h, w, 1), np.uint8)
    if fill:
        cv2.drawContours(vis, cnts, -1, color, -1)
    else:
        cv2.drawContours(vis, cnts, -1, color, 2)
    if show:
        cv2.imshow(title, vis)
        cv2.waitKey()
    return vis

def pad_image_cnt(img, cnt=None, anchor_im='centroid', anchor_bg='center', bg=(800, 800)):
    """
    Place img on a background canvas, by aligning the anchor_im position on the img 
    with the anchor_bg position on the canvas. Return the canvas with image on it.
    """
    bg_h, bg_w = bg
    if anchor_im == 'centroid':
        anchor_im = get_centroid(img)
    elif anchor_im == 'center':
        anchor_im = img.shape[1] / 2, img.shape[0] / 2
    if anchor_bg == 'center':
        anchor_bg = bg_w / 2, bg_h / 2
    
    anchor_x, anchor_y = anchor_bg
    center_x, center_y = anchor_im
    
    h, w = img.shape[0:2]
    background = np.zeros((bg_h, bg_w)).astype(np.uint8)
    background[max(anchor_y - center_y, 0):anchor_y + h - center_y,
               max(anchor_x - center_x, 0):anchor_x + w - center_x] = img[max(0, center_y - anchor_y):min(h, center_y + bg_h - anchor_y),
                max(0, center_x - anchor_x):min(w, center_x + bg_w - anchor_x)]

    if cnt is not None:
        cnt = cnt + np.matlib.repmat(np.array(anchor_bg) - np.array(anchor_im), cnt.shape[0], 1)
        return background, cnt
    return background

import time

def shape_score_cnt(cnt1_t, cnt2_t, outputD=False, title='shape_score_cnt', 
                    variant=0):
    D = scipy.spatial.distance.cdist(cnt1_t, cnt2_t, 'euclidean')
    if variant == 0:
        k = 50
        d1 = np.sort(D.min(0))
        d2 = np.sort(D.min(1))
        hABk = d1[-k]
        hBAk = d2[-k]
        HAB = max(hABk, hBAk)
        score = HAB
    elif variant == 1:
        d1 = np.sort(D.min(0))
        d2 = np.sort(D.min(1))
        hABk = d1[-int(len(d1)/10)]
        hBAk = d2[-int(len(d1)/10)]
        HAB = max(hABk, hBAk)
        score = HAB
    elif variant == 2:
        d1 = np.sort(D.min(0))
        d2 = np.sort(D.min(1))
        hABk = np.mean(d1[:int(len(d1)*0.9)])
        hBAk = np.mean(d2[:int(len(d2)*0.9)])
        HAB = max(hABk, hBAk)
        score = HAB
    if outputD:
        return score, D
    else:
        return score

def compute_score_cnt(img, cnt, img_ref, cnt_ref, 
                      anchor_im='centroid', anchor_ref='centroid', trfm=(0,0,0), 
                      show=False, showImg=False, outputD=False, title='compute_score_cnt',
                      variant=0):
    """
    Align the centroids of both images and apply transform around the centroid of the first image, 
    then compute score. 
    """
    img, cnt = pad_image_cnt(img, cnt, anchor_im)
    img_ref, cnt_ref = pad_image_cnt(img_ref, cnt_ref, anchor_ref)
    img_warp, cnt_t = transform_cnt(img, cnt, trfm, 'center')
    
    if show:
        vis1 = draw_contours(cnt_t, (800, 800), show=False, color=(0,0,255))
        vis2 = draw_contours(cnt_ref, (800, 800), show=False, color=(0,255,0))
        vis = cv2.addWeighted(vis1, 0.5, vis2, 0.5, 0)
        if showImg:
            img_warp_color = cv2.cvtColor(img_warp, cv2.cv.CV_GRAY2RGB)
            img_ref_color = cv2.cvtColor(img_ref, cv2.cv.CV_GRAY2RGB)
            visImg = cv2.addWeighted(img_warp_color, 0.5, img_ref_color, 0.5, 0)
            vis = cv2.addWeighted(vis, 1, visImg, 1, 0)
        cv2.imshow(title, vis)
#        cv2.waitKey()
    if outputD:
        score, D = shape_score_cnt(cnt_t, cnt_ref, outputD, title,variant=variant)
        return score, D
    else:
        score = shape_score_cnt(cnt_t, cnt_ref, outputD, title,variant=variant)
        return score

def compute_scores_cnt(img, cnt, img_ref, cnt_ref, tx_range, ty_range, theta_range):
    nx = len(tx_range)
    ny = len(ty_range)
    nt = len(theta_range)
    scores = np.zeros((nx,ny,nt))
    for i, tx in enumerate(tx_range):
        for j, ty in enumerate(ty_range):
            for k, theta in enumerate(theta_range):
                trfm = np.array([tx,ty,theta])
                scores[i,j,k] = compute_score_cnt(img, cnt, img_ref, cnt_ref,
                                                  'centroid', 'centroid', trfm)
    return scores

def get_centroid(img):
    """
    Return the centroid of the image (treated as binary)
    """
    m = cv2.moments(img, binaryImage=True)
    centroid = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
    return centroid
    

def transform_cnt(img, cnt, trfm, anchor_im='centroid', anchor_bg='center'):
    """
    Place img on the background canvas, by aligning the anchor_im position on the img 
    with the anchor_bg position of the background canvas, and then apply trfm 
    on the image (with rotation centered around centroid of the image). Return the 
    canvas with transformed image.
    """
    if cnt is None:
        img = pad_image_cnt(img, cnt, anchor_im, anchor_bg)
    else:
        img, cnt = pad_image_cnt(img, cnt, anchor_im, anchor_bg)
    im_centroid = get_centroid(img)
    A = cv2.getRotationMatrix2D(im_centroid, np.rad2deg(trfm[2]), 1)
    A[0, 2] = A[0, 2] + trfm[0]
    A[1, 2] = A[1, 2] + trfm[1]
    img_warp = cv2.warpAffine(img, A, img.shape[0:2])
    if cnt is not None:
        cnt = cv2.transform(np.resize(cnt, (cnt.shape[0], 1, 2)), A)
        cnt = np.squeeze(cnt)
        return img_warp, cnt
    return img_warp
    
def transformA(img, A, anchor_im='centroid', anchor_bg='center'):
    img = pad_image_cnt(img, None, anchor_im, anchor_bg)
    img_warp = cv2.warpAffine(img, A, img.shape[0:2])
    return img_warp
    
def get_contour_sample(img_orig, sample_interval=5):
    """
    Return contour sample points of a image
    """
    img = img_orig.copy()
    img[img > 10] = 255
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    mask = pad_image_cnt(img, None, (0, 0), (300, 300), bg=(1500,1500))
#    print mask
    for j in range(99):
        mask = cv2.dilate(mask, element)
#        cv2.imshow("mask_dilate", mask)
#        cv2.waitKey()
        cnts, hier = cv2.findContours(mask.copy(), cv2.cv.CV_RETR_EXTERNAL, 
                                      cv2.cv.CV_CHAIN_APPROX_TC89_L1)
        if len(cnts) == 1:
            break
#    cv2.imshow("mask_dilate", mask)
#    cv2.waitKey()
    
    mask = cv2.erode(mask, element, iterations=j+1)
#    cv2.imshow("mask", mask)
#    cv2.waitKey()
#    cv2.destroyWindow("mask")
    cnts, hier = cv2.findContours(mask.copy(), cv2.cv.CV_RETR_EXTERNAL, 
                                  cv2.cv.CV_CHAIN_APPROX_NONE)
    img_contour_pts = np.squeeze(np.vstack(cnts))
    img_contour_sample = img_contour_pts[range(0, img_contour_pts.shape[0], sample_interval), :]
    img_contour_sample = img_contour_sample - np.matlib.repmat((300,300),
                                                               img_contour_sample.shape[0], 1)
#    draw_contours(img_contour_sample, (800,800), show=True)
    return img_contour_sample
    
    
    
    
    
    
    
    
    
