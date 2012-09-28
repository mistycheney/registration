import os, sys
from pylab import *
import cv2
from registration import allen, sift, util
from registration.config import *

def detect_keypoints(stack_id, sections):
    imnames = [SECTION_FOLDER + str(stack_id) + '/' + str(stack_id) + '_' + str(sec) + '.tif' for sec in sections]
    print imnames
    d = [sift.detect(name) for name in imnames]
    return d

def find_transforms(d):
    nsection = len(d)
    err = [None] * nsection
    A = [None] * nsection
    A[0] = eye(3)
    skip = []
    d_from = d[1]
    d_to = d[0]
    for sec in range(1, nsection):
        print 'section', sec 
        A[sec], idx_match, err[sec] = sift.match(d_from, d_to)
        if A[sec] is not None:
            A[sec] = vstack([A[sec], [0, 0, 1]])
            d_to =  d_from
            if sec+1 < nsection:
                d_from = d[sec+1]
        else:
            skip.append(sec)
            if sec+1 < nsection:
                d_from = d[sec+1]
              
    return A, err, skip

def accumulate_transforms(d, A, reverse=False):
    H = [None] * len(A)
    ims = [None] * len(A)
    a_accum = eye(3)
    if reverse:
        A = [inv(a) if a is not None else None for a in A]
        for i, a in reversed(list(enumerate(A))):
            if a is not None:
                a_accum = dot(a_accum,a)
                H[i] = a_accum
                ims[i] = cv2.imread(d[i][2], 0)
    else:
        for i, a in enumerate(A):
            if a is not None:
                a_accum = dot(a_accum,a)
                H[i] = a_accum
                ims[i] = cv2.imread(d[i][2], 0) 
    H_valid = [h for h in H if h is not None]
    
    if None not in H:
        if reverse:
            ims_reg = [cv2.warpAffine(ims[k], H[k][:2, :], ims[-1].shape[::-1]) for k in reversed(range(len(H)))]
        else:
            ims_reg = [cv2.warpAffine(ims[k], H[k][:2, :], ims[0].shape[::-1]) for k in range(len(H))]
    else:
        raise ValueError('None in H')
        
    return H_valid, ims_reg

if __name__ == '__main__':    
    nsections = {3:0, 4:40, 5:32, 6:34, 9:28, 10:80}
    stack_id = 4
    nsection = nsections[stack_id]
    start_sec = 0

#    stack_id = 10
#    nsection = 50
#    start_sec = 30

    sections = range(start_sec, start_sec + nsection)
    
    d = util.conditional_load('_'.join(['d', 'stack',str(stack_id), 'start', str(start_sec)]), \
                            detect_keypoints, (stack_id, sections), regenerate=True)
    
    A, err, skip = util.conditional_load('_'.join(['A', 'model', str(MODEL), 'stack', str(stack_id),\
                                                    'start', str(start_sec)]), \
                                   find_transforms, [d], regenerate=True)
    
    print 'skipped:', skip
    
    #    k = 5; sift.draw_registration(ims[k], H[k][:2, :], ims[0])

    H, ims_reg = accumulate_transforms(d, A,reverse=False)
        
    util.Plotter('Stack '+str(stack_id), ims_reg, 0)
    
#    im_reg =[None] * nsections
#    for i,im in enumerate(ims):
#        print i
#        im_accum = im
#        for j, a in reversed(list(enumerate(A[:i+1]))):
#            print j
#            im_accum = cv2.warpAffine(im_accum, a[:2,:], ims[max(0,j-1)].shape[::-1])
##            sift.draw_overlap(im_accum, ims[max(0,j-1)])
#        im_reg[i] = im_accum
#        
#    import plotter
#    plotter.plotter(im_reg, 0)
#    show()

#    sift.draw_overlap(im_reg[10], im_reg[0])
    
#    print [i for i,a in enumerate(A) if a is None]

#16 (reduce inlier thresh to 5),

#27,28(points too few),29(points too few)
#21(up var thresh),
#22(up inliner),
#23(var, inlier),
#24(error too large),


#    for sec in range(1,len(A)):
#        if A[sec] is not None:
#            print sec
#            sift.draw_registration(d[sec][0], A[sec][:2,:], d[sec - 1][0])
#            show()

