import os, sys
import cv2
from registration import util, ransac, transform
from registration import config
import scipy
import numpy as np
import matplotlib.pyplot as plt

def detect(filename):
    im = cv2.imread(filename, 0)
    if im is None:
        raise ValueError('im is None')
    if np.ndim(im) == 3:
        im = cv2.cvtColor(im, cv2.cv.CV_RGB2GRAY)
    olddir = os.getcwd()
    os.chdir(config.TMP_FOLDER)
    imagename = 'tmp.pgm'
    cv2.imwrite(imagename, im, (cv2.cv.CV_IMWRITE_PXM_BINARY, 0))
    
    cmmd = str("/Users/yuncong/Documents/vlfeat-0.9.14/bin/maci64/sift " + 
               imagename + " --output=%.sift" + 
            " --edge-thresh=" + str(config.EDGE_THRESH) + " --peak-thresh=" + str(config.PEAK_THRESH) + 
            " --gss=%.pgm")
#             + "-vv")
#            --octaves=1 --levels=1 --first-octave=0")
    util.execute(cmmd)
    
    f = np.loadtxt('tmp.sift')
    keypoints, descriptors = np.array(f[:, :4]), np.array(f[:, 4:])
    print len(keypoints), 'keypoints found'
    
    if config.SHOW_DETECT:
        draw_keypoints(im, keypoints, 'b')
        plt.show()
    
    os.chdir(olddir)
    return keypoints, descriptors, filename

#        octave_number = 6
#        level_number = 3
#        for o in range(-1, octave_number-1):
#            for s in range(level_number):
#                gss_file = "tmp_0%d_00%d"%(o, s) if o>=0 else "tmp_%d_00%d"%(o, s)
#                print gss_file
#                gss = imread(gss_file, 0)
#                figure(); gray()
#                imshow(gss)
#                
#        dog = [[None]*level_number]*octave_number
#        dog[o][s] = gss[o][s+1] - gss[o][s]
#    
    

def match(d_from, d_to):
    kp_s, desc_s, name_from = d_from
    kp_d, desc_d, name_to = d_to
    
    if config.SHOW_MATCH:
        im_from = cv2.imread(name_from, 0)
        im_to = cv2.imread(name_to, 0)
    
    D = scipy.spatial.distance.cdist(desc_s, desc_d)

    sorted_idx = np.argsort(D, axis=1)
    idx_match_forward = [(r_idx, row[0]) for r_idx, row in enumerate(sorted_idx) 
                    if D[r_idx, row[0]] < config.DIST_RATIO * D[r_idx, row[1]]
                    and D[r_idx, row[0]] < config.DIST_THRESH]

    DT = D.T
    sorted_idx = np.argsort(DT, axis=1)
    idx_match_inv = [(r_idx, row[0]) for r_idx, row in enumerate(sorted_idx) 
                    if DT[r_idx, row[0]] < config.DIST_RATIO * DT[r_idx, row[1]]
                    and DT[r_idx, row[0]] < config.DIST_THRESH]

    idx_match = np.array([m for m in idx_match_forward if m[::-1] in idx_match_inv]) 
#    print idx_match
    
    n_match = idx_match.shape[0]
    print n_match, 'matches', 'after standout and score filtering'
    
    if config.SHOW_MATCH:
#        for i in range(n_match):
#            draw_match(im_from, kp_s[idx_match[i:i+1, 0]], im_to, kp_d[idx_match[i:i+1, 1]])
#            show()
        draw_match(im_from, kp_s[idx_match[:, 0]], im_to, kp_d[idx_match[:, 1]])
    
    if n_match > 0:
        pt_match = np.hstack([kp_s[idx_match[:, 0].T, :2], kp_d[idx_match[:, 1].T, :2]])
    else:
        raise ValueError('Failed at standout filtering')
            
    try:
        A, inlier, err = transform.transform_ransac(pt_match[:, :2], \
                                                pt_match[:, 2:], ransac.Model(config.MODEL))
        idx_match = idx_match[inlier]
#        print idx_match
    except ValueError as e:
        print >> sys.stderr, "Failed at RANSAC:", e
        return None, None, None
    
    print len(inlier), 'matches', 'after RANSAC'
    if config.SHOW_MATCH:
        draw_match(im_from, kp_s[idx_match[:, 0]], im_to, kp_d[idx_match[:, 1]])

    print 'err:', err
    print 'A:', A
    
    if config.SHOW_MATCH:
        draw_registration(im_from, A, im_to)
    
    return A, idx_match, err

def get_matching(d_from, d_to):
    try:
        A, idx_match, err = match(d_from, d_to)
        kp_from, desc_from, imname_from = d_from
        kp_to, desc_to, imname_to = d_to
        return A, kp_from[idx_match[:, 0]], desc_from[idx_match[:, 0]], \
             kp_to[idx_match[:, 1]], desc_to[idx_match[:, 1]], err
    except (ValueError, TypeError) as e:
        print e
        return None
        

def draw_circle(c, r, color):
        t = np.arange(0, 1.01, .01) * 2 * np.pi
        x = r * np.cos(t) + c[0]
        y = r * np.sin(t) + c[1]
        plt.plot(x, y, color, linewidth=2)


def draw_keypoints(im, kp, colors):
    if type(im) is str:
        print im
        im = cv2.imread(im, 0)

#    figure(); 
    plt.gray(); plt.axis('off')
#    im_circle = im[:]
#    cv2.imshow('SIFT Keypoints', im_circle)
    plt.imshow(im)
    for i in range(len(kp)):
        c = colors[i % len(colors)]
        kp0 = kp[i]
        larger_size = 1 * kp0[2]
        draw_circle(kp0[:2], larger_size, c)
        plt.plot([kp0[0], kp0[0] + larger_size * np.cos(kp0[3])],
             [kp0[1], kp0[1] + larger_size * np.sin(kp0[3])], c)
#    show()

def draw_match(im_from, kp_from, im_to, kp_to):    
    colors = 'rgbkcym'
    plt.figure()
    plt.subplot(121)
    draw_keypoints(im_from, kp_from, colors)
    plt.subplot(122)
    draw_keypoints(im_to, kp_to, colors)
    plt.show()

def draw_overlap(im_from, im_to):
    im_overlay = cv2.addWeighted(im_from, 0.5, im_to, 0.5, 0)
    plt.figure(); plt.gray(); plt.axis('off')
    plt.imshow(im_overlay)
    plt.show()
    
def draw_registration(im_from, A, im_to):
    if A is not None:
        im_from_warp = cv2.warpAffine(im_from, A, im_to.shape[::-1])
#        im_from_warp = cv2.warpAffine(im_from, A, (400,400))
        draw_overlap(im_from_warp, im_to)

            
if __name__ == '__main__':

    image_from_name = '4_10_clean.tif'
    image_to_name = 'dataset_100048576_458_100960165_4_clean.jpg'
            
#    dataset_id = 100048576
#    image_to_name = allen.get_filename(dataset_id, 477)
#
#    stack_id = 4
#    sec_num = 10
#    image_from_name = '{0}_{1}.tif'.format(stack_id, sec_num)
#    image_from_name = SECTION_FOLDER + image_from_name 
#
    d_from = detect(config.SECTION_FOLDER + image_from_name)
    d_to = detect(config.ALLEN_FOLDER + image_to_name)
    
#    im_from = cv2.imread(image_from_name, 0)
#    im_to = cv2.imread(image_to_name, 0)
    
#    h_from, w_from = im_from.shape[:2]
#    
#    W = 1000
#    H = 1000
#    
#    def pad_image(im, origin_offset):
#
#        L = 200
#        U = 200
#        h, w = im.shape[:2]
#        left_offset, up_offset = origin_offset
#        
#        left_padding = L - left_offset
#        right_padding = W - w - left_padding
#        upper_padding = U - up_offset
#        lower_padding = H - h - upper_padding
#        print left_padding,right_padding,upper_padding,lower_padding
#        im_padded = vstack([zeros([upper_padding, w + left_padding + right_padding]), \
#                                  hstack([zeros([h, left_padding]), im, zeros([h, right_padding])]), \
#                                  zeros([lower_padding, w + left_padding + right_padding])])
#        im_padded = im_padded.astype(uint8)
#        return im_padded
#    
#    for angle in range(0,360,36):
#        A = cv2.getRotationMatrix2D((w_from/2, h_from/2), angle, 1)
#        
#        cv2.imshow('', pad_image(im_from, (0,0)))
#        cv2.waitKey()
#        
#        im_from_warp = cv2.warpAffine(im_from, A, (W,H))
#        
#        vertices_warped = dot(A,array([[0,0,1],[w_from,0,1],[0,h_from,1],[w_from,h_from,1]]).T)
#        origin_offset = (min(vertices_warped[0]), min(vertices_warped[1]))
#        print origin_offset
#        
#        cv2.imshow('', pad_image(im_from_warp, origin_offset))
#        cv2.waitKey()
#        
#        im_to_padding = pad_image(im_to)
##        cv2.imshow('', im_to_padding)
##        cv2.waitKey()
#        
#    #    im_from = ndimage.interpolation.zoom(im_from, float32(im_to.shape)/im_from_warp.shape)
#        
#        pA, bins = histogram(im_from_warp, bins=range(256), density=True)
#        pB, bins = histogram(im_to_padding, bins=range(256), density=True)
#        HA = -sum(nan_to_num(pA * log(pA)))
#        HB = -sum(nan_to_num(pB * log(pB)))
#        
#        joint_histo = util.joint_histogram(im_from_warp, im_to_padding)
#    #    cv2.imshow('', joint_histo)
#    #    cv2.waitKey()
#        pAB = joint_histo / sum(joint_histo)
#        HAB = -sum(nan_to_num(pAB * log(pAB)))
#        
#        IAB = HA + HB - HAB
#        print HA, HB, HAB, IAB
#        
    A, idx_match, err = match(d_from, d_to)

#    show()
