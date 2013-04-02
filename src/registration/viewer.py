'''
Created on Mar 21, 2013

@author: yuncong
'''

import numpy as np
import cv2
from registration import scoring
from registration import score_reader

class StackViewer:
    def __init__(self, name, im_stack, i=0):
        self.im_stack = im_stack
        self.name = name
        self.i = i
        
    def show(self):
        cv2.namedWindow(self.name, cv2.cv.CV_WINDOW_AUTOSIZE)
        self.update(self.i)
        cv2.createTrackbar("section", self.name, self.i, len(self.im_stack)-1, self.update)
        cv2.waitKey()
        cv2.destroyAllWindows()
        
    def update(self, i):
        cv2.imshow(self.name, self.im_stack[i])

class StackCentroidViewer(StackViewer):
    def __init__(self, name, im_stack, i=0):
        self.name = name
        self.i = i
        self.im_stack = [scoring.pad_image_cnt(im) for im in im_stack]
        
class StackAtlasMatchViewer(StackViewer):
    def __init__(self, name, alnr, i=0, overlay=True):
        self.name = name
        self.i = i
        self.im_stack = [None]*alnr.num_subject
        for j, im in enumerate(alnr.subject_stack):
            if im is None: continue
            opt = score_reader.ScoreReader('scores_allen_%d'%j).opt
            allen_im = alnr.allen_stack[alnr.allen_match_id_stack_best[j]]
            allen_centroid = scoring.get_centroid(allen_im)
            im_centered = scoring.pad_image_cnt(im)
            im_t = scoring.transform_cnt(im_centered, None, opt,
                                         anchor_im='centroid', anchor_bg=allen_centroid)
            if overlay:
                opacity = 0.3;
                self.im_stack[j] = cv2.addWeighted(allen_im, opacity, im_t, 1 - opacity, 0)
            else:
                self.im_stack[j] = im_t
                   
class StackNeighborViewer(StackViewer):
    def __init__(self, name, alnr, i=0):
        self.name = name
        self.i = i
        self.im_stack = [None]*alnr.num_subject
        self.im_nobg_stack = [None]*alnr.num_subject
        for j, im in enumerate(alnr.subject_stack):
            if j==0 or j==alnr.num_subject-1: continue
            if im is None: continue
            im_centered = scoring.pad_image_cnt(im, None)
            if j>1:
                opt = score_reader.ScoreReader('scores_neighbor_%d'%(j)).opt    
                print j, opt[0], opt[1], opt[2], np.rad2deg(opt[2])
                prev_centroid = scoring.get_centroid(self.im_nobg_stack[j-1])
                self.im_nobg_stack[j] = scoring.transform_cnt(im_centered, None,opt,
                                                anchor_im='centroid', anchor_bg=prev_centroid)

                opacity = 0.3
                self.im_stack[j] = cv2.addWeighted(self.im_nobg_stack[j-1], opacity, 
                                                   self.im_nobg_stack[j], 1 - opacity, 0)
            else:
                prev_centroid = 'center'
                self.im_stack[j] = im_centered
                self.im_nobg_stack[j] = im_centered
            cv2.circle(self.im_nobg_stack[j], scoring.get_centroid(self.im_nobg_stack[j]),
                        5, 0)
        
#        self.im_stack = self.im_nobg_stack
        

class TransformViewer():
    def __init__(self, alnr, mode, id=0, name='TransformViewer'):
        self.name = name
        self.alnr = alnr
        self.x = 0
        self.y = 0
        self.theta = 0
        self.id = id
        self.mode = mode
        self.showImg = False
        self.load()
        
    def show(self):
        cv2.namedWindow(self.name, cv2.cv.CV_WINDOW_AUTOSIZE)
        while True:
            key = cv2.waitKey()
            if key != -1:
#                print key
                if chr(key)=='a':
                    self.x = self.x - 1
                elif chr(key)=='d':
                    self.x = self.x + 1
                elif chr(key)=='w':
                    self.y = self.y - 1
                elif chr(key)=='s':
                    self.y = self.y + 1
                elif chr(key)==']':
                    self.theta = self.theta - np.pi/180
                elif chr(key)=='[':
                    self.theta = self.theta + np.pi/180
                elif chr(key)=='=': # next
                    self.id = self.id + 1
                    self.load()
                elif chr(key)=='-': # previous
                    self.id = self.id - 1
                    self.load()
                elif chr(key)=='x': # switch modes
                    self.mode = 'allen' if self.mode=='neighbor' else 'neighbor'
                    self.load()
                    print 'switch to',self.mode,'mode'
                elif chr(key)==' ':
                    self.showImg = not self.showImg
                elif chr(key)=='q':
                    break
                else:
                    print 'no effect'
                    continue
                trfm = (self.x, self.y, self.theta)
                s,D = scoring.compute_score_cnt(self.mov, self.cnt_mov,
                                                self.ref, self.cnt_ref,
                                                'centroid', 'centroid', trfm,
                                                show=True, showImg=self.showImg,
                                                outputD=True, title=self.name)
                print 'id', self.id
                print 'pose', self.x, self.y, self.theta
                print 'score', s
                print ''
        cv2.destroyAllWindows()
    
    def load(self):
        self.mov = self.alnr.subject_stack[self.id]
        self.cnt_mov = self.alnr.subject_cnt_stack[self.id]
        
        if self.mode == 'neighbor':
            self.ref = self.alnr.subject_stack[self.id-1]
            self.cnt_ref = self.alnr.subject_cnt_stack[self.id-1]
        elif self.mode == 'allen':
            self.ref = self.alnr.allen_stack[self.alnr.allen_match_id_stack_best[self.id]]
            self.cnt_ref = self.alnr.allen_cnt_stack[self.alnr.allen_match_id_stack_best[self.id]]
                
        
if __name__ == '__main__':
    from registration import aligner
    alnr = aligner.Aligner('4')
    alnr.prepare_allen()
    alnr.prepare_subject()
    alnr.initial_shift()
    tv = TransformViewer(alnr, 'allen', id=2)
    tv.show()
    
#    alnr.prepare_allen_whole()
#    viewer = StackViewer('allen', alnr.allen_stack, i=1)
#    viewer.show()
    