'''
Aligner class.
Main class for registration, provides method for managing the optimization workflow.
'''

# from subprocess import call
import os, sys
import cPickle as pickle

from registration import config
import cv2
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from matplotlib import mlab

from registration.contour import * 
from registration import scoring
import time

class Aligner:
    def __init__(self, subject_id, num_subject):
        self.subject_id = subject_id
        self.scaling = config.SUBJ_TO_ATLAS_SCALING
        self.num_subject = num_subject
        self.subject_stack = [None] * self.num_subject
        self.subject_cnt_stack = [None] * self.num_subject
        
        self.allen_stack = dict([])
        self.allen_cnt_stack = dict([])

        self.tx_range = config.TX_RANGE
        self.ty_range = config.TY_RANGE
        self.theta_range =  config.THETA_RANGE
        
        self.dSS = dict([])
        self.dSA = dict([])
        self.dA = dict([])
        self.HSS = dict([])
        self.HSA = dict([])
        self.dGlobal = dict([])
    
    def prepare_allen(self):
        sys.stderr.write ('prepare_allen...',)
        begin = time.time()
        from registration import allen
        info = allen.query_dataset(100048576)
        
        try:
            self.allen_cnt_stack = pickle.load(open(config.SRC_FOLDER + "allen_cnt_stack.p","rb"))
        except:
            pass
        
        all_warp_name = os.listdir(config.ALLEN_FOLDER + '100048576_warp/')
        for im_id, im_info in info['section_images'].iteritems():
            if im_info['filename'] in all_warp_name:
                img = cv2.imread(config.ALLEN_FOLDER + '100048576_warp/' + im_info['filename'], 0) 
            else:
                img = cv2.imread(config.ALLEN_FOLDER + '100048576/' + im_info['filename'], 0)
                img_clean, _ = clean(img, white_bg=False)
                A = im_info['alignment2d']
                downsampling = 4
                A[:, 2] = A[:, 2] / 2 ** (downsampling - 0)
                #        print A
                img = transformA(img_clean, A, (0, 0), (0, 0))
                cv2.imwrite(config.ALLEN_FOLDER + '100048576_warp/' + im_info['filename'], img)
            
            xc,yc = get_centroid(img)
            self.dA[im_id] = np.array([xc, yc, 0])

            self.allen_stack[im_id] = img
            if im_id not in self.allen_cnt_stack: 
                self.allen_cnt_stack[im_id] = get_contour_sample(img)

#        pickle.dump(self.allen_cnt_stack, open("allen_cnt_stack.p","wb"))
        
        sys.stderr.write('%d seconds\n' % (time.time() - begin))
        
    def prepare_allen_whole(self):
        sys.stderr.write ('prepare_allen_whole...',)
        begin = time.time()
        from registration import allen
        info = allen.retrieve_specimens(5756)
        
        all_warp_name = os.listdir(config.ALLEN_FOLDER + '5756_warp/')
        for im_id, im_info in info['section_images'].iteritems():
            if im_info['filename'] in all_warp_name:
                img = cv2.imread(config.ALLEN_FOLDER + '5756_warp/' + im_info['filename'], 0)
            else:
                img = cv2.imread(config.ALLEN_FOLDER + '5756/' + im_info['filename'], 0)
                img_clean, _ = clean(img, white_bg=False)
                A = im_info['alignment2d']
                downsampling = 4
                A[:, 2] = A[:, 2] / 2 ** (downsampling - 0)
                #        print A        
                img = transformA(img_clean, A, (0, 0), (0, 0))
                cv2.imwrite(config.ALLEN_FOLDER + '5756_warp/' + im_info['filename'], img)
            
            img_contour_sample = get_contour_sample(img)
            self.allen_stack[im_id] = img
            self.allen_cnt_stack[im_id] = img_contour_sample
        sys.stderr.write('%d seconds\n' % (time.time() - begin))
    
    def prepare_subject_each_img(self, i, img):
        new_size = np.floor(self.scaling * np.array(img.shape))
        h, w = tuple(new_size.astype(np.int))
        img = cv2.resize(img, (w, h))
        self.subject_stack[i] = img
        if self.subject_cnt_stack[i] is None:
            self.subject_cnt_stack[i] = get_contour_sample(img)    

#    def timeit(func):
#        def wrapper(*arg,**kw):
#            t1 = time.time()
#            res = func(*arg,**kw)
#            t2 = time.time()
#            return (t2-t1),res,func.func_name
#    return wrapper

#    @timeit
    def prepare_subject(self):
        sys.stderr.write ('prepare_subject...'),
        begin = time.time()
        try:
            self.subject_cnt_stack = pickle.load(open(config.SRC_FOLDER + "subject_cnt_stack.p","rb"))
        except:
            pass
        for i in range(self.num_subject):
            img = cv2.imread(config.SECTION_FOLDER + '4/4_%d.tif' % i, 0)
            self.prepare_subject_each_img(i, img)
        sys.stderr.write('%d seconds\n' % (time.time() - begin))
                    
    def initial_shift(self):
        sys.stderr.write ('initial_shift...'),
        begin = time.time()
        try:
            self.allen_match_id_stack_best = pickle.load(open(config.SRC_FOLDER + 
                                        'allen_match_id_stack_best.p', 'rb'))
        except:
            min_score = 9999
            subject_interval = 90 
            allen_interval = 100
            z0_max = 13100
            z0_step = 50
            
            z0_min = (self.num_subject + 5) * subject_interval
            for z0 in range(z0_max, z0_min, -z0_step):  # 12200 should be the best
                total_score = 0
            
                allen_match_id_stack = [None] * self.num_subject
                for i in range(self.num_subject):
                    img = self.subject_stack[i]
                
                    img_z = z0 - subject_interval * i
                    allen_match_ind = np.round(img_z / allen_interval).astype(np.int)
#                    print i, z0, allen_match_ind
                    allen_match_id, allen_match_img = sorted(self.allen_stack.items())[allen_match_ind]
                    allen_match_id_stack[i] = allen_match_id
                    
            #            allen_score = compute_score(img, allen_match_img,'center','center',0, 0, 0)
                    allen_score = compute_score_cnt(img, self.subject_cnt_stack[i],
                                                    allen_match_img, self.allen_cnt_stack[allen_match_id],
                                                    'centroid', 'centroid', (0,0,0))
                    total_score += allen_score
                    
                if total_score < min_score:
                    min_score = total_score
                    self.allen_match_id_stack_best = allen_match_id_stack
        sys.stderr.write('%d seconds\n' % (time.time() - begin))
    
    def optimize_map(self, i):
        self.optimize_atlas_map(i)
        self.optimize_neighbor_map(i)
    
    def optimize_atlas_map(self, i):
        sys.stderr.write('optimize_atlas...'),
        begin = time.time()
        try:
            scores = pickle.load(open(config.SCORES_FOLDER + 'scores_allen_%d.p' % i, 'rb'))
        except:
            allen_match_id = self.allen_match_id_stack_best[i]
            allen_match_img = self.allen_stack[allen_match_id]
            allen_match_cnt = self.allen_cnt_stack[allen_match_id]    
            scores = compute_scores_cnt(self.subject_stack[i], self.subject_cnt_stack[i],
                                    allen_match_img, allen_match_cnt,
                                    self.tx_range, self.ty_range, self.theta_range)
            pickle.dump(scores, open(config.SCORES_FOLDER + 'scores_allen_%d.p' % i, 'wb'))
            
        sr = scoring.ScoreReader(scores)
        self.dSA[i] = sr.opt
        try:
            _, self.HSA[i] = sr.compute_hessian(h=1)
        except:
            self.HSA[i] = None
        
        sys.stderr.write ('%d seconds\n' % (time.time() - begin))

    def optimize_atlas(self):
        for i in range(self.num_subject):
#            sys.stderr.write(str(i))
            self.optimize_atlas_map(i)
            

    def optimize_neighbor_map(self, i):
        sys.stderr.write('optimize_neighbor...'),
        begin = time.time()
        try:
            scores = pickle.load(open(config.SCORES_FOLDER + 'scores_neighbor_%d.p' % i, 'rb'))
        except:
#        sys.stderr.write(config.SECTION_FOLDER+'4/4_'+str(i+1)+'.tif')

            img_prev = cv2.imread(config.SECTION_FOLDER + '4/4_' + str(i-1) + '.tif', 0)
            new_size = np.floor(self.scaling * np.array(img_prev.shape))
            h, w = tuple(new_size.astype(np.int))
            img_prev = cv2.resize(img_prev, (w, h))
            cnt_prev = get_contour_sample(img_prev)

            scores = compute_scores_cnt(self.subject_stack[i], self.subject_cnt_stack[i],
                                    img_prev, cnt_prev,
                                    self.tx_range, self.ty_range, self.theta_range)
            pickle.dump(scores, open(config.SCORES_FOLDER + 'scores_neighbor_%d.p' % i, 'wb'))
        sys.stderr.write ('%d seconds\n' % (time.time() - begin))

    def optimize_neighbor_each(self, i):
        sys.stderr.write('optimize_neighbor...'),
        begin = time.time()
        try:
            scores = pickle.load(open(config.SCORES_FOLDER + 'scores_neighbor_%d.p' % i, 'rb'))
        except:
            scores = compute_scores_cnt(self.subject_stack[i], self.subject_cnt_stack[i],
                                    self.subject_stack[i-1], self.subject_cnt_stack[i-1],
                                    self.tx_range, self.ty_range, self.theta_range,
                                    show=False)
            pickle.dump(scores, open(config.SCORES_FOLDER + 'scores_neighbor_%d.p' % i, 'wb'))
            
        sr = scoring.ScoreReader(scores)
        self.dSS[i] = sr.opt
        try:
            _, self.HSS[i] = sr.compute_hessian(h=1)
        except:
            self.HSS[i] = None
        
        sys.stderr.write ('%d seconds\n' % (time.time() - begin))

    def optimize_neighbor(self):
        for i in range(1, self.num_subject):
#            sys.stderr.write (str(i))
            self.optimize_neighbor_each(i)
            
    def global_optimzation(self):
        A = scipy.sparse.lil_matrix((3 * (2 * self.num_subject - 1), 3 * self.num_subject))
        b = np.zeros((3 * (2 * self.num_subject - 1), 1))
        for i in range(0, self.num_subject):
            GL = self.HSA[i]
            A[3 * i:3 * (i + 1), 3 * i:3 * (i + 1)] = GL
            b[3 * i:3 * (i + 1)] = np.dot(GL, np.atleast_2d(
                        self.dA[self.allen_match_id_stack_best[i]] - self.dSA[i]).T)
            if i > 0:
                HL = self.HSS[i]
                A[3 * self.num_subject + 3 * (i - 1): 3 * self.num_subject + 3 * i,
                   3 * (i - 1):3 * i] = -HL
                A[3 * self.num_subject + 3 * (i - 1): 3 * self.num_subject + 3 * i,
                   3 * i:3 * (i + 1)] = HL
                b[3 * self.num_subject + 3 * (i - 1): 3 * self.num_subject + 3 * i ,
                   :] = np.dot(HL, np.atleast_2d(self.dSS[i]).T)
                    
            #    print A.todense()
            #    np.dot(R, R.T.conj())
            
            #    A = scipy.sparse.coo_matrix((V,(I,J)),shape=(4,4))
            #    A = A.tocsr()
        x, residual, rank, s = np.linalg.lstsq(A.todense(), b)
        print 'residual', residual[0]
        for i in range(self.num_subject):
            self.dGlobal[i] = x[3*i:3*(i+1),:]
    
            
if __name__ == '__main__':
    alnr = Aligner('4')
    alnr.prepare_allen_whole()
    

    
