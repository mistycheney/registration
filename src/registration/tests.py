'''
Created on Mar 23, 2013

@author: yuncong
'''

from registration import aligner
from registration import plot
from registration import scoring
from registration import hessian
import cv2

alnr = aligner.Aligner('4')
alnr.prepare_allen()
alnr.prepare_subject()
alnr.initial_shift()

def test_scores(scores_name):
    i = int(scores_name.split('_')[-1])    
    sp = hessian.ScorePlotter(scores_name)
    opt = sp.opt
    print opt
    im = alnr.subject_stack[i]
    cnt = alnr.subject_cnt_stack[i]
    im_t, cnt_t = scoring.transform_cnt(im, cnt, opt)
    im_orig, cnt_orig = scoring.transform_cnt(im, cnt, (0,0,0))
    
    if 'allen' in scores_name:
        im_ref = alnr.allen_stack[alnr.allen_match_id_stack_best[i]]
        cnt_ref = alnr.allen_cnt_stack[alnr.allen_match_id_stack_best[i]]
    else:
        im_ref = alnr.subject_stack[i-1]
        cnt_ref = alnr.subject_cnt_stack[i-1]
    
    s1,D1 = scoring.compute_score_cnt(im_orig, cnt_orig, im_ref, cnt_ref, 
                              'centroid', 'centroid', show=True, outputD=True)
    s2,D2 = scoring.compute_score_cnt(im_t, cnt_t, im_ref, cnt_ref,
                              'centroid', 'centroid', show=True, outputD=True)
    cv2.destroyAllWindows()
    
    from registration import util
    util.histogram(D1.min(0))
    util.histogram(D2.min(0))
    util.histogram(D1.min(1))
    util.histogram(D2.min(1))

if __name__ == '__main__':
#    test_scores('scores_allen_5')
    test_scores('scores_neighbor_2')