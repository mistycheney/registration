'''
Created on Mar 21, 2013

@author: yuncong
'''

from registration import aligner
from registration import viewer
from registration import contour
from registration import scoring
import cv2

if __name__ == '__main__':    
    alnr = aligner.Aligner('4',40)
    alnr.prepare_allen()
    alnr.prepare_subject()
    
    alnr.initial_shift()

    alnr.optimize_atlas()
    alnr.optimize_neighbor()

#    viewer = plot.StackViewer('allen', alnr.allen_stack, i=0)
#    viewer.show()
    
#    viewer = viewer.StackAtlasMatchViewer('allen match', alnr, i=1, overlay=True)
#    viewer.show()

#    viewer = viewer.StackNeighborViewer('neighbor match', alnr, i=2)
#    viewer.show()
    
#    viewer = plot.StackCentroidViewer('subject 4', alnr.subject_stack, i=1)
#    viewer.show()
