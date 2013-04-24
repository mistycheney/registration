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
    
alnr.num_subject = 20
alnr.global_optimzation()

#sorted_allen = [alnr.allen_stack[i] for i in sorted(alnr.allen_stack.iterkeys())]
#v = viewer.StackViewer('allen', sorted_allen, i=0)
#v.show()
#    
#v = viewer.StackAtlasMatchViewer('allen match', alnr, i=0, overlay=True)
#v.show()
#
#v = viewer.StackNeighborViewer('neighbor match', alnr, i=1)
#v.show()
#    
v = viewer.StackGlobalViewer('global match', alnr, i=0)
v.show()
#    
v = viewer.StackCentroidViewer('subject 4', alnr.subject_stack, i=1)
v.show()
