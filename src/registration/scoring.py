'''
Created on Mar 22, 2013

@author: yuncong
'''

import cPickle as pickle
from registration import config
import numpy as np
from registration import util
import scipy.sparse
import scipy.sparse.linalg
import sys

class ScoreReader:
    def __init__(self, scores):    
        self.tx_range = config.TX_RANGE
        self.ty_range = config.TY_RANGE
        self.theta_range = config.THETA_RANGE
        self.grid_size = (self.tx_range[1] - self.tx_range[0],
                     self.ty_range[1] - self.ty_range[0],
                     self.theta_range[1] - self.theta_range[0])
        
        if isinstance(scores, str):
            self.scores = pickle.load(open(config.SCORES_FOLDER + scores + '.p', 'rb'))
        else:
            self.scores = scores
        self.tx_opt_ind, self.ty_opt_ind, self.theta_opt_ind = np.unravel_index(np.argmin(self.scores), self.scores.shape)
        self.opt = np.array([self.tx_range[self.tx_opt_ind],
                             self.ty_range[self.ty_opt_ind],
                             self.theta_range[self.theta_opt_ind]])
    
    def plot(self, dims):
        if dims == 'xy':
            util.plot_surface(self.scores[:, :, self.theta_opt_ind], self.tx_range, self.ty_range,
                              x_label='tx', y_label='ty', z_label='score')
        elif dims == 'xt':
            util.plot_surface(self.scores[:, self.ty_opt_ind, :], self.tx_range, self.theta_range,
                              x_label='tx', y_label='theta', z_label='score')
        elif dims == 'yt':
            util.plot_surface(self.scores[self.tx_opt_ind, :, :], self.ty_range, self.theta_range,
                              x_label='ty', y_label='theta', z_label='score')
            
    def compute_hessian(self, h=1):
        x = self.tx_opt_ind
        y = self.ty_opt_ind
        t = self.theta_opt_ind
        
        while True:
            fxy = ((self.scores[x + h, y + h, t] - self.scores[x + h, y - h, t]) - 
                   (self.scores[x - h, y + h, t] - self.scores[x - h, y - h, t])
                   ) / (4 * self.grid_size[0] * self.grid_size[1] * h * h)
            fxt = ((self.scores[x + h, y, t + h] - self.scores[x + h, y, t - h]) - 
                   (self.scores[x - h, y, t + h] - self.scores[x - h, y, t - h])
                   ) / (4 * self.grid_size[0] * self.grid_size[2] * h * h)
            fyt = ((self.scores[x, y + h, t + h] - self.scores[x, y + h, t - h]) - 
                   (self.scores[x, y - h, t + h] - self.scores[x, y - h, t - h])
                   ) / (4 * self.grid_size[1] * self.grid_size[2] * h * h)
            fxx = ((self.scores[x + 2 * h, y, t] - self.scores[x, y, t]) - 
                   (self.scores[x, y, t] - self.scores[x - 2 * h, y, t])
                   ) / (4 * (self.grid_size[0] * h) ** 2)
            fyy = ((self.scores[x, y + 2 * h, t] - self.scores[x, y, t]) - 
                   (self.scores[x, y, t] - self.scores[x, y - 2 * h, t])
                   ) / (4 * (self.grid_size[1] * h) ** 2)
            ftt = ((self.scores[x, y, t + 2 * h] - self.scores[x, y, t]) - 
                   (self.scores[x, y, t] - self.scores[x, y, t - 2 * h])
                   ) / (4 * (self.grid_size[2] * h) ** 2)
            H = np.array([[fxx, fxy, fxt],
                          [fxy, fyy, fyt],
                          [fxt, fyt, ftt]], dtype=np.float)
            
            try:
                HL = np.linalg.cholesky(H)
                break
            except np.linalg.LinAlgError as e:
                print e
                HL = None
                h = h + 1
                continue
                
        return H, HL
        
    #    x = scipy.sparse.linalg.spsolve(A, b)

if __name__ == '__main__':
    import sys
    sp = ScoreReader('scores_allen_%d'%24)
#    sp.plot('xt')
    print sp.compute_hessian()
#    sys.exit(0)


    
    
