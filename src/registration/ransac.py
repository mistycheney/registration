from pylab import *
from numpy import random
import transform
from registration.config import *
from scipy import ndimage
from scipy.misc import comb

def get_variance(data):
    return max((sqrt(var(data[:, :3], 0)) + sqrt(var(data[:, 3:6], 0))) * 0.5)

def ransac(data, model):
    if len(data) < MIN_MODEL_SAMPLES:
        raise ValueError("sample number too small")
    
#    MAX_ITER_ADAPT = min(MAX_ITER, int32(comb(len(data), MIN_MODEL_SAMPLES))*2)
    MAX_ITER_ADAPT = MAX_ITER
    print 'MAX_ITER_ADAPT', MAX_ITER_ADAPT
    
    VAR_THRESH_ADAPT = get_variance(data) * TOTAL_VAR_RATIO
    print 'VAR_THRESH_ADAPT', VAR_THRESH_ADAPT
    
    MIN_CONSENSUS_SIZE_ADAPT = min(max(MIN_MODEL_SAMPLES + 1, int(len(data) * TOTAL_CONSENSUS_RATIO)), MIN_CONSENSUS_SIZE)
#                            if len(data) <= MIN_CONSENSUS_SIZE \
#                            else MIN_CONSENSUS_SIZE
    print 'MIN_CONSENSUS_SIZE_ADAPT', MIN_CONSENSUS_SIZE_ADAPT
        
    iterations = 0
    best_model = None
    best_err = inf
    best_size = 0
    best_var = 0
    best_inlier_idxs = None
    while iterations < MAX_ITER_ADAPT:
#        print 'iteration', iterations
        est_idxs, test_idxs = random_partition(MIN_MODEL_SAMPLES, data.shape[0])
#        print 'est_idxs', est_idxs
#        print 'test_idxs', test_idxs
        est_inliers = data[est_idxs]
#        print 'est_idxs', est_idxs
        test_points = data[test_idxs]
#        print 'test_points', test_points
        est_model = model.fit(est_inliers)
#        print 'est_model', est_model
        test_err = model.get_error(test_points, est_model)
#        print 'test_err', test_err
        also_idxs = test_idxs[test_err < INLIER_DIST_THRESH] # select indices of rows with accepted points
        also_inliers = data[also_idxs]
#        print len(also_idxs), 'also_idxs'
    
#        print 'also_idxs', also_idxs
        consensus_data = concatenate((est_inliers, also_inliers))
        
#        print 'size', len(consensus_data)
#        print 'var', get_variance(consensus_data)
        
        this_size = len(consensus_data)
        this_var = get_variance(consensus_data)

        if this_var > VAR_THRESH_ADAPT and this_size >= MIN_CONSENSUS_SIZE_ADAPT:
            this_model = model.fit(consensus_data)
            this_errs = model.get_error(consensus_data, this_model)
            this_err = mean(this_errs)
            
#            print 'this_size', this_size
#            print 'this_var', this_var
#            print 'this_err', this_err
            
            if this_err < ERR_THRESH and this_size > best_size:
#            if this_err < ERR_THRESH and this_var > best_var:
                best_model = this_model
                best_err = this_err
                best_size = this_size
                best_var = this_var
                print 'best_err', best_err
                print 'best_size', best_size
                print 'best_var', best_var
                best_inlier_idxs = concatenate([est_idxs, also_idxs])
        iterations += 1
    if best_model is None:
        raise ValueError("did not meet fit acceptance criteria")
    else:
        return best_model, best_inlier_idxs, best_err
    

def random_partition(n, n_data):
    """return n random rows of data (and also the other len(data)-n rows)"""
    all_idxs = arange(n_data)
    random.shuffle(all_idxs)
    idxs1 = all_idxs[:n].T
    idxs2 = all_idxs[n:].T
    return idxs1, idxs2


class Model(object):
    def __init__(self, transform):
        self.transform = transform
        
    def fit(self, data):
        data = data.T
        fp = data[:3]
        tp = data[3:] # fp = 3*n
        if self.transform == TRANSFORM.RIGID:
            A = transform.rigid_fom_points(fp, tp)
        elif  self.transform == TRANSFORM.RIGID_SCALING:            
            A = transform.rigid_scaling_fom_points(fp, tp)
        elif self.transform == TRANSFORM.AFFINE:
            A = transform.affine_from_points(fp, tp)
        return A[:2, :]
    
    def get_error(self, data, H):
        data = data.T
        fp = data[:3]
        tp = data[3:]
        fp_transformed = dot(H, fp)
        d = float64(tp[:2] - fp_transformed)
        error = [linalg.norm(i) for i in d.T]
        return float32(error).T
