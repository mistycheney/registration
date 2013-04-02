from numpy import *
from numpy.linalg import *
from scipy import ndimage
import sys, cv2

def transform_ransac(pt_s, pt_d, model):
    import ransac
    nmatch = len(pt_s) 
    data = hstack((pt_s, ones((nmatch,1)), pt_d, ones((nmatch,1)))) # n*6
    H, ransac_data, err = ransac.ransac(data, model)
    return H, ransac_data, err

def rigid_scaling_fom_points(fp, tp):
    m = mean(fp[:2], axis=1)
    maxstd = max(std(fp[:2], axis=1)) + 1e-9
    C1 = diag([1 / maxstd, 1 / maxstd, 1]) 
    C1[0][2] = -m[0] / maxstd
    C1[1][2] = -m[1] / maxstd
    fp = dot(C1, fp)
    
    m = mean(tp[:2], axis=1)
    maxstd = max(std(tp[:2], axis=1)) + 1e-9
    C2 = diag([1 / maxstd, 1 / maxstd, 1])
    C2[0][2] = -m[0] / maxstd
    C2[1][2] = -m[1] / maxstd
    tp = dot(C2, tp)
    
    fp = float32(fp[:2,:]).T.reshape(1,-1)
    tp = float32(tp[:2,:]).T.reshape(1,-1)
    fp = fp.reshape([1,-1,2])
    tp = tp.reshape([1,-1,2])
    H = cv2.estimateRigidTransform(fp, tp, fullAffine=False)
    
    H = vstack([H, [0, 0, 1]])
    
    H = dot(linalg.inv(C2), dot(H, C1))
    return H

def rigid_fom_points(fp, tp):
    n = fp.shape[1]
    from_pt = float32(fp)[:2,:]
    to_pt = float32(tp)[:2,:]
    
    from_center = mean(from_pt, 1).reshape(2,1)
    to_center = mean(to_pt, 1).reshape(2,1)
    from_pt = from_pt - tile(from_center, [1,n])
    to_pt = to_pt - tile(to_center, [1,n])
    
    H = dot(from_pt, to_pt.T)
    [U,S,V] = svd(H)
    R = dot(V,U.T)
    T = vstack([hstack([R, -dot(R,from_center)+to_center]), [0, 0, 1]])
    return T

def homography_from_points(fp, tp):
    """ Find homography H, such that fp is mapped to tp
        using the linear DLT method. Points are conditioned
        automatically. """
    
    if fp.shape != tp.shape:
        raise RuntimeError('number of points do not match')
        
    # condition points (important for numerical reasons)
    # --from points--
    m = mean(fp[:2], axis=1)
    maxstd = max(std(fp[:2], axis=1)) + 1e-9
    C1 = diag([1 / maxstd, 1 / maxstd, 1]) 
    C1[0][2] = -m[0] / maxstd
    C1[1][2] = -m[1] / maxstd
    fp = dot(C1, fp)
    
    # --to points--
    m = mean(tp[:2], axis=1)
    maxstd = max(std(tp[:2], axis=1)) + 1e-9
    C2 = diag([1 / maxstd, 1 / maxstd, 1])
    C2[0][2] = -m[0] / maxstd
    C2[1][2] = -m[1] / maxstd
    tp = dot(C2, tp)
    
    # create matrix for linear method, 2 rows for each correspondence pair
    nbr_correspondences = fp.shape[1]
    A = zeros((2 * nbr_correspondences, 9))
    for i in range(nbr_correspondences):        
        A[2 * i] = [-fp[0][i], -fp[1][i], -1, 0, 0, 0,
                    tp[0][i] * fp[0][i], tp[0][i] * fp[1][i], tp[0][i]]
        A[2 * i + 1] = [0, 0, 0, -fp[0][i], -fp[1][i], -1,
                    tp[1][i] * fp[0][i], tp[1][i] * fp[1][i], tp[1][i]]
    
    U, S, V = linalg.svd(A)
    H = V[8].reshape((3, 3))    
    
    # decondition
    H = dot(linalg.inv(C2), dot(H, C1))
    
    # normalize and return
    return H / H[2, 2]


def affine_from_points(fp, tp):
    # fp = 3*n
    if fp.shape != tp.shape:
        raise RuntimeError('number of points do not match')
        
    # condition points
    # --from points--
    m = mean(fp[:2], axis=1)
    maxstd = max(std(fp[:2], axis=1)) + 1e-9
    C1 = diag([1 / maxstd, 1 / maxstd, 1]) 
    C1[0][2] = -m[0] / maxstd
    C1[1][2] = -m[1] / maxstd
    fp_cond = dot(C1, fp)
    
    # --to points--
    m = mean(tp[:2], axis=1)
    C2 = C1.copy() #must use same scaling for both point sets
    C2[0][2] = -m[0] / maxstd
    C2[1][2] = -m[1] / maxstd
    tp_cond = dot(C2, tp)
    
    # conditioned points have mean zero, so translation is zero
    A = concatenate((fp_cond[:2], tp_cond[:2]), axis=0)
    U, S, V = linalg.svd(A.T)
    
    # create B and C matrices as Hartley-Zisserman (2:nd ed) p 130.
    tmp = V[:2].T
    B = tmp[:2]
    C = tmp[2:4]
    
    tmp2 = concatenate((dot(C, linalg.pinv(B)), zeros((2, 1))), axis=1) 
    H = vstack((tmp2, [0, 0, 1]))
    
    # decondition
    H = dot(linalg.inv(C2), dot(H, C1))
    
    return H / H[2, 2]

if __name__ == '__main__':
#    pt_s = array([[1,2,3,4,5,6,7],[1,2,3,4,5,6,7],[1,1,1,1,1,1,1]])
#    pt_d = array([[2,3,4,5,6,7,8],[1,2,3,4,5,6,7],[1,1,1,1,1,1,1]])
    
#    from_pt = [[1,1], [1,2], [2,2], [2,1]] # A rectangle at 1,1
#    to_pt =   [[2,2], [4,4], [6,2], [4,0]] # The same, transformed
    
    fp = array([[1,1,2,2,0],[1,2,2,1,0],[1,1,1,1,1]])
    tp = array([[2,4,6,4,0],[2,4,2,0,0],[1,1,1,1,1]])
    
    R = rigid_scaling_fom_points(fp, tp)
    print R
    
    print dot(R,fp)
    
#    A = affine_from_points(pt_s, pt_d)
#    print A
    

