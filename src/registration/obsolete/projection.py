import sys, os
from pylab import *
from registration import util, allen, sift
from registration.config import *
import scipy, numpy

if __name__ == '__main__':
        
    dataset_id = 100048576
    allen_secs = allen.get_secnums(dataset_id)
    
    allen_sec_range = {4:(497, 346)}
    nsections = {3:0, 4:40, 5:32, 6:34, 9:28, 10:80}

    stack_id = 4
    nsection = nsections[stack_id]
    
#    for sec_num in range(nsection):
    sec_num = 10
    
    allen_sec_begin, allen_sec_end = allen_sec_range[stack_id]
    delta = allen_sec_end - allen_sec_begin
    initial_allen_sec = int(allen_sec_begin + float(sec_num) / nsection * delta)
    initial_allen_sec_index = argmin(abs(array(allen_secs) - initial_allen_sec))
    initial_allen_sec = allen_secs[initial_allen_sec_index]
    print 'initial_allen_sec:', initial_allen_sec
    
    test_sections = allen_secs[max(0, initial_allen_sec_index - 10):min(initial_allen_sec_index + 10, len(allen_secs))]
    print 'test_sections:', test_sections

    image_from_name = '{0}_{1}.tif'.format(stack_id, sec_num)
    image_from_fullname = SECTION_FOLDER + str(stack_id) + '/' + image_from_name
    
    matchings, test_sections = \
    util.conditional_load(image_from_name[:-4] + '_matchings', \
                          allen.get_allen_matching_multisections, \
                          (image_from_fullname, dataset_id, test_sections), \
                          regenerate=False, append_obj=test_sections)
    
#    quit()
    
    err_list = [m[-1] if m is not None else inf for m in matchings]
    print 'err_list:', err_list
    best_section = test_sections[argmin(err_list)]
    print 'best_section:', best_section
    chosen_secs_idx = [i for i, m in enumerate(matchings) if m is not None and m[-1] < RELEVANCY_THRESH]
    chosen_secs = [test_sections[i] for i in chosen_secs_idx]
    print 'chosen_secs:', chosen_secs
    
    query_keypoints = [matchings[i][1] for i in chosen_secs_idx]
    query_descriptors = [matchings[i][2] for i in chosen_secs_idx]
    allen_keypoints = [matchings[i][3] for i in chosen_secs_idx]
    allen_descriptors = [matchings[i][4] for i in chosen_secs_idx]
    
#    D = scipy.spatial.distance.cdist(query_descriptors, allen_descriptors)
#    q = array(unravel_index(find(D < 200), D.shape))
#    print q
        
    downsample = 4
    allen_keypoints_ref = [allen.points_to_reference(dataset_id, sec_num, kp_list[:, :2], downsample)\
                           for sec_num, kp_list in zip(chosen_secs, allen_keypoints)]
#    print allen_keypoints_ref
#    util.plot_keypoints_3d(allen_keypoints_ref)

    query_kp = [array([kp[:2] for kp in kp_set]) for kp_set in query_keypoints]
    query_keypoints_ref = []
    kps = []
    for sec, kp_set in zip(chosen_secs, query_kp):
        m = matchings[test_sections.index(sec)][0]
        kps_sec = dot(m, vstack([kp_set.T, ones([1, len(kp_set)])]))
        kps.append(kps_sec)
#        print kps.T
        kpset = allen.points_to_reference(dataset_id, best_section, kps_sec.T, downsample)
        query_keypoints_ref.append(kpset)
    
#    print 'query_keypoints_ref'
#    print query_keypoints_ref    
#    util.plot_keypoints_3d(query_keypoints_ref)
            
    def rigid3D(r, b, a, xt, yt, zt):
        # (roll,pitch,yaw,posterior,inferior,right), transform in this order
        T = [[cos(a)*cos(b), cos(a)*sin(b)*sin(r)-sin(a)*cos(r), cos(a)*sin(b)*cos(r)+sin(a)*sin(r), xt],
             [sin(a)*cos(b), sin(a)*sin(b)*sin(r)+cos(a)*cos(r), sin(a)*sin(b)*cos(r)-cos(a)*sin(r), yt],
             [-sin(b), cos(b)*sin(r), cos(b)*cos(r), zt],
             [0,0,0,1]]
        return T
    
    def plane_plane_intersection(n1,d1,n2,d2):
        print n1, n2, d1, d2
        a = cross(n1,n2)
        if all(a == 0):
            print 'planes are parallel'
            return None
        m = vstack([n1[:2],n2[:2]])
        b = vstack([d1,d2])
        x0 = solve(m,b)        
        return x0, a
    
    def func_yaw_pitch(yaw, pitch):
        xx = [yaw,pitch,0,0,0,0]
        return func(xx)
        
    def func_p_i(p, i):
        xx = [0,0,0,p,i,0]
        return func(xx)
        
    def func(x):
        print x
        ak = array([p for s in allen_keypoints_ref for p in s])
        ad = [p for s in allen_descriptors for p in s]
        qk = array([p for s in query_keypoints_ref for p in s])
        qd = [p for s in query_descriptors for p in s]
        
        T = rigid3D(*x)
        qk = dot(T, vstack([qk.T, ones([1,len(qk)])])).T
        qk = qk[:,:3]
        
        D = scipy.spatial.distance.cdist(qk, ak)
#        print D
        nq = len(qk)
        na = len(ak)
        nearest_neighbor = argsort(D, axis=1)[:, 0]
#        print argsort(D, axis=1)
#        static_indices = numpy.indices((nq, na))
#        print D[static_indices[0], argsort(D, axis=1)]
#        print nearest_neighbor
        dist = [scipy.spatial.distance.euclidean(qd[i], ad[nearest_neighbor[i]]) \
                for i in range(nq)]
#        raw_input()
        return mean(dist)
    
#    n = 100
#    a = numpy.random.uniform(-pi/4,pi/4,n)
#    b = numpy.random.uniform(-pi/4, pi/4, n)
#    r = numpy.random.uniform(-pi/4, pi/4, n)
#    xt = numpy.random.uniform(-1000, 1000, n)
#    yt = numpy.random.uniform(-1000, 1000, n)
#    zt = numpy.random.uniform(-1000, 1000, n)
 
    chosen_sec_idx = 4
    d_ref = chosen_secs[chosen_sec_idx]*25
    n_ref = array([1,0,0])
    
    x = array([0,pi/4,0,0,0,0])
    T = rigid3D(*x)
    
    n_query = dot(T, hstack([n_ref, 1]).T)[:3]
    d_query = best_section*25*norm(n_query)
    
    x0, a = plane_plane_intersection(n_ref,d_ref,n_query,d_query)
    print x0, a
    quit()
    
#    params = zip(*[a,b,r,xt,yt,zt])
#    fopt = [None]*n
#    for i in range(n):
#        print params[i]
#        x0 = array(params[i])
#        res = scipy.optimize.fmin_powell(func, x0, maxiter=1000, retall=True, disp=True, full_output=True)
#        res = scipy.optimize.fmin(func, x0, maxiter=1000, retall=True, disp=True, full_output=True)
#        fopt[i] = res[1]

#    import time
#    begin = time.time()
    
#    x0 = array([0,0,0,0,0,0])
#    func(x0)
#    res = scipy.optimize.anneal(func, x0, args=(), schedule='fast',\
#        full_output=1, T0=1, Tf=1e-12, maxeval=None, maxaccept=None, maxiter=400,\
#        boltzmann=1.0, learn_rate=0.5, feps=1e-06, quench=1, m=10, n=1.0, \
#        lower=array([-pi/4,-pi/4,-pi/4,-1000,-1000,-1000]), \
#        upper=array([pi/4,pi/4,pi/4,1000,1000,1000]), dwell=50, disp=True)
#    print res
    
#    util.plot_surface(func_yaw_pitch, arange(-pi/4,pi/4,0.1), arange(-pi/4,pi/4,0.1))
#    util.plot_surface(func_p_i, arange(-500,500,10), arange(-500,500,10))
    
#    print 'time:', time.time() - begin
    
#    im_from = cv2.imread(image_from_fullname, 0)
#    cv2.namedWindow('Ours', cv2.cv.CV_WINDOW_NORMAL)
#    cv2.imshow('Ours', im_from)
     
#    allen_im = [None] * len(allen_secs)
#    for i, sec in enumerate(allen_secs):
#        allen_im[i] = cv2.imread(allen.get_filename(dataset_id, sec), 0)
#    util.Plotter('Allen', allen_im, allen_secs.index(best_section))
        
#    show()
    
