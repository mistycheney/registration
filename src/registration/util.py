from subprocess import call
import cPickle
import os, sys
import numpy as np
from scipy import ndimage
#from scipy import signal, ndimage, spatial
from registration import config
import cv2
import matplotlib.pyplot as plt

def joint_histogram(im_from, im_to):
    im_from = ndimage.interpolation.zoom(im_from, np.float32(im_to.shape)/im_from.shape)
    h, w = im_from.shape[:2]
    histo = np.zeros([256,256])
    for x in range(w):
        for y in range(h):
            histo[im_from[y,x], im_to[y,x]] += 1
    return histo

def flatten(to_merge):
    to_remove = [item for sublist in to_merge for item in sublist]
    return to_remove


def plot_surface(Z, X, Y, x_label, y_label, z_label):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    X, Y = np.meshgrid(X, Y, indexing='ij')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
            linewidth=0, antialiased=False)
#    ax.set_zlim(-1.01, 1.01)
    
#    ax.zaxis.set_major_locator(LinearLocator(10))
#    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.show()


def plot_surface_func(func, X, Y):
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(X, Y)
    nrow, ncol = X.shape
    
    Z = np.zeros(X.shape)
    for i in range(nrow):
        for j in range(ncol):
            Z[i,j] = func(X[i,j],Y[i,j])
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet,
            linewidth=0, antialiased=False)
    ax.set_zlim(0, 4)
    
#    ax.zaxis.set_major_locator(LinearLocator(10))
#    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.show()


def plot_keypoints_3d(kp_set_list):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.gca(projection='3d')
#    ax = fig.add_subplot(111, projection='3d')
    
    colors = 'rgbkcym'
    for sec, kp_set in enumerate(kp_set_list):
        ax.scatter(kp_set[:, 0], kp_set[:, 1], kp_set[:, 2],
                   zdir='z', c=colors[sec % len(colors)])
    ax.set_xlabel('Posterior')
    ax.set_ylabel('Inferior')
    ax.set_zlabel('Right')
    ax.set_xlim3d([0, 20000])
    ax.set_ylim3d([0, 20000])
    ax.set_zlim3d([0, 20000])
    plt.show()

class StackViewer:
    def __init__(self, name, im_stack, i=0):
        self.im_stack = im_stack
        self.name = name
        cv2.namedWindow(self.name, cv2.cv.CV_WINDOW_NORMAL)
        self.update(i)
        
        cv2.createTrackbar("section", self.name, i, len(im_stack)-1, self.update)
        cv2.waitKey()
        cv2.destroyAllWindows()
        
    def update(self, i):
        cv2.imshow(self.name, self.im_stack[i])
        
        
def execute(cmmd):
#    print cmmd
    try:
        retcode = call(cmmd, shell=True)
        if retcode < 0:
            print >> sys.stderr, "Child was terminated by signal", -retcode
#        else:
#            print >> sys.stderr, "Child returned", retcode
    except OSError, e:
        print >> sys.stderr, "Execution failed:", e   

def pickle_save(obj, filename):
    os.chdir(config.PICKLE_FOLDER)
    dataset_file = open(filename, 'wb')
    cPickle.dump(obj, dataset_file, -1)
    dataset_file.close()
    
def pickle_load(filename):
    os.chdir(config.PICKLE_FOLDER)
    dataset_file = open(filename, 'rb')
    obj = cPickle.load(dataset_file)
    dataset_file.close()
    return obj

def conditional_load(filename, func, args, regenerate=False, append_obj=None):
    os.chdir(config.PICKLE_FOLDER)
    if os.path.exists(filename) and not regenerate:
        return pickle_load(filename)
    else:
        obj = func(*args)
        if append_obj is None:
            pickle_save(obj, filename)
            return obj
        else:
            pickle_save((obj, append_obj), filename)
            return obj, append_obj

def histogram(s):
    import matplotlib.pyplot as plt 
    hist, bins = np.histogram(s)
    width = 0.7*(bins[1]-bins[0])
    center = (bins[:-1]+bins[1:])/2
    plt.bar(center, hist, align = 'center', width = width)
    plt.show()


#def euclidean_distance_matrix(desc_s, desc_d, q):
#    import time
#    begin = time.time()
#    
##    if q:
##    else:
##        A = reshape([linalg.norm(r) ** 2 for r in desc_s], (len(desc_s), 1))
##        A = tile(A, [1, len(desc_d)])
##        B = [linalg.norm(r) ** 2 for r in desc_d]
##        B = tile(B, [len(desc_s), 1])
##        D = A + B - 2 * dot(desc_s, desc_d.T)
##        D = sqrt(D)
#    
#    print 'time', time.time() - begin
#
#    
#    return D


#Plot frequency and phase response
#def mfreqz(b,a=1):
#    w,h = signal.freqz(b,a)
#    h_dB = 20 * log10 (abs(h))
#    subplot(211)
#    plot(w/max(w),h_dB)
##    ylim(-150, 5)
#    ylabel('Magnitude (db)')
#    xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
#    title(r'Frequency response')
#    subplot(212)
#    h_Phase = unwrap(arctan2(imag(h),real(h)))
#    plot(w/max(w),h_Phase)
#    ylabel('Phase (radians)')
#    xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
#    title(r'Phase response')
#    subplots_adjust(hspace=0.5)
#
##Plot step and impulse response
#def impz(b,a=1):
#    l = len(b)
#    impulse = repeat(0.,l); impulse[0] =1.
#    x = arange(0,l)
#    response = signal.lfilter(b,a,impulse)
#    subplot(211)
#    stem(x, response)
#    ylabel('Amplitude')
#    xlabel(r'n (samples)')
#    title(r'Impulse response')
#    subplot(212)
#    step = cumsum(response)
#    stem(x, step)
#    ylabel('Amplitude')
#    xlabel(r'n (samples)')
#    title(r'Step response')
#    subplots_adjust(hspace=0.5)
#
#def plot_fft(im):
#    F1 = fftpack.fft2(im)
#    F2 = fft.fftshift(F1)
#    psd2D = abs( F2 )**2
#    figure()
#    imshow(log10(psd2D))


if __name__ == '__main__':
    pass
#    A = numpy.random.random((10,10))
#    B = numpy.random.random((10,10))
#    D1 = euclidean_distance_matrix(A, B, 1)
#    D2 = euclidean_distance_matrix(A, B, 0)
#    print all(D1-D2<0.000001)