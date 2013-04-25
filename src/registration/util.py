'''
Utility functions for plotting, etc.
'''

from subprocess import call
import cPickle
import os, sys
import numpy as np
from scipy import ndimage
#from scipy import signal, ndimage, spatial
from registration import config
import cv2
import matplotlib.pyplot as plt


def plot_surface(Z, X, Y, x_label, y_label, z_label):
    '''
    Plot a surface
    '''
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
    '''
    Plot a surface
    '''
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
        
def execute(cmmd):
    '''
    execute a system command
    '''
#    print cmmd
    try:
        retcode = call(cmmd, shell=True)
        if retcode < 0:
            print >> sys.stderr, "Child was terminated by signal", -retcode
#        else:
#            print >> sys.stderr, "Child returned", retcode
    except OSError, e:
        print >> sys.stderr, "Execution failed:", e   


def histogram(s, windowId):
    '''
    plot a histogram for array s
    '''
    import matplotlib.pyplot as plt 
    hist, bins = np.histogram(s)
    width = 0.7*(bins[1]-bins[0])
    center = (bins[:-1]+bins[1:])/2
    plt.figure(windowId)
    plt.bar(center, hist, align = 'center', width = width)
    plt.show()
    
    
def histogram2(d0, d1):
    '''
    plot two histograms
    '''
    import matplotlib.pyplot as plt
    
    hist, bins = np.histogram(d0)
    width = 0.7*(bins[1]-bins[0])
    center = (bins[:-1]+bins[1:])/2
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(121)
    ax1.bar(center, hist, align = 'center', width = width)
    
    hist, bins = np.histogram(d1)
    width = 0.7*(bins[1]-bins[0])
    center = (bins[:-1]+bins[1:])/2
    ax2 = fig1.add_subplot(122)
    ax2.bar(center, hist, align = 'center', width = width)
    
    plt.show()


if __name__ == '__main__':
    pass