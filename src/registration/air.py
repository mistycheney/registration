
from subprocess import call
import sys, os
from PIL import Image
from numpy import *
from pylab import *
import nibabel
import nifti

air_bindir = '/Users/yuncong/Documents/AIR5.3.0/bin' 

def air_reunite(vol_file, in_files):
    cmd = ' '.join([os.path.join(air_bindir, "reunite"), "'" + vol_file + "'", 'y', ' '.join(in_files)])
    print cmd
    call(cmd, shell=True)

def air_convertfmt(in_file, out_file):
    if len(in_file) == 1:    
        im = Image.open(in_file).convert('L')
    
        if in_file[-3:] != 'img':
            raw = im.tostring()
            file = open(out_file + '.img', 'w')
            file.write(raw)
            file.close();
        else:
            cmd = "cp -f '" + in_file + "' " + out_file + '.img'
            print cmd
            call(cmd, shell=True)
        return im.size
    else:
        im = Image.open(in_file[0]).convert('L')
        air_reunite(out_file+'img', in_file)
        return im.size
    

def air_makeaheader(filename, filetype, x_dim, y_dim, z_dim, x_size, y_size, z_size):
    """http://bishopw.loni.ucla.edu/AIR5/makeaheader.html"""    
    try:
        cmmd = ' '.join([os.path.join(air_bindir, 'makeaheader'), filename, str(filetype), str(x_dim), str(y_dim), str(z_dim), str(x_size), str(y_size), str(z_size)])
        print cmmd
        retcode = call(cmmd, shell=True)
        if retcode < 0:
            print >> sys.stderr, "Child was terminated by signal", -retcode
        else:
            print >> sys.stderr, "Child returned", retcode
    except OSError, e:
        print >> sys.stderr, "Execution failed:", e
    
    
def air_alignlinear(standard_file, reslice_file, air_out, model_number=26, cost_func=3, conv_thresh=0.1, std_thresh=0, reslice_thresh=0):
    """http://bishopw.loni.ucla.edu/AIR5/alignlinear.html"""
    try:
        cmmd = ' '.join([os.path.join(air_bindir, 'alignlinear'), "'" + standard_file + "'", "'" + reslice_file + "'", "'" + air_out + "'",
                          "-m " + str(model_number), "-c " + str(conv_thresh), "-x " + str(cost_func), "-t1 " + str(std_thresh), "-t2 " + str(reslice_thresh)])
        print cmmd
        retcode = call(cmmd, shell=True)
        if retcode < 0:
            print >> sys.stderr, "Child was terminated by signal", -retcode
        else:
            print >> sys.stderr, "Child returned", retcode
    except OSError, e:
        print >> sys.stderr, "Execution failed:", e


def air_reslice(air_file, output_file, interp_model=1):
    """"http://bishopw.loni.ucla.edu/AIR5/reslice.html"""
    try:
        cmmd = ' '.join([os.path.join(air_bindir, 'reslice'), "'" + air_file + "'", "'" + output_file + "'", "-o"])
        print cmmd
        retcode = call(cmmd, shell=True)
        if retcode < 0:
            print >> sys.stderr, "Child was terminated by signal", -retcode
        else:
            print >> sys.stderr, "Child returned", retcode
    except OSError, e:
        print >> sys.stderr, "Execution failed:", e

def air_register(std_img, reslice_img, reconstruct_img):
    filetype = 0
#    x_dim = 1169 
#    y_dim = 1120
    z_dim = 1
    x_size = 7.32e-3
    y_size = 7.32e-3
    z_size = 90e-3
#    x_dim = 594 
#    y_dim = 420
#    z_dim = 1
    
    std_tmp = 'test1'
    reslice_tmp = 'test2'
    rec_tmp = 'rtest2'
    air_file = 'tmp.air'
    
    [x_dim, y_dim] = air_convertfmt(std_img, std_tmp)
    air_convertfmt(reslice_img, reslice_tmp)
    air_makeaheader(std_tmp + '.img', filetype, x_dim, y_dim, z_dim, x_size, y_size, z_size)
    air_makeaheader(reslice_tmp + '.img', filetype, x_dim, y_dim, z_dim, x_size, y_size, z_size)
    air_alignlinear(std_tmp + '.img', reslice_tmp + '.img', air_file)
    air_reslice(air_file, rec_tmp + '.img')
    nifti.read_image(rec_tmp + '.img')

if __name__ == '__main__':
    image_folder = '/Users/yuncong/Documents/medical images/Test'
#    image_folder = '/Users/yuncong/Documents/medical images/HJK Samples'
    image_series1 = 'H4nissl_1 - 2010-03-29 15.09.28_'
    image_series2 = 'H4nissl_1 - 2010-03-29 15.09.28_'
#    image_series = '164413_Z0.png_';
    image_fmt = '.jpg'

    nslice = 8
    in_files1 = [os.path.join(image_folder, image_series1 + str(i) + image_fmt) for i in range(1,nslice+1)]
    in_files2 = [os.path.join(image_folder, image_series2 + str(i) + image_fmt) for i in range(1,nslice+1)]

    air_register(in_files1, in_files2)

    gray()
    show()

