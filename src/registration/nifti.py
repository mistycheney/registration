
from PIL import Image
import nibabel
import os
from pylab import *

def read_image(filename):
    im = nibabel.load(filename)
    imdata = im.get_data()
    return imdata.squeeze()

class Dim:
    sagittal,coronal,transverse = range(3)

def read_waxholm(dim, index):
    image_folder = '/Users/yuncong/Documents/medical images/Waxholm Space'
    image_name = 'canon_hist.nii'
    image_fullname = os.path.join(image_folder, image_name)    
    
    imdata = read_image(image_fullname)
#    print imdata.shape
    
    if dim == Dim.sagittal:
        slice_data = imdata[index,:,:].T
    elif dim== Dim.coronal:
        slice_data = imdata[:,index,::-1].T
    else:
        slice_data = imdata[:,:,index].T
    
    return slice_data


if __name__ == '__main__':
#    image_folder = '/Users/yuncong/Documents/medical images/Waxholm Space'
#    image_name = 'canon_hist.nii'
#    image_fullname = os.path.join(image_folder, image_name)
    slice_data = read_waxholm(Dim.sagittal, 300)
#    im = Image.fromarray(slice_data)
#    im.save(image_fullname+'_z300.tif')
    figure(); gray(); axis('off')
    imshow(slice_data)
    
    show()
