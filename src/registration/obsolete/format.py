import numpy
from PIL import Image
import itk

def PIL2array(img):
    if len(img.size) == 2: #grayscale
        return numpy.array(img.getdata(), numpy.uint8).reshape(img.size[1], img.size[0])
    elif len(img.size) == 3 and img.size[2] == 3: #RGB
        return numpy.array(img.getdata(), numpy.uint8).reshape(img.size[1], img.size[0], 3)

def array2PIL(arr, size):
    mode = 'L'
    if len(arr.shape) == 3:
        arr = arr.reshape(arr.shape[0]*arr.shape[1], arr.shape[2])
    else:
        arr = arr.reshape(arr.shape[0]*arr.shape[1], 1)
    return Image.frombuffer(mode, size, arr.tostring(),  "raw", mode, 0, 1)

# image_type = itk.F etc.
def itk2numpy(itk_image, image_type):
    itk_py_converter = itk.PyBuffer[image_type]
    image_array = itk_py_converter.GetArrayFromImage( itk_image )
    return image_array

def numpy2itk(image_array, image_type):
    itk_py_converter = itk.PyBuffer[image_type]
    itk_image = itk_py_converter.GetImageFromArray( image_array.tolist() )
    return itk_image


if __name__ == '__main__':
    import os, cv2
    image_folder = '/Users/yuncong/Documents/medical images/Test/'
    k = 0
    image_name = 'H4nissl_2_'+str(k)+'.tif'
    image_fullname = os.path.join(image_folder, image_name)
    im3 = cv2.imread(image_fullname)
    im = cv2.cvtColor(im3, cv2.COLOR_BGR2GRAY)
    
    im_pil = array2PIL(im, im.shape[::-1])
    im_pil.save('tmp.pgm')
    