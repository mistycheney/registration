import itk
import os
from pylab import *

if __name__ == '__main__':
    ImageDimension = 2
    PixelType = itk.F
    
    ImageType = itk.Image[PixelType, ImageDimension]
    
    SpaceDimension = ImageDimension
    SplineOrder = 3
    CoordinateRepType = itk.D
    
    TransformType = itk.BSplineTransform[CoordinateRepType, SpaceDimension, SplineOrder]
    OptimizerType = itk.LBFGSOptimizer
    MetricType = itk.MeanSquaresImageToImageMetric[ImageType, ImageType]
    InterpolatorType = itk.LinearInterpolateImageFunction[ImageType, itk.D]
    RegistrationType = itk.ImageRegistrationMethod[ImageType, ImageType]
    
    metric = MetricType.New()
    optimizer = OptimizerType.New()
    interpolator = InterpolatorType.New()
    registration = RegistrationType.New()
    
    registration.SetMetric(metric)
    registration.SetOptimizer(optimizer)
    registration.SetInterpolator(interpolator)
    
    transform = TransformType.New()
    registration.SetTransform(transform)
    
    fixedImage = ImageType.New()
    movingImage = ImageType.New()
    
    registration.SetFixedImage(fixedImage)
    registration.SetMovingImage(movingImage)
    
    fixedRegion = fixedImage.GetBufferedRegion()
    registration.SetFixedImageRegion(fixedRegion)
    
    fixedPhysicalDimensions = TransformType.PhysicalDimensionsType
    
    
    
    image_folder = '/Users/yuncong/Documents/medical images/Test'
    fixed_image_filename = os.path.join(image_folder,'H4nissl_2 - 2010-03-29 15.16.30_x0.3125_z0_0.tif')
    moving_image_filename = os.path.join(image_folder,'H4nissl_2 - 2010-03-29 15.16.30_x0.3125_z0_1.tif')

    fixedImageReader = itk.ImageFileReader[FixedImageType].New()
    fixedImageReader.SetFileName(fixed_image_filename)
    fixedImageReader.Update()
#    image = rd1.GetOutput()

    movingImageReader = itk.ImageFileReader[MovingImageType].New()
    movingImageReader.SetFileName(moving_image_filename)
    movingImageReader.Update()
    
    
    
    
    
    
    
    
    
    