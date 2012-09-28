'''
Created on Jul 21, 2012

@author: yuncong
'''

import itk
import os
from pylab import *


def readimage( imagetype, filename ):
    rd1 = itk.ImageFileReader[imagetype].New()
    rd1.SetFileName(filename)
    rd1.Update()
    image = rd1.GetOutput()
    return image

def smooth(filename):
    reader1 = itk.ImageFileReader.IF2.New(filename)
    image1 = reader1.GetOutput()
    smoothingFilter = itk.SmoothingRecursiveGaussianImageFilter.IF2IF2.New()
    smoothingFilter.SetInput( image1 )
    smoothingFilter.SetSigma( 2.0 )
    smoothingFilter.Update()
    smoothedImage = smoothingFilter.GetOutput()
#    itk.write( smoothedImage, pathToImages+"/smoothed.mha")

if __name__ == '__main__':
    Dimension = 2
    PixelType = itk.F
    FixedImageType = itk.Image[PixelType, Dimension]
    MovingImageType = itk.Image[PixelType, Dimension]
    
#    ImageType = itk.Image[PixelType, Dimension]
    InternalPixelType = itk.F  
    InternalImageType = itk.Image[InternalPixelType, Dimension] 
    
    image_folder = '/Users/yuncong/Documents/medical images/Test'
    fixed_image_filename = os.path.join(image_folder,'H4nissl_2 - 2010-03-29 15.16.30_x0.3125_z0_0.tif')
    moving_image_filename = os.path.join(image_folder,'H4nissl_2 - 2010-03-29 15.16.30_x0.3125_z0_1.tif')
    
#    image_folder = '/Users/yuncong/Documents/InsightToolkit-4.2.0/Examples/Data'
#    fixed_image_filename = os.path.join(image_folder,'BrainT1SliceBorder20.png')
#    moving_image_filename = os.path.join(image_folder, 'BrainProtonDensitySliceShifted13x17y.png')
    
    
#    fixed_image_filename = os.path.join(image_folder,'BrainProtonDensitySliceBorder20.png')
#    moving_image_filename = os.path.join(image_folder, 'BrainProtonDensitySliceShifted13x17y.png')
    
    fixedImageReader = itk.ImageFileReader[FixedImageType].New()
    fixedImageReader.SetFileName(fixed_image_filename)
    fixedImageReader.Update()
#    image = rd1.GetOutput()

    movingImageReader = itk.ImageFileReader[MovingImageType].New()
    movingImageReader.SetFileName(moving_image_filename)
    movingImageReader.Update()
    
    TransformType = itk.AffineTransform[itk.D, Dimension]
    OptimizerType = itk.GradientDescentOptimizer
    InterpolatorType = itk.LinearInterpolateImageFunction[InternalImageType, itk.D]
    RegistrationType = itk.ImageRegistrationMethod[InternalImageType, InternalImageType]
    MetricType = itk.MutualInformationImageToImageMetric[InternalImageType, InternalImageType]

    NormalizeFilterType = itk.NormalizeImageFilter[FixedImageType, InternalImageType]
    
    fixedNormalizer = NormalizeFilterType.New()
    movingNormalizer = NormalizeFilterType.New()
    
    fixedImage = fixedImageReader.GetOutput()
    movingImage = movingImageReader.GetOutput()
    
    fixedNormalizer.SetInput(  fixedImage)
    movingNormalizer.SetInput( movingImage)
    
    GaussianFilterType = itk.DiscreteGaussianImageFilter[InternalImageType,InternalImageType]
    fixedSmoother = GaussianFilterType.New()
    movingSmoother = GaussianFilterType.New()
    
    fixedSmoother.SetVariance( 2.0 )
    movingSmoother.SetVariance( 2.0 )
    
    fixedSmoother.SetInput( fixedNormalizer.GetOutput() )
    movingSmoother.SetInput( movingNormalizer.GetOutput() )
    
        
#    transform_type = itk.TranslationTransform[itk.D, 2]
#    optimizer_type = itk.RegularStepGradientDescentOptimizer
#    metric_type = itk.MeanSquaresImageToImageMetric[FixedImageType, MovingImageType]
#    interpolator_type = itk.LinearInterpolateImageFunction[MovingImageType, itk.D]
#    registration_type = itk.ImageRegistrationMethod[FixedImageType, MovingImageType]
    
    metric = MetricType.New()
    transform = TransformType.New()
    optimizer = OptimizerType.New()
    interpolator = InterpolatorType.New()
    registration = RegistrationType.New()
    
    registration.SetMetric( metric )
    registration.SetOptimizer( optimizer )
    registration.SetTransform( transform )
    registration.SetInterpolator( interpolator )

    fixedSmoother.Update()
    movingSmoother.Update()

    registration.SetFixedImage( fixedSmoother.GetOutput() )
    registration.SetMovingImage( movingSmoother.GetOutput() )
        
#    fixedImageReader.Update()
    fixedImageRegion = fixedNormalizer.GetOutput().GetBufferedRegion() 
    registration.SetFixedImageRegion(fixedImageRegion)
    
    initialParameters = itk.OptimizerParameters[itk.D]( transform.GetNumberOfParameters() )
    initialParameters[0] = 1.0
    initialParameters[1] = 0.0
    initialParameters[2] = 0.0
    initialParameters[3] = 1.0
    initialParameters[4] = 0.0
    initialParameters[5] = 0.0


    registration.SetInitialTransformParameters( initialParameters )
    
    metric.SetFixedImageStandardDeviation(0.4)
    metric.SetMovingImageStandardDeviation(0.4)
    
    numberOfPixels = fixedImageRegion.GetNumberOfPixels()
    numberOfSamples = int( numberOfPixels * 0.01 )
    
    
    def iterationUpdate():
        currentParameter = transform.GetParameters()
        print "M: %f   P: %f %f " % ( optimizer.GetValue(), currentParameter.GetElement(0),
                                       currentParameter.GetElement(1) )
    
    iterationCommand = itk.PyCommand.New()
    iterationCommand.SetCommandCallable( iterationUpdate )
    optimizer.AddObserver( itk.IterationEvent(), iterationCommand )
    
#    optimizer.SetMaximumStepLength( 4.00 )
#    optimizer.SetMinimumStepLength( 0.01 )
    
    metric.SetNumberOfSpatialSamples( numberOfSamples )
    
    optimizer.SetLearningRate( 0.001 )
    optimizer.SetNumberOfIterations( 300 )
    optimizer.MaximizeOn()
    
    print "Starting registration"
#    try:
    registration.Update()
#    except itk.ExceptionObject as err:
#        print "ExceptionObject caught: " + err + '\n';
#        return -1;
    
    finalParameters = registration.GetLastTransformParameters()
    
    print "Final Registration Parameters "
    
    numberOfIterations = optimizer.GetCurrentIteration()
    bestValue = optimizer.GetValue()
    
    ResampleFilterType = itk.ResampleImageFilter[MovingImageType, FixedImageType]
    resampler = ResampleFilterType.New()
    resampler.SetInput( movingImageReader.GetOutput() )
    resampler.SetTransform( registration.GetOutput().Get() )
    
    fixedImage = fixedImageReader.GetOutput()
    resampler.SetSize( fixedImage.GetLargestPossibleRegion().GetSize() )
    resampler.SetOutputOrigin( fixedImage.GetOrigin() )
    resampler.SetOutputSpacing( fixedImage.GetSpacing() )
    resampler.SetDefaultPixelValue( 100 )
    
    OutputPixelType = itk.UC
    OutputImageType = itk.Image[OutputPixelType, Dimension]
    CastFilterType = itk.CastImageFilter[FixedImageType, OutputImageType]
    WriterType = itk.ImageFileWriter[OutputImageType]
    
    writer = WriterType.New()
    caster = CastFilterType.New()
    
    caster.SetInput( resampler.GetOutput() )
    writer.SetInput( caster.GetOutput() )

    writer.SetInput( caster.GetOutput() )
    out_file = 'OutputImage.tif'
    writer.SetFileName(out_file)
    writer.Update()
    
    
#    sys.path.append('/Users/yuncong/Documents/workspace/Registration/src/root/sift')
#    
#    DifferenceFilterType = itk.SubtractImageFilter[FixedImageType,FixedImageType,FixedImageType]
#    difference = DifferenceFilterType.New()
#    difference.SetInput1( fixedImageReader.GetOutput() )
#    difference.SetInput2( resampler.GetOutput() )
#    
#    RescalerType = itk.RescaleIntensityImageFilter[FixedImageType,OutputImageType]
#    intensityRescaler = RescalerType.New()
#    intensityRescaler.SetInput( difference.GetOutput() )
#    intensityRescaler.SetOutputMinimum( 0 )
#    intensityRescaler.SetOutputMaximum( 255 )
#    resampler.SetDefaultPixelValue( 1 )
#    
#    writer2 = WriterType.New()
#    writer2.SetInput( intensityRescaler.GetOutput() )
#    
#    identityTransform = transform_type.New()
#    identityTransform.SetIdentity()
#    resampler.SetTransform( identityTransform )
    
    
    
    
    
    