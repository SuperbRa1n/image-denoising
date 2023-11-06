import numpy as np

def psnr(image1, image2, maxPixelValue):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        image1: The first image for comparison.
        image2: The second image for comparison.
        maxPixelValue: The maximum pixel value of the images.

    Returns:
        The PSNR value, which measures the quality of the images based on their mean squared error.
    """
    mse = np.mean(np.square((image1[:] - image2[:])))
    psnrValue = 10 * np.log10((maxPixelValue ** 2) / mse)
    return psnrValue

