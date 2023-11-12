import numpy as np
import cv2
import image_proposing as imp

img_initial = imp.img_normalization(cv2.imread('./image.png',cv2.IMREAD_GRAYSCALE))

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
    image1 = (image1 * 255.0).astype(np.uint8)
    image2 = (image2 * 255.0).astype(np.uint8)
    mse = np.mean((image1 - image2)**2)
    psnrValue = 10 * np.log10((maxPixelValue ** 2) / mse)
    return psnrValue