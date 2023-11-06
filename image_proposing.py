import numpy as np

def img_guassian(image, mu, sigma):
    """
    Add Gaussian noise to the input image.

    Args:
        image: The input image to which Gaussian noise will be added.
        mu: The mean of the Gaussian distribution.
        sigma: The standard deviation of the Gaussian distribution.

    Returns:
        The input image with Gaussian noise added.
    """
    return image + np.random.normal(mu, sigma, image.shape)


# 归一化处理
def img_normalization(img):
    """
    Normalize the input image by scaling its pixel values between 0 and 1.

    Args:
        img: The input image to be normalized.

    Returns:
        The normalized image with pixel values scaled between 0 and 1.
    """
    maxu = np.max(img[:])
    minu = np.min(img[:])
    return (img - minu) / (maxu - minu)
