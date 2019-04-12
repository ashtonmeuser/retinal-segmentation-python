"""
Common functions perfromed on images
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_image(image_path, greyscale=False):
    """
    Read image as color or grayscale from path
    Returns numpy array
    """
    mode = cv2.IMREAD_GRAYSCALE if greyscale else cv2.IMREAD_COLOR
    image = cv2.imread(image_path, mode)

    if image is None:
        raise ValueError('Invalid image path {}'.format(image_path))

    return image

def as_greyscale(image):
    """
    Return greyscale image from color image
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def as_inverse_green(image):
    """
    Read image green channel, invert
    """
    return cv2.bitwise_not(image[:, :, 1])

def save_image(image, image_path):
    """
    Save image to file at image_path
    """
    if image.dtype == np.bool:
        image = image.astype(np.uint8) * 255
    return cv2.imwrite(image_path, image)

def display_image(image):
    """
    Display image
    """
    plt.imshow(image, cmap='gray', interpolation='none')
    _ = plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
