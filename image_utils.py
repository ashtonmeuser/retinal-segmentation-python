"""
Common functions perfromed on images
"""

import cv2
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

def get_inverse_green_channel(image):
    """
    Read image green channel, invert
    """
    return cv2.bitwise_not(image[:, :, 1])

def save_image(image, image_path):
    """
    Save image to file at image_path
    """
    return cv2.imwrite(image_path, image)

def display_image(image):
    """
    Display image
    """
    plt.imshow(image, cmap='gray', interpolation='none')
    _ = plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

def pad_image(image, pad, value):
    """
    Apply a border of value and width pad around image
    """
    return cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=value)
