#!/usr/bin/env python3
"""
ECE 471 Project
Ashton Meuser
"""

import argparse as ap
import matplotlib.pyplot as plt
import cv2
from convolve import convolve

def read_image(image_path, greyscale=False):
    """
    Read image as grayscale from path
    Returns numpy array
    """
    mode = cv2.IMREAD_GRAYSCALE if greyscale else cv2.IMREAD_COLOR
    image = cv2.imread(image_path, mode)

    if image is None:
        raise ValueError('Invalid image path {}'.format(image_path))

    return image

def save_image(image_to_save, image_path):
    """
    Save image_to_save to file at image_path
    """
    return cv2.imwrite(image_path, image_to_save)

def show_image(image_to_show):
    """
    Display image_to_show
    Wait for user to press key before mooving on
    """
    plt.imshow(image_to_show, cmap='gray', interpolation='bicubic')
    _ = plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

def main():
    """
    Run the program
    Avoids globals
    """
    parser = ap.ArgumentParser()
    parser.add_argument('-i', '--image', help='Path to image', required=True)
    parser.add_argument('-s', '--save', help='Save image', action='store_true')
    args = parser.parse_args()

    img = read_image(args.image, greyscale=True)
    result = convolve(img, 9, lambda x: 1)

    if args.save:
        save_image(result, 'output.png')
    else:
        show_image(result)

if __name__ == '__main__':
    main()
