#!/usr/bin/env python3
"""
ECE 471 Project
Ashton Meuser
"""

import argparse as ap
import matplotlib.pyplot as plt
import cv2
from convolve import convolve

def read_image(image_path):
    """
    Read image as grayscale from path
    Returns numpy array
    """
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

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
    parser.add_argument('-i', '--image', help='Path to image', default='pear.png')
    parser.add_argument('-d', '--display', help='Display image', action='store_true')
    args = parser.parse_args()

    # Get grayscale image
    img = read_image(args.image)
    if args.display:
        show_image(img)

    # Sharpen image and return result
    res = convolve(img, 9, (lambda x: x))
    if args.display:
        show_image(res)

    save_image(res, 'sharpened.png')

if __name__ == '__main__':
    main()
