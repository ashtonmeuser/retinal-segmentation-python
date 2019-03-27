#!/usr/bin/env python3
"""
ECE 471 Project
Ashton Meuser
"""

import argparse as ap
import numpy as np
from convolve import convolve
from image_utils import read_image, display_image, save_image

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
    result = convolve(img, 9, lambda x: 1).astype(np.uint8)

    if args.save:
        save_image(result, 'output.png')
    else:
        display_image(result)

if __name__ == '__main__':
    main()
