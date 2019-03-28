#!/usr/bin/env python3
"""
ECE 471 Project
Ashton Meuser
"""

from time import time
import argparse as ap
import numpy as np
from convolve import convolve
import image_utils
from model.line_mask import generate_line_mask_list
from line_score import line_score

def main():
    """
    Run the program
    Avoids globals
    """
    parser = ap.ArgumentParser()
    parser.add_argument('-i', '--image', help='Path to image', required=True)
    parser.add_argument('-k', '--kernel', help='Window size', type=int, default=15)
    parser.add_argument('-r', '--resolution', help='Rotation resolution', type=int, default=15)
    parser.add_argument('-s', '--save', help='Save image', action='store_true')
    parser.add_argument('-d', '--display', help='Display image', action='store_true')
    parser.add_argument('-v', '--verbose', help='Verbose', action='store_true')
    args = parser.parse_args()

    start = time()

    img = image_utils.read_image(args.image)
    img = image_utils.get_inverse_green_channel(img)
    mask_list = generate_line_mask_list(args.kernel, args.resolution)
    function = lambda x: line_score(x, mask_list)
    result = convolve(img, args.kernel, function, verbose=args.verbose).astype(np.uint8)

    stop = time()

    if args.verbose:
        print('time elapsed: {:.2f}s'.format(stop - start))

    if args.save:
        image_utils.save_image(result, 'output.png')
    if args.display:
        image_utils.display_image(result)

if __name__ == '__main__':
    main()
