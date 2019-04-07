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
from create_fov_mask import create_fov_mask
from classify import train, classify, assess
from normalize_features import normalize_features

def main():
    """
    Run the program
    Avoids globals
    """
    parser = ap.ArgumentParser()
    parser.add_argument('-i', '--image', help='Image number from database', type=int, required=True)
    parser.add_argument('-k', '--kernel', help='Window size', type=int, default=15)
    parser.add_argument('-r', '--rotation', help='Rotational resolution', type=int, default=15)
    parser.add_argument('-s', '--save', help='Save image', action='store_true')
    parser.add_argument('-t', '--train', help='Train SVM model', action='store_true')
    parser.add_argument('-d', '--display', help='Display image', action='store_true')
    parser.add_argument('-v', '--verbose', help='Verbose', action='store_true')
    args = parser.parse_args()

    start = time()

    image = image_utils.read_image('DRIVE/image/{:02d}_test.tif'.format(args.image)) # Full color
    inverse_green = image_utils.as_inverse_green(image) # Line detector applied to inverse green
    fov_mask = create_fov_mask('DRIVE/mask/{:02d}_test_mask.tif'.format(args.image), args.kernel)
    mask_list = generate_line_mask_list(args.kernel, args.rotation)
    function = lambda x, y: line_score(x, y, mask_list) # Function to apply to each neighborhood
    result = convolve(inverse_green, args.kernel, function, fov_mask, 2, verbose=args.verbose)
    greyscale = image_utils.as_greyscale(image) # Final feature vector
    vectors = np.dstack((result, greyscale)) # Union of all feature vectors
    vectors = normalize_features(vectors)
    truth = image_utils.read_image('DRIVE/truth/{:02d}_test_truth.tif'.format(args.image),
                                   greyscale=True).astype(np.bool)

    if args.train:
        train(vectors, truth) # Train SVM, lengthy process
    prediction = classify(vectors)
    assess(truth, prediction)

    stop = time()

    if args.verbose:
        print('time elapsed: {:.2f}s'.format(stop - start))

    if args.save:
        image_utils.save_image(prediction, 'prediction.png')
    if args.display:
        image_utils.display_image(prediction)

if __name__ == '__main__':
    main()
