"""
Resize DR HAGIS images for faster processing
"""

import cv2

def resize_images(): # pylint: disable=R0914
    """
    Loop through TIFF converted DR HAGIS database, resize all images
    """
    input_directories = ['Fundus_Images', 'Mask_Images', 'Manual_Segmentations']
    output_directories = ['image', 'mask', 'truth']
    prefixes = ['', '_mask_orig', '_manual_orig']
    output_width = 584

    for index in range(3):
        for image_number in range(1, 41):
            padded_number = '{:02d}'.format(image_number)
            filename = 'DRHAGIS/{}/{}{}.tiff'.format(input_directories[index], image_number,
                                                     prefixes[index])
            print(filename)
            mode = cv2.IMREAD_COLOR if index == 0 else cv2.IMREAD_GRAYSCALE
            interpolation = cv2.INTER_LINEAR if index == 0 else cv2.INTER_NEAREST
            original = cv2.imread(filename, mode)
            height, width = original.shape[:2]
            scale = output_width / width
            output_dimensions = (int(width * scale), int(height * scale))
            scaled = cv2.resize(original, output_dimensions, interpolation=interpolation)
            cv2.imwrite('DRHAGIS/{}/{}.tif'.format(output_directories[index], padded_number),
                        scaled)

if __name__ == '__main__':
    resize_images()
