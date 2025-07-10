#!/usr/bin/python

'''
Function to get contours of feet's for the foot-tracking app
by Rafal May 2025
''' 


# Set environment
import cv2
import skimage
import numpy as np
from datetime import datetime


def contour(mask):
    # Encode PyTorch tensor as cv2 ndarray
    _, png_image = cv2.imencode('.png', mask.squeeze())

    if _:
        # Decode the PNG image for processing/displaying 
        png_image = cv2.imdecode(png_image, cv2.IMREAD_GRAYSCALE)

        # Condition image and find contours
        png_image = skimage.exposure.rescale_intensity(png_image, in_range='image', out_range=(0, 255)).astype(np.uint8)
        #png_image = cv2.Canny(png_image, 50, 150)
        contours, _ = cv2.findContours(png_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        
        return contours, png_image
   

if __name__ == '__main__':

    # Test execution time
    start_time_frame = datetime.now() 
    mask = None
    contours, png_image = contour(mask)

    # Print processing time
    duration = datetime.now() - start_time_frame
    print('Processing_contour_time = ', 1000 * duration.total_seconds(), '[ms]')
    