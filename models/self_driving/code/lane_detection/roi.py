import cv2
import numpy as np
from images import show_image

# black out extract_fraction portion of image (if extract_fraction is 1/2, extract bottom half)
# this allows us to focus on the ROI (lane near camera) and ignore the background and distant noise
def region_of_interest(canny, extract_fraction):
    # create mask array matching image dimensions
    height, width = canny.shape
    mask = np.zeros_like(canny)
    
    # extract top half of image
    polygon = np.array([[(0, height * extract_fraction),
                         (width, height * extract_fraction),
                         (width, height),
                         (0, height)]], np.int32)
    
    # black out top half of image
    cv2.fillPoly(mask, polygon, 255)
    show_image('ROI Mask', mask, False)
    
    # apply mask to image to blackout top half
    masked_image = cv2.bitwise_and(canny, mask)
    return masked_image