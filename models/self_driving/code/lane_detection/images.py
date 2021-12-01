import cv2
import math
import numpy as np

# display frame with title if allowed
def show_image(title, frame, show=True):
    if show:
        cv2.imshow(title, frame)
        
# draw lines following the provided coordinates and overlay onto frame
def display_lines(frame, lines, line_color=(0, 255, 0), line_width=5):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image

# calculate heading line to keep the car in the middle of the detected lanes
# overlay the heading line on top of the input image and return result
def display_heading_line(frame, steering_angle, line_color=(0, 0, 255), line_width=5):
    # NOTE - steering angles:
    # left = 0-89 degrees
    # right = 91-180 degrees
    # straight = 90 degrees
    
    # create mask/template to add heading path to
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape
    
    # convert to radians and calculate start and end point of heading path
    # r = d / 180 * pi (https://www.cuemath.com/geometry/radians-to-degrees)
    steering_angle_radian = steering_angle / 180.0 * math.pi
    
    # start point is image bottom centre
    x1 = int(width / 2)
    y1 = height # shouldn't this be 0? i.e. start from bottom of image?
    
    # end point is calculated with trigonometry
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)
    
    # draw heading path and overlay on top of input image
    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)
    
    # return image with heading path overlayed
    return heading_image