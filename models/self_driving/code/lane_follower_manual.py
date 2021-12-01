import cv2
import numpy as np
import logging
import math
from time import sleep
from lane_detection.images import show_image, display_heading_line
from lane_detection.edge_detection import detect_lane
from lane_detection.steering import compute_steering_angle, stabilize_steering_angle
#from lane_follower_ml import LaneFollowerML

'''
Manual Lane Detection and Navigation
    This module uses manually configured parameters to perform
    lane detection and navigation to keep the car in the centre
    of the detected lanes. Parameters for Canny Edge Detection and
    Hough Line Transformation in particular are all hard coded here.
    
    This model is effective for a basic set of lanes using blue lane lines
    and the small set of road objects defined in the original implementation.
    However, if the objects, lane lines or other conditions change then this
    model will become less effective and require significant adjustments.
'''

# this class manually calculates the heading path required to stay in the
# centre of the detected lanes, using trigonometry rather than a trained
# deep learning model to determine the required steering angle to follow
class LaneFollowerManual(object):
    # initialize and calibrate the car to point straight ahead
    def __init__(self, car=None):
        logging.info('Initializing Manual Lane Follower...')
        self.car = car
        self.curr_steering_angle = 90 # straight ahead
        sleep(0.01) # sleep to ensure adjustments are made
    
    # peform lane detection and adjust steering accordingly
    def follow_lane(self, frame):
        show_image('Raw Image', frame, False)
        lane_lines, frame = detect_lane(frame)
        final_frame = self.steer(frame, lane_lines)        
        return final_frame
   
    # determine steering adjustment required to follow central heading line
    def steer(self, frame, lane_lines):
        # return if no lanes detected, can't determine steering angle
        logging.debug('Steering...')
        if len(lane_lines) == 0:
           logging.error('No lane lines detected...')
           return frame
        
        # compute required steering angle and stabilize if required (for extreme, sudden steering adjustments)
        new_steering_angle = compute_steering_angle(frame, lane_lines)
        self.curr_steering_angle = stabilize_steering_angle(self.curr_steering_angle, new_steering_angle, len(lane_lines))
        #self.curr_steering_angle = new_steering_angle

        # adjust cars steering if exists
        if self.car is not None:
            self.car.front_wheels.turn(self.curr_steering_angle)
            sleep(0.01) # sleep to ensure adjustments are made
            
        # display heading line as overlay and return image
        curr_heading_image = display_heading_line(frame, self.curr_steering_angle)
        show_image('Heading', curr_heading_image)                   
        return curr_heading_image

########## TEST FUNCTIONS ##########

# load a test image, run lane detection, display lanes and calculated heading path on raw image
def test_photo(file):
    # read image and perform lane detection
    lane_follower = LaneFollowerManual()
    frame = cv2.imread(file)
    img_lanes = lane_follower.follow_lane(frame)
    
    # overlay lanes and heading line on image and disokay
    show_image('Deep Learning Lane Detection', img_lanes, True)
    logging.info('Image=%s, Suggested Steering Angle=%3d' % (file, lane_follower.curr_steering_angle))
    
    # shutdown and close sessions
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# # run lane detection and calculation of steering angle/heading line on test video
# # compute lane detection using the deep learning model and the manually coded version
# def test_video(video):
#     # initialize models and load video
#     lane_follower_manual = LaneFollowerManual()
#     lane_follower_ml = LaneFollowerML()
#     camera = cv2.VideoCapture(video + '.avi')
    
#     # skip first 3 frames (camera initializing)
#     for i in range(3):
#         _, frame = camera.read()
    
#     # create writer to output avi at 20 fps
#     video_format = cv2.VideoWriter_fourcc(*'XVID')
#     writer = cv2.VideoWriter('%s_lane_follower.avi' % video, video_format, 20.0, (320, 240)) # 20 fps
    
#     try:
#         # counter for storing/logging each frame
#         i = 0
#         while camera.isOpened():
#             # read frame and perform both versions of lane detection on separate copies
#             _, frame = camera.read()
#             frame_copy = frame.copy()
#             logging.info('Frame %s' % i)
#             img_lanes_manual = lane_follower_manual.follow_lane(frame)
#             img_lanes_ml = lane_follower_ml.follow_lane(frame_copy)
            
#             # calculate difference between deep learning and manual models
#             # write deep learning result to output video and show both model's lane detection results
#             diff = lane_follower_manual.curr_steering_angle - lane_follower_ml.curr_steering_angle
#             logging.info('[Desired=%3d] | [Model=%3d] | [Difference=%3d]' % lane_follower_manual.curr_steering_angle, lane_follower_ml.curr_steering_angle, diff)
#             writer.write(img_lanes_ml)
#             cv2.imshow('Manual Lane Detection', img_lanes_manual)
#             cv2.imshow('Deep Learning Lane Detection', img_lanes_ml)
            
#             # increment counter and quit when signal received
#             i += 1
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
    
#     # shutdown and close all sessions
#     finally:
#         camera.release()
#         writer.release()
#         cv2.destroyAllWindows()

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)    
    #test_video('/home/pi/work/PiCar/models/self_driving/data/lanes/test_lanes_video.avi')
    test_photo('/home/pi/work/PiCar/models/self_driving/data/lanes/test_lanes_image.jpg')