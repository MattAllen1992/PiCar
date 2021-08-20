import cv2
import numpy as np
import logging
import math
from time import sleep
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
        show_image('Raw Image', frame)
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

        # adjust cars steering if exists
        if self.car is not None:
            self.car.front_wheels.turn(self.curr_steering_angle)
            sleep(0.01) # sleep to ensure adjustments are made
            
        # display heading line as overlay and return image
        curr_heading_image = display_heading_line(frame, self.curr_steering_angle)
        show_image('Heading', curr_heading_image)                   
        return curr_heading_image

########## FRAME PROCESSING METHODS ##########

# perform lane detection
def detect_lane(frame):
    # identify all edges/boundaries using canny edge detection
    logging.debug('Performing lane detection...')
    edges = detect_edges(frame)
    show_image('Edge Detection', edges)
    
    # black out top half of image to focus on lanes
    # which occur in the bottom half from car's perspective
    cropped_edges = region_of_interest(edges)
    show_image('Cropped Edges', cropped_edges)
    
    # identify all left and right lane edges
    line_segments = detect_line_segments(cropped_edges)
    img_line_segments = display_lines(frame, line_segments)
    show_image('Line Segments', img_line_segments)
    
    # compute final left and right lanes (average of all available)
    lane_lines = average_slope_intercept(frame, line_segments)
    img_lane_lines = display_lines(frame, lane_lines)
    show_image('Lane Lines', img_lane_lines)
    
    # return lane line coordinates and image with lines overlayed
    return lane_lines, img_lane_lines

# canny edge detection
def detect_edges(frame):
    # convert image to HSV for ease of colour extraction
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    show_image('HSV', hsv)

    # blur image for better edge detection
    # 3, 3 strength of blur in x and y directions
    hsv_blur = cv2.GaussianBlur(hsv, (3, 3), 0) 
    
    # extract colour regions in the blue range (colour of lane lines)
    lower_blue = np.array([84, 55, 120]) # upper and lower bounds calibrated using colour_picker.py
    upper_blue = np.array([150, 215, 255])
    mask = cv2.inRange(hsv_blur, lower_blue, upper_blue)
    show_image('Blue Mask', mask)

    # convert to gray scale for simplicity of edge detection
    # can't go straight from hsv to gray, need rgb middle man
    # mask_rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2RGB)
    # mask_gray = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2GRAY)
    
    # perform canny edge detection
    # gaussian blur to reduce noise, retain/clarify edges
    # calculate gradients (Sobel edge detector kernel) to determine direction and likelihood of edge
    # non-max suppression to clarify and produce thin, crisp edges (suppress non maximum values along suspected edge)
    # hysteresis thresholding to determine final edges (looking at connections, pixel value vs. min and max thresholds)
    lower_threshold = 85 # calibrated using canny_edge_calibration.py
    upper_threshold = lower_threshold * 3 # recommended ratio in canny docs
    edges = cv2.Canny(mask, lower_threshold, upper_threshold)
    
    # return final edges
    return edges

# black out top half of image to focus on lanes only
# lanes occur in the bottom half from car's perspective
def region_of_interest(canny):
    # create mask array matching image dimensions
    height, width = canny.shape
    mask = np.zeros_like(canny)
    
    # extract top half of image
    polygon = np.array([[(0, height * 1 / 2),
                         (width, height * 1 / 2),
                         (width, height),
                         (0, height)]], np.int32)
    
    # black out top half of image
    cv2.fillPoly(mask, polygon, 255)
    show_image('ROI Mask', mask)
    
    # apply mask to image to blackout top half
    masked_image = cv2.bitwise_and(canny, mask)
    return masked_image

# perform probabilistic hough line detection to identify
# all line/edge segments in the cropped image
def detect_line_segments(cropped_edges):
    # hough parameters
    rho = 1             # distance resolution (i.e. 1 = look at each pixel)
    angle = np.pi / 180 # angle resolution (i.e. look at each radian)
    min_threshold = 10  # min # of votes/sin crossovers required to confidently define an edge
    minLineLength = 8   # min # of pixel length for a line to be valid
    maxLineGap = 4      # max # of pixels a gap in a line is allowed to contain to be valid (e.g. dashed lines)
    
    # extract definite lines from cropped edges image
    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, np.array([]), minLineLength, maxLineGap)
    return line_segments

# differentiate left and right lanes, calculating an average gradient for the final lanes
def average_slope_intercept(frame, line_segments):
    # return if no lanes detected, can't process
    lane_lines = []
    if line_segments is None:
        logging.info('No line segments detected...')
        return lane_lines
    
    # initialize storage for left and right lane lines
    height, width, _ = frame.shape
    left_fit = []
    right_fit = []
    
    # left lane occurs on left 2/3 of image
    # right lane occurs on right 2/3 of image
    boundary = 1 / 3
    left_region_boundary = width * (1 - boundary)
    right_region_boundary = width * boundary
    
    # process all line segments
    for seg in line_segments:
        # extract coordinate of segment
        for x1, y1, x2, y2 in seg:
            # ignore vertical lines, cannot handle infinie gradient in below math
            if x1 == x2:
                logging.info('Skipping vertical line segment (inf. gradient): %s' % seg)
                continue
            
            # draw straight line between start and end points
            # extract gradient and y-intercept of lane line
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            
            # positive gradient indicates left lane, negative indicates right lane
            # confirm left/right lane if it occurs in left/right 2/3 of image respectively
            # store lane in left/right lane list if it satisfies these criteria
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))
            
    # calculate the average gradient of the left and right lanes across all segments
    # calculate final lane line start and end points for each lane, store and return
    left_fit_average = np.average(left_fit, axis=0) # average the slope only
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))
        
    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))
        
    logging.debug('Lane Lines: %s' % lane_lines)
    return lane_lines

# calculate required steering angle to follow
# heading path/remain in centre of the detected lanes
def compute_steering_angle(frame, lane_lines):
    # return if no lanes detected, nothing to follow
    if len(lane_lines) == 0:
        logging.info('No lane lines detected...')
        return -90
    
    # if one lane is detected, simply follow that lane
    # cannot calculate an average path between two lanes
    height, width, _ = frame.shape
    if len(lane_lines) == 1:
        # simply extract single lane's x distance 
        logging.debug('Only detected 1 lane line, follow it: %s' % lane_lines[0])
        x1, _, x2, _ = lane_lines[0][0]
        x_offset = x2 - x1

    # if multiple lanes are detected, average the distance between them
    # and use this to calculate the heading path between the two
    else:
        # averae the x coordinates of the end points to find the middle
        _, _, left_x2, _ = lane_lines[0][0]
        _, _, right_x2, _ = lane_lines[1][0]
        #camera_mid_offset_percent = 0.02 # -0.03, 0.0, 0.03 = left, centre, right
        mid = int(width / 2) # get middle x value
        
        # calculate average x and subtract middle to find required adjustment
        # the car is facing straight ahead so the adjustment required is the diff to middle
        x_offset = (left_x2 + right_x2) / 2 - mid
    
    # draw line to middle of image
    # roi/lane only covers image bottom half
    y_offset = int(height / 2)
    
    # calculate required steering adjustment
    angle_to_mid_radian = math.atan(x_offset / y_offset)          # tan(angle) = opposite / adjacent (TOA)
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi) # convert from radians to degrees
    new_steering_angle = angle_to_mid_deg + 90                    # adjust angle relative to 90 degrees/straight ahead servo angle
    
    # return calculated steering angle
    logging.debug('New steering angle: %s' % new_steering_angle)
    return new_steering_angle

# if the new steering angle is too extreme, the car will turn dramatically left and right, bouncing from lane to lane
# this method ensures that the steering angle is never adjusted more than the max_angle_deviation in one go
# NOTE: this could be enhanced to use the history of steering adjustments (e.g. last n adjustments) to smoothen the steering even more
def stabilize_steering_angle(curr_steering_angle, new_steering_angle, num_lane_lines, max_angle_deviation_two_lanes=10, max_angle_deviation_one_lane=5):
    # set max_angle_deviation based on number of lanes detected
    # for 2 lanes we are more confident in our heading so allow more steering adjustments
    # for 1 lane we want minor changes until we see 2 lanes again and can more confidently adjust our course
    if num_lane_lines == 2:
        max_angle_deviation = max_angle_deviation_two_lanes
    else:
        max_angle_deviation = max_angle_deviation_one_lane
    
    # calcaulate required deviation to achieve new angle from current
    # if the adjustment exceeds the maximum allowed deviation
    # cap the angle to adjust by the maximum deviation only
    angle_deviation = new_steering_angle - curr_steering_angle
    if abs(angle_deviation) > max_angle_deviation:
        stabilized_steering_angle = int(curr_steering_angle + max_angle_deviation * angle_deviation / abs(angle_deviation)) # ensure that we move left/right as required
    else:
        stabilized_steering_angle = new_steering_angle
    
    # return the stabilized steering angle
    logging.info('Proposed angle: %s | Stabilized angle: %s' % (new_steering_angle, stabilized_steering_angle))
    return stabilized_steering_angle

########## HELPER FUNCTIONS ##########

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

# calculate length of specified line segment
# a2 = b2 + c2 (2 represents squared)
def length_of_line_segment(line):
    x1, y1, x2, y2 = line
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# display frame with title if allowed
def show_image(title, frame, show=True):
    if show:
        cv2.imshow(title, frame)

# calculate start and end points of line based on slope and y-intercept
def make_points(frame, line):
    # extract frame dimensions and line parameters
    height, width, _ = frame.shape
    slope, intercept = line
    
    # y start and end points are the bottom and middle of the frame
    y1 = height
    y2 = int(y1 / 2)
    
    # x start and end points are calculated as follows
    # equation of a straight line: y = mx + c
    # rearrange to find x:         x = (y - c) / m
    # finally apply max and min to ensure they don't exceed frame boundaries
    # line from right to left of frame would have x distance of "-width" (hence max(-width, ...))
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]

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

# run lane detection and calculation of steering angle/heading line on test video
# compute lane detection using the deep learning model and the manually coded version
def test_video(video):
    # initialize models and load video
    lane_follower_manual = LaneFollowerManual()
    lane_follower_ml = LaneFollowerML()
    camera = cv2.VideoCapture(video + '.avi')
    
    # skip first 3 frames (camera initializing)
    for i in range(3):
        _, frame = camera.read()
    
    # create writer to output avi at 20 fps
    video_format = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter('%s_lane_follower.avi' % video, video_format, 20.0, (320, 240)) # 20 fps
    
    try:
        # counter for storing/logging each frame
        i = 0
        while camera.isOpened():
            # read frame and perform both versions of lane detection on separate copies
            _, frame = camera.read()
            frame_copy = frame.copy()
            logging.info('Frame %s' % i)
            img_lanes_manual = lane_follower_manual.follow_lane(frame)
            img_lanes_ml = lane_follower_ml.follow_lane(frame_copy)
            
            # calculate difference between deep learning and manual models
            # write deep learning result to output video and show both model's lane detection results
            diff = lane_follower_manual.curr_steering_angle - lane_follower_ml.curr_steering_angle
            logging.info('[Desired=%3d] | [Model=%3d] | [Difference=%3d]' % lane_follower_manual.curr_steering_angle, lane_follower_ml.curr_steering_angle, diff)
            writer.write(img_lanes_ml)
            cv2.imshow('Manual Lane Detection', img_lanes_manual)
            cv2.imshow('Deep Learning Lane Detection', img_lanes_ml)
            
            # increment counter and quit when signal received
            i += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # shutdown and close all sessions
    finally:
        camera.release()
        writer.release()
        cv2.destroyAllWindows()

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)    
    test_video('/home/pi/work/PiCar/models/self_driving/data/lanes/test_lanes_video.avi')
    test_photo('/home/pi/work/PiCar/models/self_driving/data/lanes/test_lanes_image.jpg')