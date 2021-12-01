import logging
import math

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
        stabilized_steering_angle = int(curr_steering_angle + (max_angle_deviation * (angle_deviation / abs(angle_deviation)))) # ensure that we move left/right as required
    else:
        stabilized_steering_angle = new_steering_angle
    
    # return the stabilized steering angle
    logging.debug('Proposed angle: %s | Stabilized angle: %s' % (new_steering_angle, stabilized_steering_angle))
    return stabilized_steering_angle