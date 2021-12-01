import logging
import numpy as np
import math

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

# calculate length of specified line segment
# a2 = b2 + c2 (2 represents squared)
def length_of_line_segment(line):
    x1, y1, x2, y2 = line
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)