import cv2
import logging
import numpy as np

def main():
    # read in image and
    frame = cv2.imread('/home/pi/work/PiCar/models/lane_navigation/data/training_images/img_0.png')
    
    # show raw image
    cv2.imshow('raw image', frame)
    
    # detect and return lane edges
    edges = detect_edges(frame)
    
    # extract edges in bottom half of image only
    cropped_edges = get_region_of_interest(edges, 2/3)
    
    # extract absolute lane lines
    line_segments = detect_line_segments(cropped_edges)
    
    # determine final, averaged left and right lanes
    lane_lines = average_lane_lines(frame, line_segments)
    
    # show original image with hough lines overlayed
    ls_overlay = draw_lines(frame, line_segments)
    cv2.imshow("hough line", ls_overlay)
    
    # show original image with final lane lines overlayed
    ll_overlay = draw_lines(frame, lane_lines)
    cv2.imshow("final lane lines", ll_overlay)
    
    # shutdown and close sessions
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# detect edges within the image
def detect_edges(frame):
    # convert input image to HSV (from RGB to 0-180 hsv scale)
    # this allows us to extract lanes based on colour alone (avoids lighting issues etc.)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # lanes are blue in the images so extract the blue range of the spectrum
    # blue falls somewhere between 60 and 150 on a 180 degree hue scale
    lower_blue = np.array([70, 40, 40])
    upper_blue = np.array([150, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue) # extract from 90-150 from full 180 hsv range

    # show blue areas of image
    cv2.imshow('blue mask', mask)
    
    # use canny algorithm to detect, clarify and extract edges
    # 200 and 400 are the lower and upper threshold values used in hysteresis threshold procedure
    edges = cv2.Canny(mask, 200, 400)
    
    # return extracted edges
    return edges
    
# black out the top half of the image and return edges in the bottom half only
def get_region_of_interest(edges, extract_fraction):
    # extract dimensions of provided image with edges detected
    height, width = edges.shape
    
    # create a numpy array of 0s which has the same
    # properties/dimensions etc. as the edges image
    mask = np.zeros_like(edges)
    
    # extract the bottom extract_fraction of the image, this is all we're interested in for lanes
    # (imagine the car's perspective, lanes originate from the bottom)
    # np.array defines the four corners of the polygon (bl, br, tr, tl)
    # this is essentially isolating the top half of the image
    polygon = np.array([[(0, height * extract_fraction),
                        (width, height * extract_fraction),
                        (width, height),
                        (0, height),]],
                      np.int32) # define array value type as int32
    
    # polygon defines top portion of the image
    # we then black out this polygon to act as a mask
    # (255=black, polygon=top half, mask=array covering whole image)
    # this can be combined with the original image to black out the top half
    cv2.fillPoly(mask, polygon, 255)
    
    # apply mask to original image to leave us with edges/none-zero
    # pixel values for the bottom half of the image only
    cropped_edges = cv2.bitwise_and(edges, mask)
    
    # show and return cropped edges
    cv2.imshow("cropped edges", cropped_edges)
    return cropped_edges

# identify complete lines for lane borders by combining proximal, partial lines to form solid lane edges
# returned line segments consist of (x1, y1) and (x2, y2) for each segment (i.e. start and end points of lines
def detect_line_segments(cropped_edges):
    # define parameters
    rho = 1
    angle = np.pi / 180
    min_threshold = 10
    
    # detect line segments using probabilistic algo (decreased computation, lower threshold)
    line_segments = cv2.HoughLinesP(cropped_edges,   # input image with edges
                                    rho,             # distance precision in pixels (i.e. search 1 pixel at a time) 
                                    angle,           # angle precision in radians (i.e. search 1 degree at a time)
                                    min_threshold,   # min # of votes/intersections of sin functions to be a valid edge
                                    np.array([]),    # 
                                    minLineLength=8, # lines must be >8 pixels long to be a valid edge
                                    maxLineGap=4)    # gaps between lines cannot exceed 4 pixels to be a valid edge (e.g. for dashed lines)
    
    # return completed line segments
    return line_segments

# iterate through lines and draw them onto the frame
def draw_lines(frame, lines, line_colour=(0, 255, 0), line_width=2):
    # create base array of 0s in same dimensions as input frame
    line_image = np.zeros_like(frame)
    
    # iterate through lines and draw them into same dimensional space as input frame
    if lines is not None:
        for i in range(0, len(lines)):
            x1, y1, x2, y2 = lines[i][0]
            cv2.line(line_image, (x1, y1), (x2, y2), line_colour, line_width)
    
    # calculate weighted sum of 2 images
    # original image at 80% with full opacity lines on top
    # img 1 * alpha + img 2 * beta + gamma
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1) # final 1 is a gamma added to each sum
    return line_image
    
# determine a final pair of left and right lanes per frame
# by averaging the gradients and start/end points of component lane segments
# differentiate left and right lanes based on gradient and area of the frame
def average_lane_lines(frame, line_segments):
    # create var to store final determined lanes
    # return if no line segments provided/detected
    lane_lines = []
    if line_segments is None:
        logging.info('no line segments detected')
        return lane_lines
    
    # extract height and width of input image
    # create vars to store left and right determined lane lines
    height, width, _ = frame.shape # don't care about channels
    left_fit = []
    right_fit = []
    
    # left lane should be on left 2/3 of screen, right lane should be on right 2/3 of screen
    boundary = 1/3
    left_region_boundary = width * (1 - boundary)
    right_region_boundary = width * boundary
    
    # iterate through line segments to check gradient to determine if its a left or right lane
    for seg in line_segments:
        # for start and end point of each line segment
        for x1, y1, x2, y2 in seg:
            # ignore vertical lines (could use polar arithmetic here instead of skipping)
            if x1 == x2:
                logging.info('skipping vertical line segment (slope=inf): %s' % seg)
                continue
            
            # fit a straight line (degree = 1) to start and endpoints of line segment and extract gradient and y-intercept
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            
            # if gradient is positive and it's on the left 2/3 of the image, it's a left lane
            # if gradient is negative and it's on the right 2/3 of the image, it's a right lane
            # store left/right lanes in appropriate list, keeping gradient and intercept for later calc
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))
    
    # average the gradients of all left lane lines and store the start and end points
    # the car can wander out of lane, resulting in only one visible lane in an image, hence the ">0" check below
    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))
    
    # do the same for the right lane lines
    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))
    
    # log output of the determined lane lines
    logging.debug('lane lines: %s' % lane_lines)
    
    # return final, averaged lane lines
    return lane_lines

# build and return start and end points of line based on frame size and line gradient and intercept
def make_points(frame, line):
    # extract frame dimensions and line parameters
    height, width, _ = frame.shape
    slope, intercept = line
    
    # get top and middle of image y-axis
    y1 = height
    y2 = int(y1 * 1 / 2)
    
    # calculate x coordinates based on "x = y - c / m"
    # ensure that coordinates are within frame boundaries
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]

if __name__ == '__main__':
    main()