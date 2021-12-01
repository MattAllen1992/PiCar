import cv2
import numpy as np

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