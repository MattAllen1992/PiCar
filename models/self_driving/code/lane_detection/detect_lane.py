import logging
from edge_detection import detect_edges_canny
from roi import region_of_interest
from line_segments import detect_line_segments
from maths import average_slope_intercept
from images import show_image, display_lines

# perform lane detection
def detect_lane(frame):
    # identify all edges/boundaries using canny edge detection
    logging.debug('Performing lane detection...')
    edges = detect_edges_canny(frame)
    show_image('Edge Detection', edges, False)
    
    # black out top half of image to focus on lanes
    # which occur in the bottom half from car's perspective
    cropped_edges = region_of_interest(edges, 2/3)
    show_image('Cropped Edges', cropped_edges, False)
    
    # identify all left and right lane edges
    line_segments = detect_line_segments(cropped_edges)
    img_line_segments = display_lines(frame, line_segments)
    show_image('Line Segments', img_line_segments, False)
    
    # compute final left and right lanes (average of all available)
    lane_lines = average_slope_intercept(frame, line_segments)
    img_lane_lines = display_lines(frame, lane_lines)
    show_image('Lane Lines', img_lane_lines)
    
    # return lane line coordinates and image with lines overlayed
    return lane_lines, img_lane_lines