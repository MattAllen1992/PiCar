import cv2
import sys
from lane_follower_manual import LaneFollowerManual

# save frames to avi video file at provided path
def save_image_and_steering_angle(video_file):
    lane_follower_manual = LaneFollowerManual()
    video = cv2.VideoCapture(video_file + '.avi')
    
    try:
        i = 0
        while video.isOpened():
            # write image to path with incremented index for each new image
            # save steering angle in file name to save us having to store a map of image name to steering angle
            _, frame = video.read()
            cv2.imwrite("%s_%03d_%03d.png" % (video_file, i, lane_follower_manual.curr_steering_angle), frame)
            i += 1
            
            # close when signal received
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # shutdown and close all sessions
    finally:
        video.release()
        cv2.destroyAllWindows()
        
if __name__ == '__main__':
    save_image_and_steering_angle(sys.argv[1])