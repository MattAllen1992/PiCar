import cv2
import os
from datetime import datetime

# save frames to mp4 video file at provided path
def save_video():
    # get video from camera
    cap = cv2.VideoCapture(0)
    
    # extract image dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # create writer, specifying output path and video format
    #now = datetime.now()
    #now = now.strftime('%m%d%Y_%H%M%S')
    video_path = '../data/videos/study_test_video3.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(video_path, fourcc, 20.0, (width, height)) # 20 fps
    
    # while capturing images/video
    try:
        while cap.isOpened():
            print("Running!")
            # process images
            ret, frame = cap.read()
            if ret == True:
                # write image to video and show
                writer.write(frame)
                cv2.imshow('Raw Image', frame)
                
                # close when user terminates
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # break if can't read camera images
            else:
                print('Can\'t read camera images.')
                break
    
    # shutdown and close all sessions
    finally:
        print("Shutting down!")
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        
if __name__ == '__main__':
    save_video()