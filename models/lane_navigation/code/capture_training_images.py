import cv2
import os
import argparse

def main():
    # output directory to write images
    output_dir = '/home/pi/work/PiCar/models/lane_navigation/data/training_images/'
    
    parser = argparse.ArgumentParser(description='Capture training images for lane detection')
    parser.add_argument('--camera_idx', type=int, default=0,
                        help='Index of which video camera to use')
    args = parser.parse_args()
    
    # image counter for unique filenames
    img_counter = 0
    
    # start recording images
    camera = cv2.VideoCapture(args.camera_idx)    
    while camera.isOpened():
        # get image from camera
        ret, frame = camera.read()
        if ret == False:
            print('couldn\'t read camera images')
            break
        
        # save image to file
        img_name = 'img_' + str(img_counter) + '.png'
        img_path = os.path.join(output_dir, img_name)
        print('writing {}'.format(img_path))
        written = cv2.imwrite(img_path, frame)
        img_counter += 1 # increment counter for next image
        
        # close when user quits
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('user exited program')
            break
    
    # shutdown and close all sessions
    print('shutting down')
    camera.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()