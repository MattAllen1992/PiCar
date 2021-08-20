from car import Car
import logging
import sys

'''
main_test.py starts the car, using the provided video as a test input
it then applies lane following, object detection and navigation
'''

def main():
    # print out system info and start driving car at 40% speed
    logging.info('Starting Car, system info:' + sys.version)
    with Car(video_path='car/data/videos/study_test_video2.avi') as car:
        car.drive(40)
        
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()