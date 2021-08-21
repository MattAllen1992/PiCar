from car import Car
import logging
import sys

'''
main.py starts the car, using the camera to record live video input
it then applies lane following, object detection and navigation
'''

def main():
    # print out system info and start driving car at 40% speed
    logging.info('Starting Car, system info:' + sys.version)
    with Car() as car:
        car.drive(25)
        
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()