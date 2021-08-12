from car import Car
import logging
import sys

def main():
    # print out system info and start driving car at 40% speed
    logging.info('Starting Car, system info:' + sys.version)
    with Car() as car:
        car.drive(40)
        
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()