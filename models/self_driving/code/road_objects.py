from threading import Timer
import logging

# base class for a traffic object
class RoadObject(object):
    def set_car_state(self, car_state):
        pass
    
    # this decorator defines is_close_by as a static method
    # this means it can be called without instantiating the class
    # essentially a regular method just kept in this file as a logical grouping
    @staticmethod
    def is_close_by(obj, frame_height, min_height_pct=0.05):
        # if object height is more than 5% of frame height, it's close by
        obj_height = obj.bounding_box[1][1]-obj.bounding_box[0][1]
        return obj_height / frame_height > min_height_pct

# stop when a red traffic light is detected
class RedTrafficLight(RoadObject):
    def set_car_state(self, car_state):
        logging.debug('red light: stopping car')
        car_state['speed'] = 0

# keep going when a green traffic light is detected
class GreenTrafficLight(RoadObject):
    def set_car_state(self, car_state):
        logging.debug('green light: make no changes')
 
# stop when a person is detected
class Person(RoadObject):
    def set_car_state(self, car_state):
        logging.debug('pedestrian: stopping car')
        car_state['speed'] = 0

# adjust speed limit based on detected sign/input
class SpeedLimmit(RoadObject):
    # store speed limit for reference
    def __init__(self, speed_limit):
        self.speed_limit = speed_limit
    
    # set speed limit for car
    def set_car_state(self, car_state):
        logging.debug('speed limit: set limit to %d' % self.speed_limit)
        car_state['speed_limit'] = self.speed_limit

# wait at stop sign and proceed after wait time has elapsed
class StopSign(RoadObject):
    # initialize stop and wait parameters
    def __init__(self, wait_time_in_sec=3, min_no_stop_sign=20):
        self.in_wait_mode = False
        self.has_stopped = False
        self.wait_time_in_sec = wait_time_in_sec
        sef.min_no_stop_sign = min_no_stop_sign
        self.timer = None
    
    # react to stop sign
    def set_car_state(self, car_state):
        
        self.no_stop_count = self.min_no_stop_sign
        
        # if already waiting, ensure car is stopped and return to continue waiting
        if self.in_wait_mode:
            logging.debug('stop sign: 2) still waiting')
            car_state['speed'] = 0
            return
        
        # if moving, stop the car, flag that it's stopped and waiting
        # and start timer to ensure we wait for specified time duration
        if not self.has_stopped:
            logging.debug('stop sign: 1) just detected, stop and wait')            
            car_state['speed'] = 0
            self.in_wait_mode = True
            self.has_stopped = True
            self.time = Timer(self.wait_time_in_sec, self.wait_done) # duration and function to call on exit
            self.time.start()
            return
    
    # called once wait is over, resets wait mode
    def wait_done(self):
        logging.debug('stop sign: 3) finished waiting for %d seconds' % self.wait_time_in_sec)
        self.in_wait_mode = False
    
    # if the camera glitches and doesn't detect a stop sign this function
    # ensures that we see 20 consecutive frames without a stop sign (~1 second)
    # before we clear and reset all parameters to the initial state
    def clear(self):
        if self.has_stopped:
            self.no_stop_counter -= 1
            if self.no_stop_count == 0:
                logging.debug('stop sign: 4) stop sign no longer detected')
                self.has_stopped = False
                self.in_wait_mode = False