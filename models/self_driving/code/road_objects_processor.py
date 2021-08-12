# core libraries
import cv2
import logging
import datetime
import time
from road_objects import *

# edge tpu and object detection libraries
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

# performs object detection and adjust car navigation accordingly
class RoadObjectsProcessor(object):

    ########## CAR NAVIGATION METHODS ##########
    
    # model must be tflite format and quantized/compiled specifically for edge tpu
    # https://coral.withgoogle.com/web-compiler/
    def __init__(self, car=None, speed_limit=40,
                 model='/home/pi/work/PiCar/models/object_detection/data/sample_model_results/road_signs_quantized_edgetpu.tflite',
                 labels='/home/pi/work/PiCar/models/object_detection/data/sample_model_results/road_sign_labels.txt',
                 width=640, height=480):
        # assign class car and image parameters
        logging.info('Initializing Object Detector...')
        self.width = width
        self.height = height        
        self.car = car
        self.speed_limit = speed_limit
        self.speed = speed_limit
        
        # load model and labels onto edge tpu and initialize parameters
        logging.info('Initializing Edge TPU with model %s and labels %s' % (model, labels))
        self.interpreter = make_interpreter(model)         # load model onto edge tpu
        self.interpreter.allocate_tensors()                # required to initialize interpreter
        self.labels = read_label_file(labels)              # extract labels from file
        self.inference_size = input_size(self.interpreter) # get (width, height) tuple of model's inputs        
        self.min_confidence = 0.30                         # label score must exceed 30% for a conident detection
        self.num_of_objects = 3                            # number of objects to detect per frame
        logging.info('Edge TPU initialization complete.')
        
        # configure object detection visuals (bounding box, labels etc.)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.bottomLeftCornerOfText = (10, height-10)
        self.fontScale = 1
        self.fontColor = (255, 255, 255) # white
        self.boxColor = (0, 0, 255) # red
        self.boxLineWidth = 1
        self.lineType = 2
        self.annotate_text = ""
        self.annotate_text_time = time.time()
        self.time_to_show_prediction = 1.0 # ms
        
        # map traffic object processing methods
        self.road_objects = {0: GreenTrafficLight(),
                             1: Person(),
                             2: RedTrafficLight(),
                             3: SpeedLimit(25),
                             4: SpeedLimit(40),
                             5: StopSign()}
    
    # perform object detection and navigate in response to objects
    def process_road_objects(self, frame):
        logging.debug('Performing object detection...')
        objects, final_frame = self.detect_objects(frame)
        self.control_car(objects)
        logging.debug('Object detection complete.')
        return final_frame
    
    # navigate in response to detected objects
    def control_car(self, objects):
        # update car state with latest speed
        logging.debug('Car reacting to detected objects...')
        car_state = {'speed': self.speed_limit, 'speed_limit': self.speed_limit}
        
        # do nothing if no objects detected
        if len(objects) == 0:
            logging.debug('No objects detected, continue at speed limit [%s]' % self.speed_limit)
        
        # process all detected objects
        contain_stop_sign = False
        for obj in objects:
            # identify object and get relevant navigation/processing method
            obj_label = self.labels.get(obj.id, obj.id)
            processor = self.traffic_objects[obj.label_id]
            
            # if the object is close enough to respond to, call appropriate navigation response
            if processor.is_close_by(obj, self.height):
                processor.set_car_state(car_state)
            # otherwise the object is too far away so ignore it
            else:
                logging.debug('[%s] object detected but too distant: IGNORE' % obj_label)
            
            # flag if a stop sign is detected
            if obj_label == 'Stop':
                contain_stop_sign = True
            
            # if no stop sign is detected for ~1 second, reset navigation parameters
            if not contain_stop_sign:
                self.traffic_objects[5].clear()
            
            # update speed according to latest car state
            self.resume_driving(car_state)
        
        # apply latest car state parameters (i.e. adjust speed as appropriate)
        def resume_driving(self, car_state):
            # track old speed and set new speed
            old_speed = self.speed
            self.speed_limit = car_state['speed_limit']
            self.speed = car_state['speed']
            
            # if 0, stop the car
            if self.speed == 0:
                self.set_speed(0)
            # otherwise adjust speed as appropriate
            else:
                self.set_speed(self.speed_limit)                
            logging.debug('Adjusting speed from %d to %d' % (old_speed, self.speed))
            
            # if 0, remain stopped for 1 second to allow
            # stop sign processing etc. to run its course
            if self.speed == 0:
                logging.debug('Stop for 1 second')
                time.sleep(1)
        
        # update car speed if available and store speed
        # variable regardless of whether or not a car exists        
        def set_speed(self, speed):
            self.speed = speed
            if self.car is not None:
                logging.debug('Setting car speed to %d' % speed)
                self.car.back_wheels.speed = speed
                
        ########## IMAGE PROCESSING METHODS ##########
        
        # perform object detection and overlay onto original frame
        # return detected objects for further use/analysis
        def detect_objects(self, frame):
            # adjust image for compatibility and optimal image detection
            logging.debug('Detecting objects...')            
            start_ms = time.time()
            img = frame
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.resize(img_rgb, self.inference_size)
            
            # perform object detection
            # only detect objects if >30% confident
            # extract 3 objects (maximum) per frame
            run_inference(self.interpreter, img_rgb.tobytes())
            objs = get_objects(self.interpreter, self.min_confidence)[:self.num_of_objects]
            
            # overlay bounding boxes, confidence scores and labels onto original image
            img_objs = append_objs_to_img(img_rgb, self.inference_size, objs, self.labels)
            inference_time = time.time() - start_ms
            logging.debug('.1f FPS' % (1.0 / inference_time))
            
            # return detected objects and image with overlayed objects
            return objs, img_objs
                    
        # overlay object detection bounding box, label and confidence/score onto original image
        # https://github.com/google-coral/examples-camera/blob/master/opencv/detect.py
        def append_objs_to_img(img, inference_size, objs, labels):
            # return raw image if no objects detected
            if len(objs) == 0:
                logging.debug('No objects detected...')
                return img
            
            # extract image properties and scale to match detected objects to raw image
            height, width, channels = img.shape
            scale_x, scale_y = width / inference_size[0], height / inference_size[1]
            
            # process all detected objects
            for obj in objs:
                # scale bounding box and extract key coordinates for overlay
                bbox = obj.bbox.scale(scale_x, scale_y)
                bottom_left = (int(bbox.xmin), int(bbox.ymin))
                top_right = (int(bbox.xmax), int(bbox.ymax))
                top_left_text = (int(bbox.xmin), int(bbox.ymax) + 15)
                
                # format score and label
                percent = int(10 * obj.score)
                obj_lbl = labels.get(obj.id, obj.id)
                label = '{}% {}'.format(percent, obj_lbl)
                logging.debug('Detected object [%s] with confidence [%s]' % (obj_lbl, percent))
                
                # overlay bounding box, score and label onto original image
                img = cv2.rectangle(img, bottom_left, top_right, self.boxColor, self.boxLineWidth)
                img = cv2.putText(img, label, top_left_text, self.fontColor, self.fontScale, self.boxColor, self.lineType)
            
            # return image with object detection overlay
            return img
        
########## UTILITY FUNCTIONS ##########
        
# display image on screen
def show_image(title, frame, show=False):
    if show:
        cv2.imshow(title, frame)
            
########## TEST FUNCTIONS ##########
        
# perform object detection on test image
def test_photo(file):
    processor = RoadObjectsProcessor()
    frame = cv2.imread(file)
    img_objs = processor.process_road_objects(frame)
    show_image('Detected Objects', img_objs)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# perform object detection on stop sign
def test_stop_sign(file):
    processor = RoadObjectsProcessor()
    frame = cv2.imread(file)
    
    for i in range(3):
        img_objs = processor.process_road_objects(frame)
        img_name = 'Stop {}'.format(i+1)
        show_image(img_name, img_objs)
        time.sleep(2)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# perform object detection on full video
def test_video(video):
    # load processor and video
    processor = RoadObjectsProcessor()
    camera = cv2.VideoCapture(video + '.avi')
    
    # ignore first 3 frames (whilst camera is initializing)
    for i in range(3):
        _, frame = camera.read()
    
    # configure writer for video with object detection overlay
    video_format = cv2.VideoWriter_fourcc(*'XVID')
    time_now = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    writer = cv2.VideoWriter('%s_overlay_%s.avi' % (video, time_now), video_format, 20.0, (320, 240)) # 20 fps
    
    try:
        i = 0
        while camera.isOpened():
            # write raw video frames to png file
            _, frame = camera.read()
            cv2.imwrite('%s_%03d.png' % (video, i), frame)
            
            # write frames with object detection to png and video files
            img_objs = processor.process_road_objects(frame)
            cv2.imwrite('%s_%03d.png' % (video, i), frame)
            writer.write(img_objs)
            
            # display video with object detection overlay
            cv2.imshow('Detected Objects', img_objs)
            
            # increment counter for file names and wait for shutdown/video end
            i += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # shutdown and close all sessions
    finally:
        camera.release()
        writer.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)-5s:%(asctime)s: %(message)s')
    
    # run tests on test images (no states, only images)
    test_photo('/home/pi/work/PiCar/models/self_driving/data/objects/red_light.jpg')
    test_photo('/home/pi/work/PiCar/models/self_driving/data/objects/green_light.jpg')
    test_photo('/home/pi/work/PiCar/models/self_driving/data/objects/person.jpg')
    test_photo('/home/pi/work/PiCar/models/self_driving/data/objects/limit_25.jpg')
    test_photo('/home/pi/work/PiCar/models/self_driving/data/objects/limit_40.jpg')
    test_photo('/home/pi/work/PiCar/models/self_driving/data/objects/no_obj.jpg')
    test_photo('/home/pi/work/PiCar/models/self_driving/data/objects/distant_obj.jpg')
    
    # run test on stop sign (has state)
    test_stop_sign('/home/pi/work/PiCar/models/self_driving/data/objects/stop_sign.jpg')