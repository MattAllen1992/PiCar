# standard libraries
from PIL import Image
import numpy as np
import datetime
import argparse # helps build cmd interfaces
import time
import cv2
import os

# edge tpu and object detection libraries
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

def main():
    # provide model (tflite format) and labels (text file of index to label) paths as args
    default_dir = '/home/pi/work/PiCar/models/object_detection'
    default_model = '/home/pi/work/PiCar/models/object_detection/data/results/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    default_labels = '/home/pi/work/PiCar/models/object_detection/data/results/coco_labels.txt'
    
    # build a cmd interface allowing the provision of object detectin data files and parameters
    # use "object_detection_demo.py -h" in the cmd to get help and suggestions for arguments
    parser = argparse.ArgumentParser(description='Build object detection model by providing model and labels.')
    parser.add_argument('--model', required=False, default=os.path.join(default_dir, default_model),
                        help='File path of .tflite model')
    parser.add_argument('--labels', required=False, default=os.path.join(default_dir, default_labels),
                        help='File path of labels file')
    parser.add_argument('--top_k', type=int, default=3,
                        help='Number of categories with highest score to display')
    parser.add_argument('--camera_idx', type=int, default=0,
                        help='Index of which video camera to use')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='Classifier minimum score threshold')
    args = parser.parse_args() # get args from cmd (or use defaults if not provided)
    
    # load model into edge tpu interpreter and labels into model
    print('Loading {} model with {} labels.'.format(args.model, args.labels))
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()           # required to initialize interpreter
    labels = read_label_file(args.labels)    # extract labels from file
    inference_size = input_size(interpreter) # get (width, height) tuple of model's inputs
    
    # start recording images with camera
    camera = cv2.VideoCapture(args.camera_idx)
    
    # video writer to save object detection run
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
    
    try:
        # while capturing images
        while camera.isOpened():
            try:
                # capture images
                start_ms = time.time()
                ret, frame = camera.read() # read frame from camera, ret is true if ok, false if no frames returned
                if ret == False:
                    print('cannot read camera images')
                    break
                
                # adjust image and perform object detection
                img = frame
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # adjust rgb colorspace - https://machinelearningknowledge.ai/opencv-tutorial-image-colorspace-conversion-using-cv2-cvtcolor/
                img_rgb = cv2.resize(img_rgb, inference_size) # resize image to match model input dimensions
                run_inference(interpreter, img_rgb.tobytes()) # run img (uint8 1D array) through model and make predictions/inferences
                objs = get_objects(interpreter, args.threshold)[:args.top_k] # get top_k labels above threshold prediction value for each object (object id, score and Bbox)
                img_with_obj = attach_objs_to_img(img_rgb, inference_size, objs, labels)
                
                # write image to output video
                #writer.write(img_with_obj)
                
                # show image with object detection (boxes, labels, score)
                # stop recording images when user closes the terminal
                cv2.imshow('frame', img_rgb)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                
            except:
                # print stacktrace but don't close camera/break loop
                print('couldn\'t process image')
                traceback.print_exc()
                
    finally:
        # close camera, writer and session
        print('closing camera, halting object detection')
        camera.release()
        writer.release()
        cv2.destroyAllWindows()
            
# attach detected objects (id, score and bounding box) to image
def attach_objs_to_img(img, inference_size, objs, labels):
    # extract image parameters and calculate scale of image to model tensor/matrix/array
    height, width, channels = img.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    
    # iterate through each detected object and apply id, label and score
    for obj in objs:
        # create object's bounding box using calculated scale (ensure box is correct size)
        # get bottom left and top right coordinates for creating rectangle later on
        bbox = obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)
        
        # convert confidence score into % and build final label
        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))
        
        # build bounding box rectangle and add to image
        img = cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2) # image to add box to, bottom left, top right, colour (green), line width
        img = cv2.putText(img, label, (x0, y0+30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2) # add label to image (location, font, fontsize, colour and line width
        
        # return image with bounding box, label and score attached
        return img

# invoke main method
if __name__ == '__main__':
    main()