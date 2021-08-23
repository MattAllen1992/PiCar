from picar import Servo
import time

def main():
    pan_servo = Servo.Servo(1)
    tilt_servo = Servo.Servo(2)
    try:
        tilt_servo.write(0)
        print('Set to 0')
        time.sleep(2)
        tilt_servo.write(180)
        print('Set to 180')
        time.sleep(2)
        tilt_servo.write(90)
        print('Set to 90')
        time.sleep(2)
        tilt_servo.write(135)
        print('Set to 135')
        time.sleep(2)
        
    except KeyboardInterrupt:
        # reset camera to 0 if user interrupts
        tilt_servo.write(0)
    
    finally:
        # reset camera to 0 when finished
        tilt_servo.write(0)

if __name__=='__main__':
    main()