from picar import Servo
import time

def main():
    front_wheels = picar.Front_Wheels()
    #front_wheels = Servo.Servo(0)
    for _ in range(5):
        front_wheels.turn_left()
        print('Turn left')
        time.sleep(5)
        front_wheels.turn_right()
        print('Turn right')
        time.sleep(5)
        front_wheels.turn_straight()
        print('Turn straight')
        time.sleep(5)
        front_wheels.turn(45)
        print('Turn 45 degrees')
        time.sleep(5)
        front_wheels.turn(90)
        print('Turn 90 degrees')
        time.sleep(5)
        front_wheels.turn(135)
        print('Turn 135 degrees')
        time.sleep(5)

if __name__=='__main__':
    main()