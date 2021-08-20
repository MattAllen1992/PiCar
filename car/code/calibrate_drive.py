from picar import Servo
import time

def main():
    back_wheels = picar.Back_Wheels()
    #back_wheels = Servo.Servo(0)
    for _ in range(5):
        # accelerate from 0 to 100%
        for i in range(100):
            back_wheels.speed = i
            print('Driving at %s%' % i)

        # drive at various speeds
        time.sleep(5)
        back_wheels.speed = 30
        print('Driving at 30')
        time.sleep(5)
        back_wheels.speed = 70
        print('Driving at 70')
        time.sleep(5)
        back_wheels.speed = 20
        print('Driving at 20')
        time.sleep(5)
        back_wheels.speed = 50
        print('Driving at 50')
        time.sleep(5)
        back_wheels.speed = 15
        print('Driving at 15')
        time.sleep(5)

if __name__=='__main__':
    main()