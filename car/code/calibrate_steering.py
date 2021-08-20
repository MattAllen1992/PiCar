import picar
import time

def main():
    # straight = 90
    # left = 45
    # right = 135
    # DO NOT EXCEED LEFT OR RIGHT ANGLES (i.e. <45 or >135
    front_wheels = picar.front_wheels.Front_Wheels()
    try:
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
        
    except KeyboardInterrupt:
        # handle user keyboard interrupt
        front_wheels.turn_straight()
    
    finally:
        # reset wheels to straight when done
        front_wheels.turn_straight()

if __name__=='__main__':
    main()