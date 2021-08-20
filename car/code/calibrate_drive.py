import picar
import time

def main():
    back_wheels = picar.back_wheels.Back_Wheels()
    delay = 0.01 # delay is required to process requests, otherwise results in issues processing each request
    try:
        # accelerate from 0 to 100%
        for i in range(100):
            back_wheels.speed = i
            print('Driving at ', i)
            time.sleep(delay)

        # drive at various speeds
        back_wheels.speed = 30
        print('Driving at 30')
        time.sleep(2)
        back_wheels.speed = 70
        print('Driving at 70')
        time.sleep(2)
        back_wheels.speed = 20
        print('Driving at 20')
        time.sleep(2)
        back_wheels.speed = 50
        print('Driving at 50')
        time.sleep(2)
        back_wheels.speed = 15
        print('Driving at 15')
        time.sleep(2)
        
    except KeyboardInterrupt:
        # stop if user interrupts
        back_wheels.stop()
        
    finally:
        # stop if script completes
        back_wheels.stop()

if __name__=='__main__':
    main()