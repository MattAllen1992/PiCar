'''
This script displays the live RGB and B&W video feed from the USB
camera on the PiCar using OpenCV2 to read and display the stream of images
'''

import cv2

def main():
    # create camera instance, selecting first camera with index -1
    camera = cv2.VideoCapture(-1)

    # set width and height of image
    camera.set(3, 640)
    camera.set(4, 480)

    # infinite loop to continuously process stream of camera images
    while(camera.isOpened()):
        # get and show raw image
        _, image = camera.read()
        cv2.imshow('Raw', image)

        # convert to and show black and white image
        bwImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow('B&W', bwImage)

        # waits until user presses 'q' on the keyboard to quit
        # ord('q') is the unicode value of 'q' on the keyboard
        # cv2.waitKey(1) returns a 32-bit integer corresponding to the pressed key
        # 0xFF is a bit mask which allows us to extract only the bits corresponding to the key
        # https://stackoverflow.com/questions/10493411/what-is-bit-masking
        # https://stackoverflow.com/questions/53357877/usage-of-ordq-and-0xff
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # close session(s) tidily
    cv2.destroyAllWindows()

# call main method
if __name__ == '__main__':
    main()
