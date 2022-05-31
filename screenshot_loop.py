import mss
import cv2
import time
import os
import pyscreenshot as ImageGrab
start_time = time.time()
display_time = 0.5

def ensure_dir():
    directory = os.path.dirname('./datasets/crabs/')
    print(directory)
    if not os.path.exists('./datasets/crabs/'):
        os.makedirs('./datasets/crabs/')

ensure_dir()

monitor = {"top": 23, "left": 0, "width": 974, "height": 835}
title = "FPS benchmark"

sct = mss.mss()
img = 0
mob = 'crab'
while True:
    # -- include('examples/showgrabfullscreen.py') --#

    if __name__ == '__main__':
        # grab fullscreen
        im = ImageGrab.grab([0,0,974,835])
        # save image file
        im.save(r'./datasets/crabs/' + mob + '_' + str(img) + '.jpg')

        # show image in a window
    # -#
    img += 1

    time.sleep(display_time)
    if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
