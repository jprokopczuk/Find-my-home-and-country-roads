# Here simple tresholding based on opencv

import cv2 as cv
import os


def simple_tresholding():
    # CONFIG_DIR = os.path.dirname(__file__) + "/example_picture/"

    # img = cv.imread(CONFIG_DIR+'22678960_15.png', 0)

    # _, th1 = cv.threshold(img, 150, 255, cv.THRESH_BINARY)

    # # _, th2 = cv.threshold(img, 200, 255, cv.THRESH_BINARY_INV)

    # #_, th3 = cv.threshold(img, 100, 255, cv.THRESH_TRUNC)

    # # _, th4 = cv.threshold(img, 100, 255, cv.THRESH_TOZERO)

    # # _, th5 = cv.threshold(img, 100, 255, cv.THRESH_TOZERO_INV)

    # # th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    # # th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 5, 2)

    # cv.imshow("Image", img)

    # cv.imshow("th2", th1)
    # cv.imshow("th3",th3)

    # cv.imshow("th1", th1)
    # cv.imshow("th2", th2)
    # cv.imshow("th3", th3)
    # cv.imshow("th4", th4)
    # cv.imshow("th5", th5)

    window_detection_name = 'Simple thresholding'

    cv.namedWindow(window_detection_name)


    #while True:

    CONFIG_DIR = os.path.dirname(__file__) + "/example_picture/"

    frame = cv.imread(CONFIG_DIR+'22678915_15.png', 3)

    frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    frame_threshold = cv.inRange(frame_HSV, (101, 0, 0), (115, 84, 255))


    frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)


    frame_threshold = cv.inRange(frame_HSV, (101, 0, 0), (115, 84, 255))

    width = 720
    height = 720
    dim = (width, height)

    resized = cv.resize(frame_threshold, dim, interpolation=cv.INTER_AREA)

    cv.imshow(window_detection_name, resized)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    simple_tresholding()
    print("aa")
