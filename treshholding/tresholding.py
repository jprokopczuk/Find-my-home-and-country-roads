# # Here simple tresholding based on opencv

# import cv2 as cv
# import os


# def simple_tresholding():
#     # CONFIG_DIR = os.path.dirname(__file__) + "/example_picture/"

#     # img = cv.imread(CONFIG_DIR+'22678960_15.png', 0)

#     # _, th1 = cv.threshold(img, 150, 255, cv.THRESH_BINARY)

#     # # _, th2 = cv.threshold(img, 200, 255, cv.THRESH_BINARY_INV)

#     # #_, th3 = cv.threshold(img, 100, 255, cv.THRESH_TRUNC)

#     # # _, th4 = cv.threshold(img, 100, 255, cv.THRESH_TOZERO)

#     # # _, th5 = cv.threshold(img, 100, 255, cv.THRESH_TOZERO_INV)

#     # # th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
#     # # th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 5, 2)

#     # cv.imshow("Image", img)

#     # cv.imshow("th2", th1)
#     # cv.imshow("th3",th3)

#     # cv.imshow("th1", th1)
#     # cv.imshow("th2", th2)
#     # cv.imshow("th3", th3)
#     # cv.imshow("th4", th4)
#     # cv.imshow("th5", th5)

#     window_detection_name = 'Simple thresholding'

#     cv.namedWindow(window_detection_name)


#     #while True:

#     CONFIG_DIR = os.path.dirname(__file__) + "/example_picture/"

#     frame = cv.imread(CONFIG_DIR+'22678915_15.png', 3)

#     frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
#     frame_threshold = cv.inRange(frame_HSV, (101, 0, 0), (115, 84, 255))


#     frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)


#     frame_threshold = cv.inRange(frame_HSV, (101, 0, 0), (115, 84, 255))

#     width = 720
#     height = 720
#     dim = (width, height)

#     resized = cv.resize(frame_threshold, dim, interpolation=cv.INTER_AREA)

#     cv.imshow(window_detection_name, resized)

#     cv.waitKey(0)
#     cv.destroyAllWindows()


# if __name__ == "__main__":
#     simple_tresholding()
#     print("aa")


# import the necessary packages
from cv2 import threshold
from skimage.metrics import structural_similarity as compare_ssim
import argparse
import imutils
import cv2
import os
import numpy as np

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-f", "--first", required=True,
# 	help="first input image")
# ap.add_argument("-s", "--second", required=True,
# 	help="second")
# args = vars(ap.parse_args())

CONFIG_DIR = os.path.dirname(__file__) + "/example_picture/"

mask_1 = cv2.imread(CONFIG_DIR + "22828990_15D.tif")
mask_2 = cv2.imread(CONFIG_DIR + "22828990_15.tif")

mask_3 = mask_1 + mask_2


vis2 = cv2.cvtColor(mask_3, cv2.COLOR_RGB2GRAY)

#cv2.CvtColor(vis0, vis2, cv2.CV_GRAY2BGR)

# # load the two input images
# imageA = cv2.imread(args["first"])
# imageB = cv2.imread(args["second"])
# convert the images to grayscale
#grayA = cv2.cvtColor(mask_1, cv2.COLOR_BGR2GRAY)
#grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

photo_1 = cv2.imread(CONFIG_DIR + "22828990_15.tiff")

frame_HSV = cv2.cvtColor(photo_1, cv2.COLOR_BGR2HSV)

frame_threshold = cv2.inRange(frame_HSV, (101, 0, 0), (115, 84, 255))


frame_HSV = cv2.cvtColor(photo_1, cv2.COLOR_BGR2HSV)

frame_threshold = cv2.inRange(frame_HSV, (101, 0, 0), (115, 84, 255))


#print(frame_threshold.shape)

(score, diff) = compare_ssim(vis2, frame_threshold, full=True)
diff = (diff * 255).astype("uint8")
print(f"SSIM: {score}")
print(f"SSIM: {diff}")


width = 720
height = 720
dim = (width, height)

kernel = np.ones((4,4),np.uint8)

resized = cv2.resize(diff, dim, interpolation=cv2.INTER_AREA)

#frame_threshold = cv2.dilate(frame_threshold ,kernel,iterations = 1)
frame_threshold = cv2.morphologyEx(frame_threshold, cv2.MORPH_CLOSE, kernel)
frame_threshold = cv2.morphologyEx(frame_threshold, cv2.MORPH_OPEN, kernel)


cnts = cv2.findContours(frame_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
total = 0

print(photo_1.shape)

# Put background colour
thresholded_photo = np.zeros(photo_1.shape)
thresholded_photo[:, :, 0:1] = 0
thresholded_photo[:, :, 2] = 1

for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    mask = np.zeros(photo_1.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [c], [255,255,255])
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    pixels = cv2.countNonZero(mask)
    total += pixels

    if pixels > 900:
        # Road
        cv2.fillPoly(thresholded_photo, [c], [255,0,0])
        #cv2.putText(frame_threshold, '{}'.format(pixels), (x,y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    else:
        # Building
        cv2.fillPoly(thresholded_photo, [c], [0,255,0])



cv2.imshow("Tresholding", thresholded_photo)
cv2.waitKey(0)
cv2.destroyAllWindows()