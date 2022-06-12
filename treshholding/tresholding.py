# Here simple tresholding based on opencv

# import the necessary packages
from cv2 import threshold
from skimage.metrics import structural_similarity as compare_ssim

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


CONFIG_DIR = os.path.dirname(__file__) + "/example_picture/"


def main():
    mask_1 = cv2.imread(CONFIG_DIR + "22828990_15D.tif")
    mask_2 = cv2.imread(CONFIG_DIR + "22828990_15.tif")

    mask_3 = mask_1 + mask_2

    


    vis2 = cv2.cvtColor(mask_3, cv2.COLOR_RGB2GRAY)

    photo_1 = cv2.imread(CONFIG_DIR + "22828990_15.tiff")

    frame_HSV = cv2.cvtColor(photo_1, cv2.COLOR_BGR2HSV)

    frame_threshold = cv2.inRange(frame_HSV, (101, 0, 0), (115, 84, 255))


    frame_HSV = cv2.cvtColor(photo_1, cv2.COLOR_BGR2HSV)

    frame_threshold = cv2.inRange(frame_HSV, (101, 0, 0), (115, 84, 255))

    kernel = np.ones((4,4),np.uint8)

    #resized = cv2.resize(diff, dim, interpolation=cv2.INTER_AREA)

    frame_threshold = cv2.morphologyEx(frame_threshold, cv2.MORPH_CLOSE, kernel)
    frame_threshold = cv2.morphologyEx(frame_threshold, cv2.MORPH_OPEN, kernel)


    cnts = cv2.findContours(frame_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  
    # Put background colour
    thresholded_photo = np.zeros(photo_1.shape)
    thresholded_photo[:, :, 0] = 1
    thresholded_photo[:, :, 1:2] = 0

    # mask_size = photo_1.shape
    # mask_size=[photo_1.shape[0],photo_1.shape[1],1]

    
    
    # print(mask_size)

    mask_roads = np.zeros(photo_1.shape)
    mask_buildings = np.zeros(photo_1.shape)

    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        mask = np.zeros(photo_1.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [c], [255,255,255])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        pixels = cv2.countNonZero(mask)

        contour, _ = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if pixels > 800:
            # Road
            cv2.fillPoly(thresholded_photo, [c], [0,0,255])
            cv2.fillPoly(mask_roads, [c], [255,255,255])
        else:
            # Building
            cv2.fillPoly(thresholded_photo, [c], [0,255,0])
            cv2.fillPoly(mask_buildings, [c], [255,255,255])

    #print(frame_threshold.shape)

    # mask_buildings = cv2.bitwise_not(mask_buildings)

    # print(mask_buildings.shape)
    # cv2.imshow("aaaa",mask_1)
    # cv2.waitKey(0) 
  
    # #closing all open windows 
    # cv2.destroyAllWindows()
    
    # Dlaczego nie dziala?
    print(mask_buildings.shape)
    mask_buildings = cv2.cvtColor(mask_buildings, cv2.COLOR_RGB2GRAY)
    print(mask_buildings.shape)
    print(mask_1.shape)
    mask_1 = cv2.cvtColor(mask_1, cv2.COLOR_RGB2GRAY)
    print(mask_1.shape)

    #(score, diff) = compare_ssim(mask_1, mask_buildings, full=True)
    # diff = (diff * 255).astype("uint8")
    #print(f"Building accuracy: {score}")


    plt.figure(figsize=(12, 8))
    plt.subplot(121)
    plt.title('Original photo')
    plt.imshow(photo_1[:, :, :], cmap='jet')
    plt.subplot(122)
    plt.title('Prediction on test image')
    plt.imshow(thresholded_photo[:, :], cmap='jet')
    plt.show()

if __name__ == "__main__":
    main()