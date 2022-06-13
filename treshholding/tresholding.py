# Here simple tresholding based on opencv

# import the necessary packages
from cv2 import threshold
from skimage.metrics import structural_similarity as compare_ssim

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


CONFIG_DIR = os.path.dirname(__file__) + "/example_picture/"

def count_precision(predicted_mask, real_mask) -> float:
    common_mask = np.bitwise_and(predicted_mask, real_mask)
    pixels_predicted_mask = cv2.countNonZero(predicted_mask)
    pixels_real_mask = cv2.countNonZero(real_mask)
    pixels_common_mask = cv2.countNonZero(common_mask)

    return pixels_common_mask/(pixels_real_mask)

def count_recall(predicted_mask, real_mask) -> float:
    common_mask = np.bitwise_and(predicted_mask, real_mask)
    pixels_predicted_mask = cv2.countNonZero(predicted_mask)
    # pixels_real_mask = cv2.countNonZero(real_mask)
    pixels_common_mask = cv2.countNonZero(common_mask)

    return pixels_common_mask/(pixels_predicted_mask)

def main():
    mask_1 = cv2.imread(CONFIG_DIR + "22828990_15D.tif")
    mask_2 = cv2.imread(CONFIG_DIR + "22828990_15.tif")

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

    mask_roads = np.zeros(photo_1.shape,  np.uint8)
    mask_buildings = np.zeros(photo_1.shape,  np.uint8)

    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        mask = np.zeros(photo_1.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [c], [255,255,255])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        pixels = cv2.countNonZero(mask)

        if pixels > 800:
            # Road
            cv2.fillPoly(thresholded_photo, [c], [0,0,255])
            cv2.fillPoly(mask_roads, [c], [255,255,255])
        else:
            # Building
            cv2.fillPoly(thresholded_photo, [c], [0,255,0])
            cv2.fillPoly(mask_buildings, [c], [255,255,255])

    mask_buildings = cv2.cvtColor(mask_buildings, cv2.COLOR_RGB2GRAY)
    mask_1 = cv2.cvtColor(mask_1, cv2.COLOR_RGB2GRAY)

    mask_roads = cv2.cvtColor(mask_roads, cv2.COLOR_RGB2GRAY)
    mask_2 = cv2.cvtColor(mask_2, cv2.COLOR_RGB2GRAY)

    print(f"Precision for buildings: {count_precision(predicted_mask = mask_buildings, real_mask = mask_1)}")
    print(f"Recall for buildings: {count_recall(predicted_mask = mask_buildings, real_mask = mask_1)}")

    print(f"Precision for roads: {count_precision(predicted_mask = mask_roads, real_mask = mask_2)}")
    print(f"Recall for roads: {count_recall(predicted_mask = mask_roads, real_mask = mask_2)}")

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