# Here simple tresholding based on opencv

# import the necessary packages
from cv2 import imshow, threshold
from skimage.metrics import structural_similarity as compare_ssim

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


CONFIG_DIR = os.path.dirname(__file__) + "/example_picture/"

def count_precision(predicted_mask, real_mask) -> float:
    common_mask = np.bitwise_and(predicted_mask, real_mask)
    # pixels_predicted_mask = cv2.countNonZero(predicted_mask)
    pixels_real_mask = cv2.countNonZero(real_mask)
    pixels_common_mask = cv2.countNonZero(common_mask)

    return pixels_common_mask/(pixels_real_mask)

def count_recall(predicted_mask, real_mask) -> float:
    common_mask = np.bitwise_and(predicted_mask, real_mask)
    pixels_predicted_mask = cv2.countNonZero(predicted_mask)
    # pixels_real_mask = cv2.countNonZero(real_mask)
    pixels_common_mask = cv2.countNonZero(common_mask)

    return pixels_common_mask/(pixels_predicted_mask)

def find_ellipses(thresh):
        contours,_ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        if len(contours) != 0:
            for cont in contours:
                if len(cont) < 5:
                    break
                elps = cv2.fitEllipse(cont)
                return elps  #only returns one ellipse for now
        return None

def tresholding(original_photo: np.ndarray) -> np.ndarray:
    frame_HSV = cv2.cvtColor(original_photo, cv2.COLOR_BGR2HSV)

    # Treshold photo
    frame_threshold = cv2.inRange(frame_HSV, (101, 0, 0), (115, 84, 255))

    # Filters kernels
    kernel_close = np.ones((4,4),np.uint8)
    kernel_open = np.ones((1,1),np.uint8)

    # Filtering
    frame_threshold = cv2.morphologyEx(frame_threshold, cv2.MORPH_CLOSE, kernel_close)
    frame_threshold = cv2.morphologyEx(frame_threshold, cv2.MORPH_OPEN, kernel_open)

    return frame_threshold

def get_classes(original_photo: np.ndarray, frame_threshold: np.ndarray):
    cnts = cv2.findContours(frame_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
  
    # Put background colour
    thresholded_photo = np.zeros(original_photo.shape)
    thresholded_photo[:, :, 0] = 1
    thresholded_photo[:, :, 1:2] = 0

    mask_roads = np.zeros(original_photo.shape,  np.uint8)
    mask_buildings = np.zeros(original_photo.shape,  np.uint8)

    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        mask = np.zeros(original_photo.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [c], [255,255,255])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        pixels = cv2.countNonZero(mask)

        elipse = find_ellipses(mask)

        # First filter; objects smaller than 10px aren't analyze
        if pixels > 10:
            # If second axis is equal zero qualify it as
            if elipse[1][0] == 0:
                # Road
                cv2.fillPoly(thresholded_photo, [c], [0,0,255])
                cv2.fillPoly(mask_roads, [c], [255,255,255])
                continue

            if (elipse[1][1]/elipse[1][0])>2:
                # Road
                cv2.fillPoly(thresholded_photo, [c], [0,0,255])
                cv2.fillPoly(mask_roads, [c], [255,255,255])
            else:
                # Building
                cv2.fillPoly(thresholded_photo, [c], [0,255,0])
                cv2.fillPoly(mask_buildings, [c], [255,255,255])
    
    return mask_buildings, mask_roads, thresholded_photo

def make_tresholding():
    mask_buildings_original = cv2.imread(CONFIG_DIR + "22678915_15D.tif")
    mask_roads_original = cv2.imread(CONFIG_DIR + "22678915_15.tif")

    original_photo = cv2.imread(CONFIG_DIR + "22678915_15.png")

    frame_threshold =  tresholding(original_photo)

    plt.figure(figsize=(12, 8))
    plt.title('Tresholded frame')
    plt.imshow(frame_threshold[:], cmap='jet')
    plt.show()

    (mask_buildings, mask_roads, thresholded_photo) = get_classes(original_photo, frame_threshold)

    plt.figure(figsize=(12, 8))
    plt.subplot(131)
    plt.title('Mask buildings')
    plt.imshow(mask_buildings[:], cmap='jet')
    plt.subplot(132)
    plt.title('Mask roads')
    plt.imshow(mask_roads[:], cmap='jet')
    plt.subplot(133)
    plt.title('Mask coloured')
    plt.imshow(thresholded_photo[:], cmap='jet')
    plt.show()
    
    mask_buildings = cv2.cvtColor(mask_buildings, cv2.COLOR_RGB2GRAY)
    mask_buildings_original = cv2.cvtColor(mask_buildings_original, cv2.COLOR_RGB2GRAY)

    mask_roads = cv2.cvtColor(mask_roads, cv2.COLOR_RGB2GRAY)
    mask_roads_original = cv2.cvtColor(mask_roads_original, cv2.COLOR_RGB2GRAY)

    print(f"Precision for buildings: {count_precision(predicted_mask = mask_buildings, real_mask = mask_buildings_original)}")
    print(f"Recall for buildings: {count_recall(predicted_mask = mask_buildings, real_mask = mask_buildings_original)}")

    print(f"Precision for roads: {count_precision(predicted_mask = mask_roads, real_mask = mask_roads_original)}")
    print(f"Recall for roads: {count_recall(predicted_mask = mask_roads, real_mask = mask_roads_original)}")

    plt.figure(figsize=(12, 8))
    plt.subplot(121)
    plt.title('Original photo')
    plt.imshow(original_photo[:, :, :], cmap='jet')
    plt.subplot(122)
    plt.title('Prediction on test image')
    plt.imshow(thresholded_photo[:, :], cmap='jet')
    plt.show()