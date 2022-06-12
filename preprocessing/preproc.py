import numpy as np
from PIL import Image
import glob
from keras.utils import np_utils
import random
import matplotlib.pyplot as plt
import albumentations as A
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

def load_data(type):
    #Load images from directory
    X_names = glob.glob("data/buildings/" + type + "/*.tiff")
    X_names.sort()
    Y_buildings_names = glob.glob("data/buildings/" + type + "_labels/*.tif")
    Y_buildings_names.sort()
    Y_roads_names = glob.glob("data/roads/" + type + "_labels/*.tif")
    Y_roads_names.sort()
    l=[]
    size=128
    size_x=46
    #Cropping the images
    for fname in X_names:
        im=np.array(Image.open(fname).crop((size_x, size_x, 1500-size_x, 1500-size_x)))
        tiles = [im[x:x + size, y:y + size] for x in range(0, im.shape[0], size) for y in range(0, im.shape[1], size)]
        for i in range(len(tiles)):
            l.append(tiles[i])
    X = np.array(l)
    l = []
    for fname in Y_buildings_names:
        im = np.array(Image.open(fname).crop((size_x, size_x, 1500-size_x, 1500-size_x)))
        tiles = [im[x:x + size, y:y + size] for x in range(0, im.shape[0], size) for y in range(0, im.shape[1], size)]
        for i in range(len(tiles)):
            l.append(tiles[i])
    Y = np.array(l)
    l = []

    for fname in Y_roads_names:
        im = np.array(Image.open(fname).crop((size_x, size_x, 1500-size_x, 1500-size_x)))
        tiles = [im[x:x + size, y:y + size] for x in range(0, im.shape[0], size) for y in range(0, im.shape[1], size)]
        for i in range(len(tiles)):
            l.append(tiles[i])
    Y_r = np.array(l)
    #Generating class masks
    X = X.astype('float32')
    Y = Y.astype('float32')
    Y_r = Y_r.astype('float32')
    X /= 255
    Y /= 255
    Y_r /= 127.5
    Y = Y[:, :, :, 0]
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            for k in range(Y.shape[2]):
                if Y[i][j][k] < Y_r[i][j][k]:
                    Y[i][j][k] = Y_r[i][j][k]

    Y_mask = Y
    Y_cat = np_utils.to_categorical(Y, 3)

    return X, Y_cat, Y_mask


def generate_data_set():

    X_train, Y_train_cat, Y_train_mask = load_data("train")
    X_val, Y_val_cat, Y_val_mask = load_data("val")
    X_test, Y_test_cat, Y_test_mask = load_data("test")

    return X_train, Y_train_cat, Y_train_mask, X_val, Y_val_cat, X_test, Y_test_cat

def split_train(X,Y_cat,Y_mask,number):
    X = X[0:number]
    Y_cat = Y_cat[0:number]
    Y_mask = Y_mask[0:number]
    return X,Y_cat,Y_mask

def split(X,Y_cat,number):
    X = X[0:number]
    Y_cat = Y_cat[0:number]
    return X,Y_cat

def display_label(X, Y_cat):
    test_img_number = random.randint(0, len(X))
    test_img = X[test_img_number]
    ground_truth = Y_cat[test_img_number]
    #Display essential images
    plt.figure(figsize=(12, 8))
    plt.subplot(141)
    plt.title('Testing Image')
    plt.imshow(test_img)
    plt.subplot(142)
    plt.title('Background Label')
    plt.imshow(ground_truth[:, :, 0])
    plt.subplot(143)
    plt.title('Buildings Label')
    plt.imshow(ground_truth[:, :, 1])
    plt.subplot(144)
    plt.title('Roads Label')
    plt.imshow(ground_truth[:, :, 2])
    plt.show()

def augmentation(X_train,Y_train_cat,X_val,Y_val_cat):
    #Set parameters of the augmentation
    transform = A.Compose(
        [
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
            A.HorizontalFlip(p=0.5),
        ]
    )
    #Make modifications to the images
    for i in range(X_train.shape[0]):
        transformed_im_train = transform(image=X_train[i], mask=Y_train_cat[i])
        X_train[i] = transformed_im_train['image']
        Y_train_cat[i] = transformed_im_train['mask']

    for i in range(X_val.shape[0]):
        transformed_im_val = transform(image=X_val[i], mask=Y_val_cat[i])
        X_val[i] = transformed_im_val['image']
        Y_val_cat[i] = transformed_im_val['mask']

    print("Augumentation is done")

    return X_train, Y_train_cat, X_val, Y_val_cat

def show_random_image(data_set, data_set_mask, data_set_name, n):

        for i in range(n):
            img_number = random.randint(0, len(data_set))
            test_img = data_set[img_number]
            ground_truth = data_set_mask[img_number]

            plt.figure(figsize=(12, 8))
            plt.subplot(141)
            plt.title('Random ' + data_set_name + ' image : ' + str(i))
            plt.imshow(test_img)
            plt.subplot(142)
            plt.title('Random ' + data_set_name + ' background mask : ' + str(i))
            plt.imshow(ground_truth[:, :, 0])
            plt.subplot(143)
            plt.title('Random ' + data_set_name + ' buildings mask : ' + str(i))
            plt.imshow(ground_truth[:, :, 1])
            plt.subplot(144)
            plt.title('Random ' + data_set_name + ' roads mask : ' + str(i))
            plt.imshow(ground_truth[:, :, 2])
            plt.show()

def generate_class_weights(Y_train_cat,Y_train_mask):
    labelencoder = LabelEncoder()
    n, h, w = Y_train_mask.shape
    #Reshape collective mask
    train_masks_reshaped = Y_train_mask.reshape(-1, 1)
    train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
    train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)
    #Generate original weights
    np.unique(train_masks_encoded_original_shape)
    class_weights = class_weight.compute_class_weight(class_weight="balanced",
                                                      classes=np.unique(train_masks_reshaped_encoded),
                                                      y=train_masks_reshaped_encoded)
    Y_class_weights = np.zeros((Y_train_cat.shape[0], Y_train_cat.shape[1], Y_train_cat.shape[2]), dtype=float)
    print("Class weights are...:", class_weights)
    #Correct the weight values
    class_weights[0] += 1.5 * class_weights[0]
    class_weights[1] -= 0.53 * class_weights[1]
    class_weights[2] -= 0.58 * class_weights[2]
    print("Normalized class weights are...:", class_weights)
    #Assign weights to the pixels of masks
    for i in range(Y_train_cat.shape[0]):
        for j in range(Y_train_cat.shape[1]):
            for k in range(Y_train_cat.shape[2]):
                for l in range(3):
                    if Y_train_cat[i][j][k][l] > 0.5:
                        Y_class_weights[i][j][k] = class_weights[l]
    return Y_class_weights