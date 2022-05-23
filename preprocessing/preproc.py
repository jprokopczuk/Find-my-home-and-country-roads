import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import glob
from keras.utils import np_utils

def load_data(type):
    X_names = glob.glob("data/buildings/" + type + "/*.tiff")
    X_names.sort()
    Y_buildings_names = glob.glob("data/buildings/" + type + "_labels/*.tif")
    Y_buildings_names.sort()
    Y_roads_names = glob.glob("data/roads/" + type + "_labels/*.tif")
    Y_roads_names.sort()
    X = np.array([np.array(Image.open(fname).resize((256,256))) for fname in X_names])
    Y = np.array([np.array(Image.open(fname).resize((256,256))) for fname in Y_buildings_names])
    Y_r = np.array([np.array(Image.open(fname).resize((256,256))) for fname in Y_roads_names])
    X = X.astype('float32')
    Y = Y.astype('float32')
    Y_r = Y_r.astype('float32')
    X /= 255
    Y /= 255
    Y_r /= 127.5
    Y = Y[:, :, :, 0]
    print(Y.shape[0])
    print(Y.shape[1])
    print(Y.shape[2])
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            for k in range(Y.shape[2]):
                if Y[i][j][k] < Y_r[i][j][k]:
                    Y[i][j][k] = Y_r[i][j][k]

    Y_cat = np_utils.to_categorical(Y, 3)

    return X, Y_cat


def generate_data_set():

    X_train, Y_train_cat = load_data("train")
    X_val, Y_val_cat = load_data("val")
    X_test, Y_test_cat = load_data("test")

    return X_train, Y_train_cat, X_val, Y_val_cat, X_test, Y_test_cat

