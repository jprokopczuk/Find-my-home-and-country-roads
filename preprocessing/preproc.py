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
    l=[]
    size=128
    for fname in X_names:
        im=np.array(Image.open(fname).crop((46, 46, 1454, 1454)))
        tiles = [im[x:x + size, y:y + size] for x in range(0, im.shape[0], size) for y in range(0, im.shape[1], size)]
        for i in range(len(tiles)):
            l.append(tiles[i])
    X = np.array(l)
    l = []
    for fname in Y_buildings_names:
        im = np.array(Image.open(fname).crop((46, 46, 1454, 1454)))
        tiles = [im[x:x + size, y:y + size] for x in range(0, im.shape[0], size) for y in range(0, im.shape[1], size)]
        for i in range(len(tiles)):
            l.append(tiles[i])
    Y = np.array(l)
    l = []

    for fname in Y_roads_names:
        im = np.array(Image.open(fname).crop((46, 46, 1454, 1454)))
        tiles = [im[x:x + size, y:y + size] for x in range(0, im.shape[0], size) for y in range(0, im.shape[1], size)]
        for i in range(len(tiles)):
            l.append(tiles[i])
    Y_r = np.array(l)

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

    Y_mask = Y
    Y_cat = np_utils.to_categorical(Y, 3)

    return X, Y_cat, Y_mask


def generate_data_set():

    X_train, Y_train_cat, Y_train_mask = load_data("train")
    X_val, Y_val_cat, Y_val_mask = load_data("val")
    X_test, Y_test_cat, Y_test_mask = load_data("test")

    return X_train, Y_train_cat, Y_train_mask, X_val, Y_val_cat, X_test, Y_test_cat

