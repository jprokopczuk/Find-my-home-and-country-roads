# Here neural network based on Keras (?) and preprocessing/augemntation
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation # Types of layers to be used in our model
from keras.layers import Conv2D, Conv3D, MaxPooling2D, UpSampling2D, GlobalAveragePooling2D, Flatten
from tensorflow.keras.layers import BatchNormalization
from keras import optimizers
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.metrics import Recall, Precision
import keras.backend as K

def jaccard_index(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

def dice(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 2*(intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f)+1.0)

def generate_model(n_classes,IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
 model = Sequential()
 for i in range(2):
    model.add(Conv2D(64, (3, 3), input_shape=(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
 model.add(MaxPooling2D(pool_size=(2,2)))
 for i in range(2):
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
 model.add(MaxPooling2D(pool_size=(2,2)))
 for i in range(2):
    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
 model.add(MaxPooling2D(pool_size=(2,2)))
 for i in range(2):
    model.add(Conv2D(512, (3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
 model.add(MaxPooling2D(pool_size=(2,2)))
 for i in range(2):
    model.add(Conv2D(512, (3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
 model.add(UpSampling2D(size=(2,2)))
 for i in range(2):
    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
 model.add(UpSampling2D(size=(2,2)))
 for i in range(2):
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
 model.add(UpSampling2D(size=(2,2)))
 for i in range(2):
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
 model.add(UpSampling2D(size=(2,2)))
 model.add(Conv2D(n_classes, (3, 3), padding="same"))
 model.add(Activation('softmax'))

 model.summary()
 adam = keras.optimizers.Adam(learning_rate=0.001)
 model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy', jaccard_index, dice], sample_weight_mode="temporal")


 return model