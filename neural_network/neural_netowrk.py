# Here neural network based on Keras (?) and preprocessing/augemntation
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation # Types of layers to be used in our model
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, GlobalAveragePooling2D, Flatten
from tensorflow.keras.layers import BatchNormalization
from keras import optimizers

def generate_model():
 model = Sequential()   
 for i in range(2):
    model.add(Conv2D(64, (3, 3), input_shape=(1500,1500,3)))
    model.add(BatchNormalization()) 
    model.add(Activation('relu'))
 model.add(MaxPooling2D(pool_size=(2,2)))
 for i in range(2):
    model.add(Conv2D(128, (3, 3)))
    model.add(BatchNormalization()) 
    model.add(Activation('relu'))
 model.add(MaxPooling2D(pool_size=(2,2)))
 for i in range(3):
    model.add(Conv2D(256, (3, 3)))
    model.add(BatchNormalization()) 
    model.add(Activation('relu'))
 model.add(MaxPooling2D(pool_size=(2,2)))
 for i in range(3):
    model.add(Conv2D(512, (3, 3)))
    model.add(BatchNormalization()) 
    model.add(Activation('relu'))
 model.add(MaxPooling2D(pool_size=(2,2)))
 for i in range(3):
    model.add(Conv2D(512, (3, 3)))
    model.add(BatchNormalization()) 
    model.add(Activation('relu'))
 model.add(UpSampling2D(pool_size=(2,2)))
 for i in range(3):
    model.add(Conv2D(256, (3, 3)))
    model.add(BatchNormalization()) 
    model.add(Activation('relu'))
 model.add(UpSampling2D(pool_size=(2,2)))
 for i in range(2):
    model.add(Conv2D(128, (3, 3)))
    model.add(BatchNormalization()) 
    model.add(Activation('relu'))
 model.add(UpSampling2D(pool_size=(2,2)))
 for i in range(2):
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization()) 
    model.add(Activation('relu'))
 model.add(UpSampling2D(pool_size=(2,2)))
 for i in range(2):
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization()) 
    model.add(Activation('relu'))
 model.add(Conv2D(3, (3, 3)))
 model.add(Activation('softmax')) 

 model.summary()
 adam = optimizers.Adam(lr=0.001)
 model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


 return model
