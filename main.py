#!/usr/bin/env python 3

from preprocessing import preproc
import matplotlib.pyplot as plt
import matplotlib.image as img
from neural_network import neural_netowrk
import numpy as np

def main():
    X_val, Y_val_cat, X_test, Y_test_cat = preproc.generate_data_set()

    print(X_val.shape)
    print(Y_val_cat.shape)
    print(X_test.shape)
    print(Y_test_cat.shape)
    model=neural_netowrk.generate_model(3,1500,1500,3)

    history = model.fit(X_test, Y_test_cat,
                        batch_size=16,
                        verbose=1,
                        epochs=50,
                        validation_data=(X_val, Y_val_cat),
                        # class_weight=class_weights,
                        shuffle=False)

    model.save('test.hdf5')
    #plt.imshow(Y_train_cat[1, :, :, :])

if __name__=="__main__":
    main()