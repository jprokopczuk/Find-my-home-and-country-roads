#!/usr/bin/env python 3

from preprocessing import preproc
import matplotlib.pyplot as plt
import matplotlib.image as img
from neural_network import neural_netowrk
import numpy as np
import os


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    X_train, Y_train_cat, X_val, Y_val_cat, X_test, Y_test_cat = preproc.generate_data_set()

    print(X_val.shape)
    print(Y_val_cat.shape)
    print(X_test.shape)
    print(Y_test_cat.shape)
    model=neural_netowrk.generate_model(3,128,128,3)

    history = model.fit(X_train, Y_train_cat,
                        batch_size=16,
                        verbose=1,
                        epochs=5,
                        validation_data=(X_val, Y_val_cat),
                        # class_weight=class_weights,
                        shuffle=False)

    model.save('test.hdf5')

    loss,acc=model.evaluate(X_test, Y_test_cat)
    print(f"Accuracy is = {acc*100}%")

if __name__=="__main__":
    main()