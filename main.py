#!/usr/bin/env python 3

from preprocessing import preproc
import matplotlib.pyplot as plt
import matplotlib.image as img
from neural_network import neural_netowrk
import numpy as np
import os
import random


def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    X_train, Y_train_cat, Y_train_mask, X_val, Y_val_cat, X_test, Y_test_cat = preproc.generate_data_set()
    #X_train, Y_train_cat, Y_train_mask, X_test, Y_test_cat = preproc.generate_data_set()

    X_train = X_train[0:2000]
    Y_train_cat = Y_train_cat[0:2000]
    Y_train_mask = Y_train_mask[0:2000]
    X_test = X_test[0:400]
    Y_test_cat = Y_test_cat[0:400]
    X_val = X_val[0:400]
    Y_val_cat = Y_val_cat[0:400]
    print(X_train.shape)
    print(Y_train_cat.shape)
    print(X_val.shape)
    print(Y_val_cat.shape)
    print(X_test.shape)
    print(Y_test_cat.shape)
    # model=neural_netowrk.generate_model(3,128,128,3)

    test_img_number = random.randint(0, len(X_test))
    test_img = X_test[test_img_number]
    ground_truth = Y_test_cat[test_img_number]

    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(ground_truth[:, :, 0])
    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(ground_truth[:, :, 1])
    plt.subplot(233)
    plt.title('Testing Label')
    plt.imshow(ground_truth[:, :, 2])
    plt.show()

    # Normalizacja wag
    from sklearn.preprocessing import LabelEncoder
    labelencoder = LabelEncoder()
    n, h, w = Y_train_mask.shape
    train_masks_reshaped = Y_train_mask.reshape(-1, 1)
    train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
    train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)

    np.unique(train_masks_encoded_original_shape)
    print(Y_train_mask.shape)
    print(train_masks_reshaped.shape)
    print(train_masks_reshaped_encoded.shape)
    from sklearn.utils import class_weight
    class_weights = class_weight.compute_class_weight(class_weight="balanced",
                                                      classes=np.unique(train_masks_reshaped_encoded),
                                                      y=train_masks_reshaped_encoded)
    Y_class_weights = np.zeros((Y_train_cat.shape[0], Y_train_cat.shape[1], Y_train_cat.shape[2]), dtype=float)
    print("Class weights are...:", class_weights)
    class_weights[0] = 1
    class_weights[1] = 1.4
    class_weights[2] = 1.6
    for i in range(Y_train_cat.shape[0]):
        for j in range(Y_train_cat.shape[1]):
            for k in range(Y_train_cat.shape[2]):
                for l in range(3):
                    if Y_train_cat[i][j][k][l] > 0.5:
                        Y_class_weights[i][j][k] = class_weights[l]

    sample_weights = class_weight.compute_sample_weight('balanced', y=train_masks_reshaped_encoded)
    print("Class weights are...:", class_weights)
    print("Sample weights are...:", sample_weights)
    class_weights_dict = dict(zip(np.unique(train_masks_reshaped_encoded), class_weights))

    print("Class weights are...:", class_weights_dict)
    print("Sample weights are...:", sample_weights)
    model = neural_netowrk.generate_model(3, 128, 128, 3)
    '''history = model.fit(X_train, Y_train_cat,
                        batch_size=16,
                        verbose=1,
                        epochs=50,
                        validation_data=(X_val, Y_val_cat),
                        class_weight=class_weights_dict,
                        shuffle=False)'''
    history = model.fit(X_train, Y_train_cat,
                        batch_size=16,
                        verbose=1,
                        epochs=70,
                        validation_data=(X_val, Y_val_cat),
                        sample_weight=Y_class_weights,
                        shuffle=False)
    model.save('test.hdf5')

    # Jesli masz juz model to go wczytaj
    # model.load_weights('test.hdf5')

    loss, acc = model.evaluate(X_test, Y_test_cat)
    print(f"Accuracy is = {acc * 100}%")

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    plt.plot(epochs, acc, 'y', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training and validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    for i in range(5):
        test_img_number = random.randint(0, len(X_test))
        test_img = X_test[test_img_number]
        ground_truth = Y_test_cat[test_img_number]
        test_img_norm = test_img[:, :, 0][:, :, None]
        test_img_input = np.expand_dims(test_img, 0)
        print(test_img_input.size)
        prediction = (model.predict(test_img_input))
        predicted_img = np.argmax(prediction, axis=3)[0, :, :]

        plt.figure(figsize=(12, 8))
        plt.subplot(231)
        plt.title('Testing Image')
        plt.imshow(test_img[:, :, :], cmap='gray')
        plt.subplot(232)
        plt.title('Testing Label')
        plt.imshow(ground_truth[:, :, :], cmap='jet')
        plt.subplot(233)
        plt.title('Prediction on test image')
        plt.imshow(predicted_img[:, :], cmap='jet')
        plt.show()


if __name__ == "__main__":
    main()