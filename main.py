#!/usr/bin/env python 3

from preprocessing import preproc
import matplotlib.pyplot as plt
import matplotlib.image as img
from neural_network import neural_netowrk
import numpy as np
import os
import random
import albumentations as A
import cv2
from mlxtend.plotting import plot_confusion_matrix

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

    #Augumentation
    transform = A.Compose(
        [
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
            A.HorizontalFlip(p=0.5),
        ]
    )

    for i in range(X_train.shape[0]):
        transformed_im_train = transform(image=X_train[i], mask=Y_train_cat[i])
        X_train[i] = transformed_im_train['image']
        Y_train_cat[i] = transformed_im_train['mask']


    for i in range(X_val.shape[0]):
        transformed_im_val = transform(image=X_val[i], mask=Y_val_cat[i])
        X_val[i] = transformed_im_val['image']
        Y_val_cat[i] = transformed_im_val['mask']

    print("Augumentation is done")
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

    show_random_image(X_train, Y_train_cat, "train", 4)
    show_random_image(X_val, Y_val_cat, "val", 4)
    show_random_image(X_test, Y_test_cat, "test", 4)

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
    class_weights[1] = 1.18
    class_weights[2] = 1.4
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
                        epochs=100,
                        validation_data=(X_val, Y_val_cat),
                        sample_weight=Y_class_weights,
                        shuffle=False)
    model.save('test.hdf5')

    # Jesli masz juz model to go wczytaj
    #model.load_weights('test.hdf5')
    loss, acc, jaccard, dice = model.evaluate(X_test, Y_test_cat)
    #loss, acc, jaccard, dice = model.evaluate(X_train, Y_train_cat)
    print(f"Accuracy is = {acc * 100}%")
    print(f"Loss is = {loss * 100}%")
    print(f"Jaccard index is = {jaccard * 100}%")
    print(f"Dice index is = {dice * 100}%")

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

    jaccard = history.history['jaccard_index']
    val_jaccard = history.history['val_jaccard_index']
    plt.plot(epochs, jaccard, 'y', label='Training Jaccard Index')
    plt.plot(epochs, val_jaccard, 'r', label='Validation Jaccard Index')
    plt.title('Training and validation Jaccard Index')
    plt.xlabel('Epochs')
    plt.ylabel('Jaccard Index')
    plt.legend()
    plt.show()

    dice = history.history['dice']
    val_dice = history.history['val_dice']
    plt.plot(epochs, dice, 'y', label='Training dice')
    plt.plot(epochs, val_dice, 'r', label='Validation dice')
    plt.title('Training and validation dice')
    plt.xlabel('Epochs')
    plt.ylabel('dice')
    plt.legend()
    plt.show()

    for i in range(20):
        T_0=0
        T_01=0
        T_02 = 0
        T_1 = 0
        T_10 = 0
        T_12 = 0
        T_2 = 0
        T_20 = 0
        T_21 = 0
        test_img_number = random.randint(0, len(X_test))
        test_img = X_test[test_img_number]
        ground_truth = Y_test_cat[test_img_number]
        test_img_norm = test_img[:, :, 0][:, :, None]
        test_img_input = np.expand_dims(test_img, 0)
        print(test_img_input.size)
        prediction = (model.predict(test_img_input))
        predicted_img = np.argmax(prediction, axis=3)[0, :, :]
        for j in range(ground_truth.shape[0]):
            for k in range(ground_truth.shape[1]):
                for l in range(ground_truth.shape[2]):
                    if ground_truth[j][k][l]==1:
                        if l==0:
                            if predicted_img[j][k]==0:
                                T_0 += 1
                            elif predicted_img[j][k]==1:
                                T_01 += 1
                            else:
                                T_02 += 1
                        elif l==1:
                            if predicted_img[j][k] == 1:
                                T_1 += 1
                            elif predicted_img[j][k] == 0:
                                T_10 += 1
                            else:
                                T_12 += 1
                        else:
                            if predicted_img[j][k] == 2:
                                T_2 += 1
                            elif predicted_img[j][k] == 0:
                                T_20 += 1
                            else:
                                T_21 += 1
        Matrix=[[T_0,T_01,T_02],
                [T_10,T_1,T_12],
                [T_20,T_21,T_2]]
        arr=np.array(Matrix)
        print(arr)
        BackgroundPrecision=(T_0)/(T_0+T_01+T_02)
        BackgroundRecall = (T_0) / (T_0 + T_10 + T_20)
        BackgroundF1=2*(BackgroundRecall*BackgroundPrecision)/(BackgroundRecall+BackgroundPrecision)
        BuildingPrecision = (T_1) / (T_1 + T_10 + T_12)
        BuildingRecall = (T_1) / (T_1 + T_01 + T_21)
        BuildingF1 = 2 * (BuildingRecall * BuildingPrecision) / (BuildingRecall + BuildingPrecision)
        RoadPrecision = (T_2) / (T_2 + T_20 + T_21)
        RoadRecall = (T_2) / (T_2 + T_02 + T_12)
        RoadF1 = 2 * (RoadRecall * RoadPrecision) / (RoadRecall + RoadPrecision)
        print(f"Precision (Background) = {BackgroundPrecision*100}%")
        print(f"Precision (Building) = {BuildingPrecision*100}%")
        print(f"Precision (Road) = {RoadPrecision*100}%")
        print(f"Recall (Background) = {BackgroundRecall*100}%")
        print(f"Recall (Building) = {BuildingRecall*100}%")
        print(f"Recall (Road) = {RoadRecall*100}%")
        print(f"F1 Score (Background) = {BackgroundF1*100}%")
        print(f"F1 Score (Building) = {BuildingF1*100}%")
        print(f"F1 Score (Road) = {RoadF1*100}%")
        class_names = ['Background', 'Building', 'Road']

        plt.figure(figsize=(12, 8))
        plt.subplot(131)
        plt.title('Testing Image')
        plt.imshow(test_img[:, :, :], cmap='gray')
        plt.subplot(132)
        plt.title('Testing Label')
        plt.imshow(ground_truth[:, :, :], cmap='jet')
        plt.subplot(133)
        plt.title('Prediction on test image')
        plt.imshow(predicted_img[:, :], cmap='jet')
        fig, ax = plot_confusion_matrix(conf_mat=arr,colorbar=True,class_names=class_names)
        plt.show()


if __name__ == "__main__":
    main()