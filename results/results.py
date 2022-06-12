import matplotlib.pyplot as plt
import random
import numpy as np
from mlxtend.plotting import plot_confusion_matrix

def print_metric(history, metric, val_metric):
    loss = history.history[metric]
    val_loss = history.history[val_metric]
    epochs = range(1, len(loss) + 1)
    label='Training '+metric
    val_label = 'Validation ' + metric
    title='Training and validation '+metric

    #Display graph of the metric
    plt.plot(epochs, loss, 'y', label=label)
    plt.plot(epochs, val_loss, 'r', label=val_label)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend()
    plt.show()



def display_prediction(model,X_test, Y_test_cat):
    #Elements of the confusion matrix
    T_0 = 0
    T_01 = 0
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
    test_img_input = np.expand_dims(test_img, 0)
    prediction = (model.predict(test_img_input))
    predicted_img = np.argmax(prediction, axis=3)[0, :, :]
    #Compare pixels of the ground truth and prediction image
    for j in range(ground_truth.shape[0]):
        for k in range(ground_truth.shape[1]):
            for l in range(ground_truth.shape[2]):
                if ground_truth[j][k][l] == 1:
                    if l == 0:
                        if predicted_img[j][k] == 0:
                            T_0 += 1
                        elif predicted_img[j][k] == 1:
                            T_01 += 1
                        else:
                            T_02 += 1
                    elif l == 1:
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
    Matrix = [[T_0, T_01, T_02],
              [T_10, T_1, T_12],
              [T_20, T_21, T_2]]
    arr = np.array(Matrix)
    print(arr)
    #protection against dividing by 0
    if T_0 == 0:
        T_0 += 0.001
    if T_01 == 0:
        T_01 += 0.001
    if T_02 == 0:
        T_02 += 0.001
    if T_10 == 0:
        T_10 += 0.001
    if T_1 == 0:
        T_1 += 0.001
    if T_12 == 0:
        T_12 += 0.001
    if T_20 == 0:
        T_20 += 0.001
    if T_21 == 0:
        T_21 += 0.001
    if T_2 == 0:
        T_2 += 0.001
    #Calculate precision and recall of every class
    BackgroundPrecision = (T_0) / (T_0 + T_01 + T_02)
    BackgroundRecall = (T_0) / (T_0 + T_10 + T_20)
    BackgroundF1 = 2 * (BackgroundRecall * BackgroundPrecision) / (BackgroundRecall + BackgroundPrecision)
    BuildingPrecision = (T_1) / (T_1 + T_10 + T_12)
    BuildingRecall = (T_1) / (T_1 + T_01 + T_21)
    BuildingF1 = 2 * (BuildingRecall * BuildingPrecision) / (BuildingRecall + BuildingPrecision)
    RoadPrecision = (T_2) / (T_2 + T_20 + T_21)
    RoadRecall = (T_2) / (T_2 + T_02 + T_12)
    RoadF1 = 2 * (RoadRecall * RoadPrecision) / (RoadRecall + RoadPrecision)
    print(f"Precision (Background) = {BackgroundPrecision * 100}%")
    print(f"Precision (Building) = {BuildingPrecision * 100}%")
    print(f"Precision (Road) = {RoadPrecision * 100}%")
    print(f"Recall (Background) = {BackgroundRecall * 100}%")
    print(f"Recall (Building) = {BuildingRecall * 100}%")
    print(f"Recall (Road) = {RoadRecall * 100}%")
    print(f"F1 Score (Background) = {BackgroundF1 * 100}%")
    print(f"F1 Score (Building) = {BuildingF1 * 100}%")
    print(f"F1 Score (Road) = {RoadF1 * 100}%")
    class_names = ['Background', 'Building', 'Road']

    #Display essential graphs
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
    fig, ax = plot_confusion_matrix(conf_mat=arr, colorbar=True, class_names=class_names)
    plt.show()