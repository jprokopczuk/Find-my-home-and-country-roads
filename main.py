#!/usr/bin/env python 3

from preprocessing import preproc
from neural_network import neural_netowrk
from results import results

def main():
    #Generate data sets
    X_train, Y_train_cat, Y_train_mask, X_val, Y_val_cat, X_test, Y_test_cat = preproc.generate_data_set()

    #Split data sets into smaller batches
    X_train, Y_train_cat, Y_train_mask = preproc.split_train(X_train, Y_train_cat, Y_train_mask, 2000)
    X_test, Y_test_cat = preproc.split(X_test, Y_test_cat, 400)
    X_val, Y_val_cat = preproc.split(X_val, Y_val_cat, 400)

    #Display random image with labels
    preproc.display_label(X_test,Y_test_cat)

    #Perform data augumentation
    X_train,Y_train_cat,X_val,Y_val_cat=preproc.augmentation(X_train,Y_train_cat,X_val,Y_val_cat)

    #Show some preprocessed images
    preproc.show_random_image(X_train, Y_train_cat, "train", 2)
    preproc.show_random_image(X_val, Y_val_cat, "val", 2)
    preproc.show_random_image(X_test, Y_test_cat, "test", 2)

    #Calculate class weights
    Y_class_weights=preproc.generate_class_weights(Y_train_cat,Y_train_mask)

    #Generate SegNet model
    model = neural_netowrk.generate_model(3, 128, 128, 3)

    #Train the model
    history = model.fit(X_train, Y_train_cat,
                        batch_size=16,
                        verbose=1,
                        epochs=100,
                        validation_data=(X_val, Y_val_cat),
                        sample_weight=Y_class_weights,
                        shuffle=False)
    model.save('test.hdf5')

    #If you already have trained model just load it
    #model.load_weights('test.hdf5')

    #Print essential metrics
    loss, acc, jaccard, dice = model.evaluate(X_test, Y_test_cat)

    print(f"Accuracy is = {acc * 100}%")
    print(f"Loss is = {loss * 100}%")
    print(f"Jaccard index is = {jaccard * 100}%")
    print(f"Dice index is = {dice * 100}%")

    #Display graphs of the metrics
    results.print_metric(history,'loss','val_loss')
    results.print_metric(history, 'accuracy', 'val_accuracy')
    results.print_metric(history, 'jaccard_index', 'val_jaccard_index')
    results.print_metric(history, 'dice', 'val_dice')

    #Show random predictions and their confusion matrices
    for i in range(3):
        results.display_prediction(model,X_test,Y_test_cat)


if __name__ == "__main__":
    main()