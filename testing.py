#############
#   Testing the model on the test set 
#############
import tensorflow as tf
from tensorflow import keras
from keras.utils import plot_model
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# get the model
def getModel (frontalOnly, augmented):
    if (augmented):
        return 'Final_train_Augmented_model_100_epochs_frontal' if frontalOnly else 'Final_train_Augmented_model_90_epochs_both'
    else:
        return 'Final_train_model_30_epochs_frontal' if frontalOnly else 'Final_train_model_25_epochs_both'


def main():
    # which model to test
    modelType = int(input("Choose the model to use. (Enter only the number) \n 1. Frontal Non-Augmented Model\n 2. Both Non-Augmented Model\n 3. Frontal Augmented Model\n 4. Both Augmented Model\n"))
    
    getFrontalOnly = True if modelType == 1 or modelType == 3 else False
    getAugmented = True if modelType > 2 else False

    print("The chosen option. FrontalOnly: ", getFrontalOnly, " getAugmented: ", getAugmented)

    modelName = getModel(getFrontalOnly, getAugmented)
    testSetName = 'test_frontal' if getFrontalOnly else 'test_both'

    print("\nModel to use: ", modelName, "\nTestset to use: ", testSetName)

    # get the test dataset and make sure it is in grayscale
    test_dt = keras.preprocessing.image_dataset_from_directory(directory= testSetName,
															image_size= (48,48),
                                                            color_mode= "grayscale",
															label_mode='categorical',
                                                            shuffle=False)
    
    print("Shape of data: ", test_dt)

    ## Load the model
    model = keras.models.load_model(modelName)
    # model.summary() # Check model architecture

    test_y = test_dt.class_names
    # print("Classnames of test set", test_y)

    loss, acc = model.evaluate(test_dt, verbose=2, batch_size= 500)
    print('\nLoss: ', loss)
    print('Accuracy on test set: {:.2f}%'.format(100 * acc))

    ## get Predictions
    y_prob = model.predict(test_dt, verbose = 2, batch_size = 500)
    print(y_prob)
    y_classes = y_prob.argmax(axis = -1)
    print(max(y_classes))

    y_pred = np.array(test_y)[y_classes.astype(int)]
    print(y_pred)

    test_label = np.concatenate([y for x, y in test_dt], axis = 0).argmax(axis = -1)
    test_label = np.array(test_y)[test_label.astype(int)]
    print(test_label)

    confusionMatrix = confusion_matrix(test_label, y_pred, labels = test_y, normalize = 'true')
    print(confusionMatrix)

    # Visualize Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix = confusionMatrix, display_labels = test_y)
    disp.plot()
    plt.show()


if __name__ == "__main__":
    main()
