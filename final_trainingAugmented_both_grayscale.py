###
#   Training Model with the augmented grayscale images for both frontal and profile faces 
###

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Rescaling, Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras_tuner import RandomSearch # which parameter will be best 
from keras_tuner.engine.hyperparameters import HyperParameters
import matplotlib.pyplot as plt
import os
import random
import numpy as np

seedNum = 1234
epochs = 90 # number of times model will be trained
batchSize = 500

plotFileName = "final_Both_Augmented_%s_epochs_grayscale"%(epochs)
modelFileName = "Final_train_Augmented_model_%s_epochs_both"%(epochs)

## set the seed
def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

## Model creation
def createModel():
	model = Sequential()
	model.add(Rescaling((1./255), input_shape= (48,48,1))) # Standardize the data since they are in RGB channels
	# layer 1
	model.add(Conv2D(filters= 24, kernel_size= 2, activation= "relu"))    
	model.add(MaxPooling2D(pool_size= 2))
	# layer 2
	model.add(Conv2D(filters= 72, kernel_size= 4, activation= "relu"))    
	model.add(MaxPooling2D(pool_size= 2))

	model.add(Flatten())
	model.add(Dense(units= 72, activation="relu"))    # Dense is used to make this a fully connected model and is the hidden layer.
	model.add(Dropout(rate= 0.4))
	model.add(Dense(7,activation="softmax"))     # the output layer contains only 7 neurons which decide to which category image belongs to

	# compile Models
	model.compile(loss= 'categorical_crossentropy', 
               	optimizer= keras.optimizers.Adam(learning_rate= 0.001), 
                metrics=['accuracy'])

	return model

## Get visual representation of the trained model
def visualizeTrainingModel (trainModel, epochs):
	train_acc = trainModel.history['accuracy']
	validation_acc = trainModel.history['val_accuracy']

	train_loss = trainModel.history['loss']
	validation_loss = trainModel.history['val_loss']

	epochs_range = range(epochs)

	plt.figure(figsize=(8, 8))
	plt.subplot(1, 2, 1)
	plt.plot(epochs_range, train_acc, label='Training Accuracy')
	plt.plot(epochs_range, validation_acc, label='Validation Accuracy')
	plt.legend(loc='lower right')
	plt.title('Training and Validation Accuracy')

	plt.subplot(1, 2, 2)
	plt.plot(epochs_range, train_loss, label='Training Loss')
	plt.plot(epochs_range, validation_loss, label='Validation Loss')
	plt.legend(loc='upper right')
	plt.title('Training and Validation Loss')
	plt.savefig(plotFileName)	# save plot
	plt.show()

	return

set_seeds(seedNum)

# # Read in the data and split into train and validation sets
train_dt, validation_dt= keras.preprocessing.image_dataset_from_directory(directory= 'augmented_train_both',
                                                            labels= "inferred",
                                                            color_mode= "grayscale",
															image_size= (48,48),
															shuffle= True,
                                                            validation_split = 0.15,
                                                            subset = "both",
															batch_size= batchSize,
															seed= seedNum,
															label_mode='categorical')

## Classnames
print("Training dataset Classnames: ", train_dt.class_names)
print("Validation dataset Classnames: ", validation_dt.class_names)
print(train_dt.element_spec)

model = createModel()

# fit the model to the train and validation set
trainModel = model.fit(train_dt, validation_data= validation_dt, batch_size= batchSize, epochs= epochs)
# Save `trainModel` to file in the current working directory
model.save(modelFileName)

visualizeTrainingModel(trainModel, epochs) 	# get visualisation

