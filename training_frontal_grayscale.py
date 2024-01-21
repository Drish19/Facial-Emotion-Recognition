
#########
#   https://www.tensorflow.org/tutorials/images/classification
#
#	Train the model on the cleaned data for frontal images only
# 	All done in grayscale 
#########
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
epochs = 200 # number of times model will be trained 
batchSize = 500

tuningFileName = "tuned_model_frontal_grayscale"
plotFileName = "Frontal_%s_epochs_grayscale"%(epochs)
modelFileName = "train_model_%s_epochs_frontal"%(epochs)

## set the seed
def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

## Model creation
def createModel(hyperparams):
	model = Sequential()
	model.add(Rescaling((1./255), input_shape= (48,48,1))) # Standardize the data since they are in RGB channels
	# layer 1
	model.add(Conv2D(filters= hyperparams.Int('conv_filter_1', min_value = 12, max_value = 100, step = 12),
					kernel_size= hyperparams.Choice('conv_kernel_1', values = [2,4]),
                    activation= "relu"))    
	model.add(MaxPooling2D(pool_size= hyperparams.Choice('maxpool_1', values = [2,4])))
	# layer 2
	model.add(Conv2D(filters= hyperparams.Int('conv_filter_2', min_value = 12, max_value = 100, step = 12),
					kernel_size= hyperparams.Choice('conv_kernel_2', values = [2,4]),
                    activation= "relu"))    
	model.add(MaxPooling2D(pool_size= hyperparams.Choice('maxpool_2', values = [2,4])))

	model.add(Flatten())
	model.add(Dense(units= hyperparams.Int('dense_1', min_value = 12, max_value = 88, step = 12),
                	activation="relu"))    # Dense is used to make this a fully connected model and is the hidden layer.
	model.add(Dropout(rate= hyperparams.Choice('dropout', values = [0.1,0.2,0.3,0.4])))
	model.add(Dense(7,activation="softmax"))     # the output layer contains only 7 neurons which decide to which category image belongs to

	# compile Model
	model.compile(loss= 'categorical_crossentropy', 
               	optimizer= keras.optimizers.Adam(hyperparams.Choice('lr_rate', values=[1e-2,1e-3])), 
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
train_dt, validation_dt= keras.preprocessing.image_dataset_from_directory(directory= 'train_frontal',
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

## model tuning
# provide you with more insights about how many conv layer/ maxPool layer you should have
tuner = RandomSearch(hypermodel= createModel, objective= 'val_accuracy', max_trials= 10, directory= tuningFileName, overwrite= True)

tuner.search_space_summary()

# now search for best parameter for this particular build model
tuner.search(train_dt, epochs = 20, validation_data = validation_dt, batch_size = batchSize)

# Get the optimal hyperparameters
opt_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print("Optimal Hyperparameters in Model")
print("Layer 1: conv_filter: ", opt_hps.get('conv_filter_1') , "conv_kernel: ", opt_hps.get('conv_kernel_1')  , "MaxPool: ", opt_hps.get('maxpool_1'))
print("Layer 2: conv_filter: ", opt_hps.get('conv_filter_2') , "conv_kernel: ", opt_hps.get('conv_kernel_2')  , "MaxPool: ", opt_hps.get('maxpool_2'))
print("Dense Layer after Flatten: ", opt_hps.get('dense_1'))
print("DropOut: ", opt_hps.get('dropout'))
print("Learning rate: ", opt_hps.get('lr_rate'))

# Build the model with the optimal hyperparameters and train it on the data
model = tuner.hypermodel.build(opt_hps)
model.summary()

# # After tuning get best model
tuned_Model = tuner.get_best_models(num_models=1)[0]
tuned_Model.summary()

# fit the model to the train and validation set
trainModel = model.fit(train_dt, validation_data= validation_dt, batch_size= batchSize, epochs= epochs)
# Save `trainModel` to file in the current working directory
model.save(modelFileName)

visualizeTrainingModel(trainModel, epochs) 	# get visualisation
