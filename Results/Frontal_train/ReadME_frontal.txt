Found 5200 files belonging to 7 classes.
Using 4420 files for training.
Using 780 files for validation.

Training dataset Classnames:  ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
Validation dataset Classnames:  ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

(TensorSpec(shape=(None, 48, 48, 1), dtype=tf.float32, name=None), TensorSpec(shape=(None, 7), dtype=tf.float32, name=None))

Search space summary
Default search space size: 9
conv_filter_1 (Int)
{'default': None, 'conditions': [], 'min_value': 12, 'max_value': 100, 'step': 12, 'sampling': 'linear'}
conv_kernel_1 (Choice)
{'default': 2, 'conditions': [], 'values': [2, 4], 'ordered': True}
maxpool_1 (Choice)
{'default': 2, 'conditions': [], 'values': [2, 4], 'ordered': True}
conv_filter_2 (Int)
{'default': None, 'conditions': [], 'min_value': 12, 'max_value': 100, 'step': 12, 'sampling': 'linear'}
conv_kernel_2 (Choice)
{'default': 2, 'conditions': [], 'values': [2, 4], 'ordered': True}
maxpool_2 (Choice)
{'default': 2, 'conditions': [], 'values': [2, 4], 'ordered': True}
dense_1 (Int)
{'default': None, 'conditions': [], 'min_value': 12, 'max_value': 88, 'step': 12, 'sampling': 'linear'}
dropout (Choice)
{'default': 0.1, 'conditions': [], 'values': [0.1, 0.2, 0.3, 0.4], 'ordered': True}
lr_rate (Choice)
{'default': 0.01, 'conditions': [], 'values': [0.01, 0.001], 'ordered': True}


Optimal Parameter:
Optimal Hyperparameters in Model
Layer 1: conv_filter:  24 conv_kernel:  2 MaxPool:  2
Layer 2: conv_filter:  72 conv_kernel:  4 MaxPool:  2
Dense Layer after Flatten:  72
DropOut:  0.4
Learning rate:  0.001
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 rescaling_1 (Rescaling)     (None, 48, 48, 1)         0

 conv2d_2 (Conv2D)           (None, 47, 47, 24)        120

 max_pooling2d_2 (MaxPooling  (None, 23, 23, 24)       0
 2D)

 conv2d_3 (Conv2D)           (None, 20, 20, 72)        27720

 max_pooling2d_3 (MaxPooling  (None, 10, 10, 72)       0
 2D)

 flatten_1 (Flatten)         (None, 7200)              0

 dense_2 (Dense)             (None, 72)                518472

 dropout_1 (Dropout)         (None, 72)                0

 dense_3 (Dense)             (None, 7)                 511

=================================================================
Total params: 546,823
Trainable params: 546,823
Non-trainable params: 0


