#!usr/bin/env python

'''

This is the model.py pattern recomended by TensorFlow

* `serving_input_fn()`
*  `train_and_evaluate()`

Resource:
https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/image_classification/solutions/tpu_models/trainer/model.py

'''


import os
import shutil

import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras import layers
from tensorflow.keras import models

from . import util

# PARAM

NCLASSES = len(util.CLASSNAMES)
LEARNING_RATE = 0.0001
DROPOUT = 0.2


######
# Build Model
#####

def build_model(output_dir: str, hub_handle):

    """
    Compiles keras model for image classification.

    Args:
     ``output_dir``: directory where
            model artifacts will b e saved.
     ``hub_handle``: callable object, saved model.
    """

    model = models.Sequential([
        hub.KerasLayer(hub_handle, trainable=False)
        layers.Dropout(rate=DROPOUT),
        layers.Dense(
            NCLASSES,
            activation='softmax',
            kernel_regularizer=tf.regularizers.l2(LEARNING_RATE))]
    model.build((None,)+(util.IMG_HEIGHT, util.IMG_WIDTH, util.IMG_CHANNELs))
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return model

# TODO add train and evaluate


def train_and_evaluate():

    #TODO
    pass
