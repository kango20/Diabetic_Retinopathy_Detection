# import libraries
import warnings
warnings.filterwarnings('ignore')
import random as python_random
import numpy as np
import pandas as pd
import os
import shutil
import random

# data handling libraries
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ML libraries 
import tensorflow as tf
import tensorflow.keras as kb
from tensorflow.keras import regularizers
import matplotlib.pyplot
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.callbacks import EarlyStopping
import evaluation as Eval

# set seeds to get standard results
import keras
keras.utils.set_random_seed(44)
tf.config.experimental.enable_op_determinism()

class Model:
    def __init__(self, train, valid, test) -> None:
        self.train_gen, self.valid_gen, self.test_gen = train, valid, test
        self.model, self.history = self.create_model()
        
        

# create pre trained model 
    def create_model(self):
        base_model=tf.keras.applications.EfficientNetB3(
            include_top=False,
            weights="imagenet",
            input_shape=(224, 224,3),
            pooling='max',
            classifier_activation="softmax",
        )

        # fine tuning/freezing layers
        fine_tune_at = 100
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

        # adding layers to EfficientNetB3 model
        model=kb.Sequential([
            base_model,
            kb.layers.Flatten(),
            kb.layers.Dense(512, activation='relu'),
            kb.layers.Dense(256, activation='relu'),
            kb.layers.Dense(128, activation='relu'),
            kb.layers.Dense(64, activation='relu'),
            kb.layers.Dense(32, activation='relu'),
            kb.layers.Dense(5, activation='softmax')
        ])

        model.compile(optimizer=kb.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy', Eval.precision, Eval.recall, Eval.f1score])

        model.summary()
        

        # training the model
        history = model.fit(self.train_gen,
                            epochs = 50,
                            validation_data = self.valid_gen
                            )
        self.model = model
        self.history = history

        return model, history

