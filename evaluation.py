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

# set seeds to get standard results
import keras
keras.utils.set_random_seed(44)
tf.config.experimental.enable_op_determinism()

class Eval:
    def __init__(self, Data_Gen, Model) -> None:
        self.train = Data_Gen.train
        self.valid = Data_Gen.valid
        self.test = Data_Gen.test
        self.train_gen = Data_Gen.train_gen
        self.valid_gen = Data_Gen.valid_gen
        self.test_gen = Data_Gen.test_gen
        self.model = Model.model
        self.history = Model.history

    def data_size(self):
        # print the dataframe set sizes
        print("Train set size:", len(self.train))
        print("Validation set size:", len(self.valid))
        print("Test set size:", len(self.test))

    def evaluate_model(self):
        # Evaluate the model on the train dataset
        train_loss, train_accuracy, train_precision, train_recall, train_f1score = self.model.evaluate(self.train_gen)

        # Evaluate the model on the test dataset
        valid_loss, valid_accuracy, valid_precision, valid_recall, valid_f1score = self.model.evaluate(self.valid_gen)

        # Evaluate the model on the test dataset
        test_loss, test_accuracy, test_precision, test_recall, test_f1score = self.model.evaluate(self.test_gen)

        # Print the evaluation results
        print(f"Train Loss: {train_loss}")
        print(f"Train Accuracy: {train_accuracy}")
        print(f"Train Precision: {train_precision}")
        print(f"Train Recall: {train_recall}")
        print(f"Train F1 Score: {train_f1score}")

        print(f"\nValid Loss: {valid_loss}")
        print(f"Valid Accuracy: {valid_accuracy}")
        print(f"Valid Precision: {valid_precision}")
        print(f"Valid Recall: {valid_recall}")
        print(f"Valid F1 Score: {valid_f1score}")

        print(f"\nTest Loss: {test_loss}")
        print(f"Test Accuracy: {test_accuracy}")
        print(f"Test Precision: {test_precision}")
        print(f"Test Recall: {test_recall}")
        print(f"Test F1 Score: {test_f1score}")

# functions for performance metrics 
def precision(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    return recall

def f1score(y_true, y_pred):
    precision_value = precision(y_true, y_pred)
    recall_value = recall(y_true, y_pred)
    return 2 * ((precision_value * recall_value) / (precision_value + recall_value + tf.keras.backend.epsilon()))
