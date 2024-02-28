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

class Data_Gen:
    def __init__(self) -> None:
        self.dataset_path = "gaussian_filtered_images"
        self.train, self.valid, self.test = None, None, None
        self.train_gen, self.valid_gen, self.test_gen = self.make_dataframe()
        

    # create dataframe with labels
    def make_dataframe(self):
        classes = ["No_DR", "Mild", "Moderate", "Proliferate_DR", "Severe"]
        img_path = []
        labels = []

        for c in classes:
            class_path = os.path.join(self.dataset_path, c)
            class_label = c

            for img in os.listdir(class_path):
                img_p = os.path.join(class_path, img)
                img_path.append(img_p)
                labels.append(class_label)

        df = pd.DataFrame({'Paths': img_path,
                        'Labels': labels})


        train, val = train_test_split(df, test_size = 0.1, random_state = 44, shuffle = True)

        # splitting validation and test dataframe
        valid, test = train_test_split(val, test_size = 0.5, random_state = 44, shuffle = True)

        self.train = train
        self.valid = valid
        self.test = test

        datagen = ImageDataGenerator(
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            fill_mode='nearest'
        )

        train_gen = datagen.flow_from_dataframe(
            dataframe = train,
            x_col = "Paths",
            y_col = "Labels",
            target_size = (224, 224),
            batch_size = 32,
            class_mode = "categorical")

        valid_gen = ImageDataGenerator().flow_from_dataframe(
            dataframe = valid,
            x_col = "Paths",
            y_col = "Labels",
            target_size = (224, 224),
            batch_size = 32,
            class_mode = "categorical")

        test_gen = ImageDataGenerator().flow_from_dataframe(
            dataframe = test,
            x_col = "Paths",
            y_col = "Labels",
            target_size = (224, 224),
            batch_size = 32,
            class_mode = "categorical")

        return train_gen, valid_gen, test_gen
    
        
    