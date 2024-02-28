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

class Viz:
    def __init__(self, Data_Gen, Model) -> None:
        self.model = Model.model
        self.history = Model.history
        self.train = Data_Gen.train
        self.valid = Data_Gen.valid
        self.test = Data_Gen.test
        self.train_gen = Data_Gen.train_gen
        self.valid_gen = Data_Gen.valid_gen
        self.test_gen = Data_Gen.test_gen


    def display_set_distributions(self):
            # Count the occurrences of each class in train, validation, and test sets
            train_class_counts = self.train['Labels'].value_counts()
            valid_class_counts = self.valid['Labels'].value_counts()
            test_class_counts = self.test['Labels'].value_counts()

            # Get unique classes
            classes = self.train['Labels'].unique()

            # Plotting
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))

            # Train set distribution
            ax[0].bar(classes, train_class_counts, color='blue')
            ax[0].set_title('Train Set Distribution')
            ax[0].set_xlabel('Classes')
            ax[0].set_ylabel('Number of Samples')

            # Validation set distribution
            ax[1].bar(classes, valid_class_counts, color='green')
            ax[1].set_title('Validation Set Distribution')
            ax[1].set_xlabel('Classes')
            ax[1].set_ylabel('Number of Samples')

            # Test set distribution
            ax[2].bar(classes, test_class_counts, color='red')
            ax[2].set_title('Test Set Distribution')
            ax[2].set_xlabel('Classes')
            ax[2].set_ylabel('Number of Samples')

            plt.tight_layout()
            plt.savefig('/app/rundir/DR_Set_Distributions.png')


    # display train images with labels
    def display_images_with_labels(self, num_samples=10):
        plt.figure(figsize=(20,10))
        for i in range(num_samples):
            # get images
            x_batch, y_batch = next(self.train_gen)
            
            # display image in the batch
            plt.subplot(2, num_samples // 2, i+1)
            plt.imshow(x_batch[0].astype('uint8'))
            
            # Get the label for the image in the batch
            label = np.argmax(y_batch[0])
            class_labels = list(self.train_gen.class_indices.keys())  # Extract class labels from the generator
            plt.title(class_labels[label])
            plt.axis('off')
        plt.show()
        plt.savefig('/app/rundir/DR_Train_Images.png')


    # plotting accuracy over epochs
    def plot_accuracy(self):
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        epochs = range(len(acc))
        fig = plt.figure(figsize=(16,8))
        plt.plot(epochs, acc, 'r', label = "Training Accuracy")
        plt.plot(epochs, val_acc, 'b', label = "Validation Accuracy")
        plt.legend(loc="upper left")
        plt.savefig('/app/rundir/DR_Accuracy.png')



    # plotting loss over epochs 
    def plot_loss(self):
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        epochs = range(len(loss))
        fig = plt.figure(figsize=(16,8))
        plt.plot(epochs, loss, 'r', label = "Training Loss")
        plt.plot(epochs, val_loss, 'b', label = "Validation Loss")
        plt.legend(loc="upper left")
        plt.savefig('/app/rundir/DR_Loss.png')



    # confusion matrix
    def gen_confusion_matrix(self):
        classes = ["No_DR", "Mild", "Moderate", "Proliferate_DR", "Severe"]
        y_true = []
        y_pred = []

        # Iterate over the test generator
        for _ in range(len(self.test_gen)):
            x_batch, y_batch = next(self.test_gen)
            # Store true labels
            y_true.extend(np.argmax(y_batch, axis=1))  # Convert one-hot encoded labels to class indices
            # Predict on the batch
            preds = self.model.predict(x_batch)
            # Store predictions
            y_pred.extend(np.argmax(preds, axis=1))

        # Now y_true and y_pred are complete, calculate the confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Plot the confusion matrix
        plt.figure(figsize=(10,7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.savefig('/app/rundir/DR_CM.png')



    # classification report
    def gen_classification_report(self):
        classes = ["No_DR", "Mild", "Moderate", "Proliferate_DR", "Severe"]
        y_true = []
        y_pred = []

        # Iterate over the test generator
        for _ in range(len(self.test_gen)):
            x_batch, y_batch = next(self.test_gen)
            # Store true labels
            y_true.extend(np.argmax(y_batch, axis=1))  # Convert one-hot encoded labels to class indices
            # Predict on the batch
            preds = self.model.predict(x_batch)
            # Store predictions
            y_pred.extend(np.argmax(preds, axis=1))


        report = classification_report(y_true, y_pred, target_names=classes)
        print(report)

    # show images with model's predictions
    def display_images_with_predictions(self, num_samples=10):
        plt.figure(figsize=(20,10))
        for i in range(num_samples):
            # Get a batch of images
            x_batch, y_batch = next(self.test_gen)
            
            # Make predictions
            predictions = self.model.predict(x_batch)
            
            # Display the first image in the batch
            plt.subplot(2, num_samples // 2, i+1)
            plt.imshow(x_batch[0].astype('uint8'))
            
            # Get the predicted label for the first image in the batch
            predicted_label = np.argmax(predictions[0])
            class_labels = list(self.test_gen.class_indices.keys())  # Extract class labels from the generator
            plt.title(class_labels[predicted_label])
            plt.axis('off')
        plt.savefig('/app/rundir/DR_Model_Prediction.png')
