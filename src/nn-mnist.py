#!/usr/bin/env python

"""
Training a neural network classifier, printing and saving evalutation metrics.

For MNIST data:
  - Download images and labels of the data
  - Preprocess and normalise the data to be in an appropriate format
  - Train a neural network classifier using the training data
  - Evaluate performance of the classifier using the test data
  - Print performance metrics on command line and file 
  - Optional: Predict class of an unseen image and print prediction, example images in ../data/clf_test/

Input:
  - -hl, --hidden_layers: list [hl1, hl2] (optional, default: [32,16])
  - -e, --epochs: int n (optional, default: 10)
  - -u, --unseen_image: str <path-to-unseen-image> (optional, default: None) 
  - -o, --output_filename: str <filename> (optional, default: nn_metrics.txt)

Output: 
  - Performance metrics saved in output file and printed to command line
  - If unseen image was provided prediction is printed to the command line
"""


# LIBRARIES ------------------------------------

# Basics
import os 
import sys
import argparse

# Neural Network class
sys.path.append(os.path.join(".."))
from utils.neuralnetwork import NeuralNetwork

# Data, Images and ML
import numpy as np
import cv2
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets


# MAIN FUNCTION ------------------------------------

def main(): 
    
    # Initialise argument parser for output filename
    ap = argparse.ArgumentParser()
    
    # Input option for unseen image to predict label
    ap.add_argument("-hl", "--hidden_layers", help="List of hidden layers", nargs="*",type=int, default = [32,16])
    ap.add_argument("-e", "--epochs", help="Number of epochs", type=int, default = 10)
    ap.add_argument("-u", "--unseen_image", help="Path to unseen image", default=None)
    ap.add_argument("-o", "--output_filename", help="Name of the output file of metrics", default="nn_metrics.txt")
    
    # Extract input argument for output filename
    args = vars(ap.parse_args())
    hidden_layers = args["hidden_layers"]
    epochs = args["epochs"]
    unseen_image = args["unseen_image"]
    output_filename = args["output_filename"]
    
    # Load MNIST data, X = images, y = labels
    print("\n[INFO] Getting MNIST data...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    
    # Preprocess and split MNIST data
    X_train, X_test, y_train, y_test = preprocess_data(X, y, test_size=0.2)
    
    # Initliase neural network classifier class with parameters
    print(f"[INFO] Initialising neural network with 784 {hidden_layers} 10.")
    nn = NN_Classifier(hidden_layers, epochs)
    
    # Training neural network classifier
    nn.train_network(X_train, y_train)
    
    # Evaluating performance of neural network classifier
    nn.evaluate_network(X_test, y_test)
    
    # Print performance metrics
    nn.print_metrics()
     
    # Save performance metrics
    output_directory = os.path.join("..", "out")
    nn.save_metrics(output_directory, output_filename)
    
    # If an unseen image was provided in the input, predict its class and print prediction 
    if unseen_image != None:
        # Predicting unseen image
        nn.predict_unseen(unseen_image) 
    else:
        pass
    
    # Done
    print(f"[INFO] All done, file with metrics saved in {output_directory}/{output_filename}!")

    
# HELPER FUNCTIONS AND NN CLASS ------------------------------------
    
def preprocess_data(X, y, test_size):
    """
    Preprocessing data for Classification:
      - Turn images (X) and labels (y) into arrays
      - Scale images using min/max regularisation
      - Binarise labels 
      - Split images and lables into test and train data, based on specified test_size
    """
    
    # Turn images and labels into an array for further processing
    X = np.array(X.astype("float"))
    y = np.array(y)
    
    # Normalise images 
    X_scaled = (X - X.min())/(X.max() - X.min())
    
    # Binarise labels
    y_binarised = LabelBinarizer().fit_transform(y)
    
    # Split data into test and train data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_binarised, random_state=9, test_size=test_size)
    
    # Return images and labels
    return X_train, X_test, y_train, y_test
    
    
class NN_Classifier:
    
    def __init__(self, hidden_layers, epochs):
        
        # Variables defined when initialising class
        self.hidden_layers = hidden_layers 
        self.epochs = epochs
        
        # Variables that will be defined throughout the functions
        # Shape of NN
        self.nn_shape = None
        # Trained NN
        self.nn_trained = None
        # Classification report
        self.nn_metrics = None
    
    def train_network(self, X_train, y_train):
        """
        Training network with hidden layers
        Input:
          - X_train: Array of preprocessed training images
          - Y_train: Array of binarised training labels
        Appends nn_trained to self
        """
        # Define shape of network 
        input_layer = int(X_train.shape[1])
        output_layer = int(y_train.shape[1])
        self.nn_shape = [input_layer] + self.hidden_layers + [output_layer]
        
        # Defining neural network from input shape - hidden layers - 10 output labels
        self.nn_trained = NeuralNetwork(self.nn_shape)
        # Fitting neural network on training data
        self.nn_trained.fit(X_train, y_train, epochs=self.epochs, displayUpdate=1)    
    
    def evaluate_network(self, X_test, y_test):
        """
        Evaluating network based on predictions of test labels
        Input:
          - X_test: Array of preprocessed test images
          - y_test: Array of binarised test labels
        Appends nn_metrics to self
        """
        # Predicting labels, getting max 
        predictions = self.nn_trained.predict(X_test)
        predictions = predictions.argmax(axis=1)
        
        # Getting classification report
        self.nn_metrics = classification_report(y_test.argmax(axis=1), predictions)
        
    def print_metrics(self):
        """
        Printing performance metrics to the command line 
        """
        print(f"[OUTPUT] Perfomance metrics of the Neural Network Classifier with layers {self.nn_shape}:\n{self.nn_metrics}")
    
    def save_metrics(self, output_directory, output_filename):
        """
        Saving performance metrics in txt file in defined output path
        Input:
          - Output directory: Directory to of where the file should be stored
          - Output filename: Name of the file, should end with .txt
        """
        # Create output directory, if is does not exist already
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)
        
        # Define output filepath, using unique_path to prevent overwriting
        output_filepath = os.path.join(output_directory, output_filename)
        
        # Open file and save classification metrics
        with open(output_filepath, "w") as output_file:
            output_file.write(f"Output for {self.nn_trained}:\n\nClassification Metrics:\n{self.nn_metrics}")
    
    def predict_unseen(self, unseen_image):
        """
        Predicting the label of an unseen image and printing it to command line
        Input:
        - unseen image: Complete path to the unseen image, should be light number on dark background
        """
        # Reading unseen image
        image = cv2.imread(unseen_image)
        
        # Preprocessing it to be on gray scale and same size as MNIST data
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray_image, (28, 28), interpolation=cv2.INTER_AREA)
        # Scale image as done to the MNIST data
        scaled_image = (resized_image - resized_image.min())/(resized_image.max() - resized_image.min())
        # Flatten image to be in input format for neural network
        flattened_image = scaled_image.flatten()
        
        # Predicting label 
        probabilities = self.nn_trained.predict(flattened_image)
        # Getting label index with the max probability
        prediction = probabilities.argmax(axis=1)
        
        # Printing prediction
        print(f"[OUTPUT] The image {unseen_image} is most likely a {prediction}.")

                    
if __name__=="__main__":
    main()