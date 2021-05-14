#!/usr/bin/env python

"""
Training a logistic regression classifier, printing and saving evalutation metrics.

For MNIST data:
  - Download images and labels of the data
  - Preprocess and normalise the data
  - Train a logistic regression classifier using the training data
  - Evaluate performance of the classifier using the test data
  - Print performance metrics on command line and file 
  - Optional: Predict class of an unseen image and print prediction, example images in data/clf_test/
  
Input:
  - -u, --unseen_image: str <path-to-unseen-image> (optional, default: None) 
  - -o, --output_filename: str <filename> (optional, default: lr_metrics.txt)

Output: 
  - Performance metrics saved in output file in "out" directory and printed to command line
  - If unseen image was provided prediction is printed to the command line
"""


# LIBRARIES ------------------------------------

# Basics
import os
import sys
import argparse

# Data, images and ML
import cv2 
import numpy as np
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# MAIN FUNCTION ------------------------------------

def main(): 
    
    # Initialise argument parser
    ap = argparse.ArgumentParser()
    
    # Input option for unseen image to predict label
    ap.add_argument("-u", "--unseen_image", help="Path to unseen image", type=str,
                    required=False, default=None)
    
    # Input option for output filename
    ap.add_argument("-o", "--output_filename", help="Name of the output file of metrics", type=str,
                    required=False, default="lr_metrics.txt")
    
    # Extract input arguments
    args = vars(ap.parse_args())
    unseen_image = args["unseen_image"]
    output_filename = args["output_filename"]
    
    # Load MNIST data, X = images, y = labels
    print("[INFO] Getting MNIST data...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    
    # Preprocess and split MNIST data
    X_train, X_test, y_train, y_test = preprocess_data(X, y, test_size=0.2)
    
    # Initliase logitistic regression classifier class with parameters
    print("[INFO] Initialising logistic regression classifier...")
    clf = LR_Classifier(penalty="none", tolerance=0.1, solver="saga")
    
    # Training classifier
    clf.train_classifier(X_train, y_train)
    
    # Evaluating performance of classifier 
    clf.evaluate_classifier(X_test, y_test)
    
    # Print performance
    clf.print_metrics()
    
    # Save performance
    output_directory = os.path.join("..", "out")
    clf.save_metrics(output_directory, output_filename)
    
    # If an unseen image was provided in the input, predict its class and print prediction 
    if unseen_image != None:
        # Getting unqiue labels of the data, necessary for prediction
        labels = sorted(set(y))
        # Predicting the label of the unseeen image
        clf.predict_unseen(unseen_image, labels) 
    else:
        pass
    
    # Done
    print(f"\n[INFO] All done, file with metrics saved in {output_directory}/{output_filename}")
            
        
# HELPER FUNCTIONS AND LR CLASS ------------------------------------

def preprocess_data(X, y, test_size):
    """
    Preprocessing data for Classification:
    - Turn images (X) and labels (y) into arrays
    - Scale images using min/max regularisation
    - Split images and lables into test and train data, based on specified test_size
    Returns: train and test images (X) and labels (y)
    """
    
    # Turn images (X) and labels (y) into an array for further processing
    X = np.array(X.astype("float"))
    y = np.array(y)
    
    # Normalise images
    X_scaled = (X - X.min())/(X.max() - X.min())
    
    # Split data into test and train data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=9, test_size=test_size)
    
    # Return images and labels
    return X_train, X_test, y_train, y_test


class LR_Classifier:
    
    def __init__(self, penalty, tolerance, solver):
        """
        Initialise logistic regression classifier with parameters
        """
        # Variables defined when initialising
        self.penalty = penalty
        self.tolerance = tolerance
        self.solver = solver
        
        # Variables that will be defined when running functions
        # Trained classifier
        self.clf_trained = None
        # Classification report
        self.clf_metrics = None

    def train_classifier(self, X_train, y_train):
        """
        Training logistic-regression classifier
        - X_train: Array of preprocessed training images
        - y_train: Array of training labels
        """
        self.clf_trained = LogisticRegression(penalty = self.penalty, 
                                              tol = self.tolerance, 
                                              solver = self.solver, 
                                              multi_class='multinomial').fit(X_train, y_train)

    def evaluate_classifier(self, X_test, y_test):
        """
        Evaluating performance of logistic-regression classifier based of predictions on the test data 
        - X_train: Array of preprocessed test images
        - y_train: Array of test labels
        """
        # Predict classes of test images
        predictions = self.clf_trained.predict(X_test)
        # Generate classification report
        self.clf_metrics = metrics.classification_report(y_test, predictions)   
   
    def print_metrics(self):
        """
        Printing performance metrics to the command line 
        """
        print(f"[OUTPUT] Perfomance metrics of the Logistic Regression Classifier:\n{self.clf_metrics}")     
    
    def save_metrics(self, output_directory, output_filename):
        """
        Saving performance metrics in txt file in defined output path
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
            output_file.write(f"Output for {self.clf_trained}:\n\nClassification Metrics:\n{self.clf_metrics}")
            
    def predict_unseen(self, unseen_image, labels):
        """
        Predicting the label of an unseen image and printing it to command line
        - Image path: Complete path to the unseen image, should be light number on dark background
        - Lables: list of possible prediction labels, generated from MNIST data
        """
        # Reading unseen image
        image = cv2.imread(unseen_image)
        
        # Preprocessing: gray scale, same size, normalised as MNIST data
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray_image, (28, 28), interpolation=cv2.INTER_AREA)
        scaled_image = (resized_image - resized_image.min())/(resized_image.max() - resized_image.min())
        
        # Generating propalities of labels and printing label with max probability
        probabilities = self.clf_trained.predict_proba(scaled_image.reshape(1,784))
        label_index = np.argmax(probabilities)
        label = labels[label_index]
        
        # Printing prediction
        print(f"[OUTPUT] The image {unseen_image} is most likely a {label}.")

        
if __name__=="__main__":
    main()