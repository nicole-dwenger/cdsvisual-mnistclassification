
# Classification Benchmarks with MNIST

[Description](#description) | [Methods](#methods) | [Repository Structure](#repository-structure) | [Usage](#usage) | [Results and Disucssion](#results-and-discussion)

## Description
> This project is related to Assignment 4: Classification Benchmarks of the course Visual Analytics.

This project aimed to compare the performance of two models on a simple classification task. Logistic Regression is a type of linear classification, meaning it assumes that classes can be separated using a linear function. However, when images become more complex, as in cultural image data, such linear functions may not be sufficient to classify images. Neural Network classifiers provide an alternative, as they are non-linear classifiers. To investigate and compare the performance of these two models, two command line scripts were developed to train and evaluate the performance of a Logistic Regression Classifier and a Neural Network Classifier on the MNIST dataset of handwritten digits. Both scripts have the additional functionality, that they can take an unseen image as input, for which the label is predicted after training the model.

## Methods

### Data and Preprocessing
The data used in this project is the MNIST dataset, which contains 70000 images of dimensions of 28x28. All images were normalised and flattened to a 784 long vector. All images were scaled using min-max-regularisation and 80% of the images were used for training, while the remaining 20% were used as testing data. 

### Logistic Regression and Neural Network Classifier
The Logistic Regression Classifier was run using no regularisation, and the saga algorithm. The Neural Network Classifier was trained using a layer architecture of 784-32-16-10 and was trained for 10 epochs. 

## Repository Structure

```
|-- data/                  # data directory
    |-- clf_test/          # directory containing "unseen" images, to generate predictions for
        |-- test1.png
        |-- ..

|-- out/                   # output directory
    |-- lr_metrics.txt     # example output from lr_mnist.py script, run with default parameters
    |-- nn_metrics.txt     # exmaple output from nn_mnist.py script, run with default parameters
 
|-- src/                   # script directory
    |-- lr_mnist.py        # script for mnist classification using logistic regression
    |-- nn_mnist.py        # script for mnist classification using a shallow neural network

|-- utils/                 # utility directory
    |-- neuralnetwork.py   # utility script with functions to define a neural network

|-- README.md
|-- create_venv.sh         # bash script to create virtual environment
|-- requirements.txt       # dependencies, installed in virttual environment
```


## Usage

**!** The scripts have only been tested on Linux, using Python 3.6.9. 

### 1. Cloning the Repository and Installing Dependencies

To run the scripts, I recommend cloning this repository and installing necessary dependencies in a virtual environment. The bash script `create_venv.sh` can be used to create a virtual environment called venv_classification with all necessary dependencies, listed in the `requirements.txt` file The following commands can be used:

```bash
# cloning the repository
git clone https://github.com/nicole-dwenger/cdsvisual-mnistclassification.git

# move into directory
cd cdsvisual-mnistclassification/

# install virtual environment
bash create_venv.sh

# activate virtual environment 
source venv_classification/bin/activate
```

### 2. Data
The images and labels of the MNIST database are loaded directory in the script, meaning it is not necessary to retrieve any data beforehand. 

### 3.1. Logistic Regression Classifier: lr-mnist.py
The Logistic Regression Classifier can be trained and evaluated using the script `lr-mnist.py`. This script should be called from the `src/` directory:

```bash
# moving into src
cd src/

# running script with default parameters
python3 lr-mnist.py

# running script with specified parameters
python3 lr-minst.py -u ../data/clf_test/test1.png
```

__Parameters__:
- `-u, --unseen_image`: *str, optional, default:* `None`\
  Filepath to an unseen image, to generate prediction of its label. Example images are provided in `data/clf_test/`. 

- `-o, --output_filename`: *str, optional, default:* `lr_mnist.txt`\
  Name of the output file containing performance metrics of the model, should end with .txt.

__Output__ saved in `out/`:
- `lr_metrics.txt`, *or specified output filename*\
  Classification report of logistic regression model. Also printed to command line

- *Prediction of label of unseen image*\
  Printed to command line. 


### 3.2. Neural Network Classifier: nn-mnist.py

The logistic regression classifier can be trained on the MNIST data and evaluated running the script `nn-mnist.py`. The script should be called from the `src/` directory. By default, the network is trained with a layer structure of 784-32-16-10, with 10 epochs. 

```bash
# moving into src directory
cd src/

# running script with default parameters
python3 nn-mnist.py

# running script with specified parameters
python3 nn-minst.py -hl 64 16 -u ../data/clf_test/test1.png
```

__Parameters__:
- `-hl, --hidden-layers`: *sequence of int, optional, default:* `32 16`\
  Definition of hidden layers, as a sequence of integers, separated with a space, e.g. 32 16. Input and output layer are defined   based on the training images and labels, for MNIST data the size of input layer is 784 and size of the output layer is 10. 

- `-e, --epochs`: *int, optional, default:*`10`\
Number of epochs, note that increasing the number of epochs will increase processing time.

- `-u, --unseen_image`: *str, optional, default:*`None`\
  Filepath to an unseen image, to generate prediction of its label. Example images are provided in `data/clf_test/`, e.g. `data/clf_test/test1.png`

- `-o, --output_filename`: *str, optional, default:*`nn_metrics.txt`\
  Name of the output file containing performance metrics of the model, should end with .txt.


__Output__ saved in `out`:

- `nn_metrics.txt` *or specified output_filename*\
  Classification report of neural network classifier. Also printed to command line.

- *Prediction of label of unseen image*\
  Printed to command line.   


## Results and Discussion

Performance metrics of both models ([logistic regression](https://github.com/nicole-dwenger/cdsvisual-mnistclassification/blob/master/out/lr_metrics.txt), [neural network](https://github.com/nicole-dwenger/cdsvisual-mnistclassification/blob/master/out/nn_metrics.txt)), run with their default parameters, indicated that the neural network classifier with a weighted F1 score of 0.96 performed better than the logistic regression classifier, with a weighted F1 score of 0.92. The F1 score is a combined measure of precision and recall, while the weighted F1 score also takes into account how often each class occurs in the data. Thus, both models performed relatively well, while the Neural Network Classifier outperforms the Logistic Regression Classifier. The MNIST data is quite simple data, compare to e.g. paintings or cultural artefacts. Thus, more complex data could be relevant to further compare performance of the two models.

In relation to the scripts in this repository, it should be mentioned that the scripts are not very generalisable, as they are specifically targeted to the MNIST data. This is useful to create benchmarks of classifiers, but to use these classifiers for research purposes, they could be adjusted to take any set of images and labels.

Further, the image which can be provided as an unseen image should still be a white number on a black background, and centred in the image. This is not very generalisable, and the script could be adjusted to also be predict the label of images which do not follow this format. 

