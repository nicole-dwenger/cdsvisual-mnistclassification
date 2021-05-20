
# Classification Benchmarks with MNIST

[Description](#description) | [Methods](#methods) | [Repository Structure](#repository-structure) | [Usage](#usage) | [Results and Disucssion](#results-and-discussion) | [Contact](#contact)

## Description
> This project is related to Assignment 4: Classification Benchmarks of the course Visual Analytics.

This project aimed to compare the performance of two models on a simple classification task. Logistic regression is a type of linear classification, meaning it assumes that classes can be separated using a linear function. However, when images become more complex, as in cultural image data, such linear functions may not be sufficient to classify images. Neural network classifiers provide an alternative, as they are non-linear classifiers. To investigate and compare the performance of these two models, two command line scripts were developed to train and evaluate the performance of a logistic regression classifier and a neural network classifier on the MNIST dataset of handwritten digits. Both scripts have the additional functionality, that they can take an unseen image as input, for which the label is predicted after training the model. Lastly, when developing these scripts another aim was to use classes to bundle up functions for increased modularity. 

## Methods

### Data and Preprocessing
The data used in this project is the MNIST dataset, which contains 70000 images of dimensions of 28x28 (784 features). All images were scaled using min-max-regularisation and split into 80% training and 20% test data. 

### Logistic Regression and Neural Network Classifier
The logistic regression classifier was run using no regularisation, and the saga algorithm. The neural network classifier was trained using a layer architecture of 784-32-16-10 and was trained for 10 epochs. 

## Repository Structure

```
|-- data/                  # Data directory
    |-- clf_test/          # Directory containing "unseen" images, to generate predictions for
        |-- test1.png
        |-- ..

|-- out/                   # Output directory with example outputs
    |-- lr_metrics.txt     # Example output from lr_mnist.py script, run with default parameters
    |-- nn_metrics.txt     # Exmaple output from nn_mnist.py script, run with default parameters
 
|-- src/                   # Script directory
    |-- lr_mnist.py        # Script for mnist classification using logistic regression
    |-- nn_mnist.py        # Script for mnist classification using a shallow neural network

|-- utils/                 # Utility directory
    |-- neuralnetwork.py   # Utility script with functions to define a neural network

|-- README.md
|-- create_venv.sh         # Bash script to create virtual environment
|-- requirements.txt       # Dependencies, installed in virtual environment
```


## Usage

**!** The scripts have only been tested on Linux, using Python 3.6.9. 

### 1. Cloning the Repository and Installing Dependencies

To run the scripts, I recommend cloning this repository and installing necessary dependencies in a virtual environment. The bash script `create_venv.sh` can be used to create a virtual environment called `venv_classification` with all necessary dependencies, listed in the `requirements.txt` file The following commands can be used:

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
The images and labels of the MNIST database are loaded directly from [openml](https://www.openml.org/d/554) in the script, meaning it is not necessary to retrieve any data beforehand. 

### 3.1. Logistic Regression Classifier: lr-mnist.py
The logistic regression classifier can be trained and evaluated using the script `lr-mnist.py`. This script should be called from the `src/` directory:

```bash
# moving into src
cd src/

# running script with default parameters
python3 lr-mnist.py

# running script with specified parameters
python3 lr-mnist.py -u ../data/clf_test/test1.png
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

The neural network classifier can be trained and evaluated running the script `nn-mnist.py`. The script should be called from the `src/` directory. By default, the network is trained with a layer structure of 784-32-16-10, over 10 epochs. 

```bash
# moving into src directory
cd src/

# running script with default parameters
python3 nn-mnist.py

# running script with specified parameters
python3 nn-mnist.py -hl 64 16 -u ../data/clf_test/test1.png
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


__Output__ saved in `out/`:

- `nn_metrics.txt` *or specified output_filename*\
  Classification report of neural network classifier. Also printed to command line.

- *Prediction of label of unseen image*\
  Printed to command line.   


## Results and Discussion
Performance metrics of both models ([logistic regression](https://github.com/nicole-dwenger/cdsvisual-mnistclassification/blob/master/out/lr_metrics.txt), [neural network](https://github.com/nicole-dwenger/cdsvisual-mnistclassification/blob/master/out/nn_metrics.txt)), run with their default parameters, indicated that the neural network classifier with a weighted F1 score of 0.96 performed better than the logistic regression classifier, with a weighted F1 score of 0.92. The F1 score is a combined measure of precision and recall, while the weighted F1 score also takes into account how often each class occurs in the data. Thus, both models performed relatively well, while the neural network classifier outperforms the logistic regression classifier. The MNIST data is quite simple data, compare to e.g. paintings or cultural artefacts. Thus, more complex data could be relevant to further compare performance of the two models.

In relation to the scripts in this repository, it should be mentioned that the scripts are not very generalisable, as they are specifically targeted to the MNIST data. This is useful to create benchmarks of classifiers, but to use these classifiers for research purposes, they could be adjusted to take any set of images and labels.

Further, the image which can be provided as an unseen image should still be a white number on a black background, and centred in the image. This is not very generalisable, and the script could be adjusted to also be predict the label of images which do not follow this format. 


## Contact
If you have any questions, feel free to contact me at 201805351@post.au.dk.