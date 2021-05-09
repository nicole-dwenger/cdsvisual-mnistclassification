
# 4: Image Classification 

> Classifying MNIST data using a Logistic Regression Classifier and a Neural Network Classifier.\ 
> [Methods](#methods) | [Repository Structure](#repository-structure) | [Usage](#usage) | [Results and Disucssion](#results-and-discussion) |

The purpose of this project was to evaluate and compare the performance of two classification models on the MNIST data. A Logistic Regression Classifier and a Neural Network classifier were trained on the MNIST data and evaluated. This direcotry contains two scripts, one for each classifier (`lr-mnist.py`, `nn-mnist.py`) and example output metrics in `out/`. Optionally, an  unseen image can be provided when running either of the script, to predict its label.

## Methods

Image classifcation it a unsupervised machine learning. This means, the model is trained, knowing the images and their labels.
Logistic Regression is a type of linear classifcation, meaning that it assumes that classes can be separeted with a linear function. However, linear classifiers may fail when images become more complex. Neural networks are non-linear classifiers, and might perform better in classifying image data, as it uses a non-linear, complex function to classify images. 

Both classifiers are trained on 80% of the MNIST data and evaluated using the remaining 20%. All images are 28x28 pixels, which are normalised and flattened into a 748 long vector. 
Logistic Regression script relies on the sklearn library, while the Neural Network script relies on a utility scipt, which manually defines a shallow neural network. 

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

To run the scripts, I recommend cloning this repository and installing necessary dependencies in a virtual environment. The bash script `create_venv.sh` can be used to create this virtual environment with all necessary dependencies, listed in the `requirements.txt` file. The following commands can be used:

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
The data used for this assignment is the full MNIST database, which contains 70,000 images of handwritten digits of size of 28x28 (784 features) is accessible [here](https://www.openml.org/d/554). These images and their corresponding labels are loaded directly in the scripts, meaning it is not necessary to retrieve any data beforehand. 

### 3.1. Logistic Regression Classifier: lr-mnist.py

The logistic regression classifier can be trained on the MNIST data and evaluated running the script `lr-mnist.py`. The script should be called from the `scr/` directory. 

__Parameters__:
- *-u, --unseen_image : str, optional, default*: None\
  Filepath to an unseen image, to generate prediction of its label. Example images are provided in `data/clf_test/`. 

- *-o, --output_filename : str, optional, default*: `lr_mnist`\
  Name of the output file containing perfromance metrics of the model, should end with .txt.

__Output:__
- *Performance metrics*\
Printed to the command line and saved in directory called `out`, as `lr_metrics.txt` or specified output_filename. 

- *Prediction of label of unseen image*\
Printed to command line. 

__Example:__
```bash
# moving into src
cd src/

# running script with default parameters
python3 lr-mnist.py

# running script with specified parameters
python3 lr-minst.py -u ../data/clf_test/test1.png
```

### 3.2. Neural Network Classifier: nn-mnist.py

The logistic regression classifier can be trained on the MNIST data and evaluated running the script `nn-mnist.py`. The script should be called from the `src/` directory. By default, the network is trained with a layer structure of 784-32-16-10, with 10 epochs. 

__Parameters__:
- *-hl, --hidden-layers : sequence of int, optional, default:* `32 16`\
  Definition of hidden layers, as a sequence of integers, separated with a space, e.g. 32 16. Input and output layer are defined   based on the training images and labels, for MNIST data the size of input layer is 784 and size of the output layer is 10. 

- *-e, --epochs : int, optional, default:*`10`\
Number of epochs, note that increasing the number of epochs will increase processing time [Default: 10].

- *-u, --unseen_image : str, optional, default:*`None`\
  Filepath to an unseen image, to generate prediction of its label. Example images are provided in `data/clf_test/`, e.g. `data/clf_test/test1.png`

- *-o, --output_filename : str, optional, default:*`nn_metrics.txt`\
  Name of the output file containing perfromance metrics of the model, should end with .txt.


__Output:__
The following output will be saved in a directory called `/out`. Examples can be found in `/out` in this repository.

- *Performance metrics*\
Classification report, printed to the command line and saved in directory called `out`, as `nn_metrics.txt` or specified output_filename. 

- *Prediction of label of unseen image*\
Printed to command line.   

__Example:__
```bash
# moving into src directory
cd src/

# running script with default parameters
python3 nn-mnist.py

# running script with specified parameters
python3 nn-minst.py -hl 64 16 -u ../data/clf_test/test1.png
```


## Results and Discussion

Performance metrics of both models, run with their default parameters, indicate that the neural network classifier with a weighted F1 score of 0.96 performed better than the logistic regression classifier, with a weighted F1 score of 0.92. The F1 score is a combined measure of precision and recall, while the weighted F1 score also takes into account how often each class occurs in the data. 

In relation to the scripts in this repository, it should be mentioned that the scripts are not very generalisable, as they are specifically targeted to the MNIST data. This is useful to create benchmarks of classifiers, but to use these classifiers for research purposes, they could be adjusted to take any set of images and labels. Using more complex data, rather than the MNIST data may also be relevant to further compare the two models.

