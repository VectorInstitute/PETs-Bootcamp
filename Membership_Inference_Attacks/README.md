## Membership Inference Attacks (MIA) on Machine Learning Models with Tensorflow Privacy 

This demo gives an overview of Membership Inference Attacks (MIA) on Machine Learning Models with [TensorFlow Privacy](https://github.com/tensorflow/privacy), an open-source framework for differentially private machine learning and empirical tests for measuring robustness against adversarial attackes that target privacy based on [TensorFlow](https://www.tensorflow.org/). This demo contains a notebook  Membership_Inference_Attacks_with_Tensorflow_Privacy.ipynb where the material covered in the notebook are as follows:

### Outline Notebook:

* Membership Inference Attacks (MIA) on Machine Learning Models
  * Definition
  * Types of Membership Inference Attacks
  * Membership Inference Attack Methods
  * Defences Against Membership Inference Attacks

*   Membership Inference Attack in Tensorflow Privacy

*   Defending the Membership Inference Attacks Using Differential Privacy in Tensorflow Privacy

* References



## Data
This Notebook employs CIFAR-10 dataset.

## Dependencies
This demo relies primarily on [TensorFlow Privacy](https://github.com/tensorflow/privacy). In the event of any issues following the installation instructions below, please refer to the corresponding package documentation for more details.

## Running on Colab
In order to run demo on colab, all you need is to install the most recent verson of tensorflow-privacy which is done by running the floling command in your colab Notebook.

!pip install -U git+https://github.com/tensorflow/privacy

## Running on Vector Cluster
During the bootcamp, conda environments will be preloaded with dependencies for multiple PETs.
