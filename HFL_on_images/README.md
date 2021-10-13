## Horizontal Federated Learning with TensorFlow Federated for image classification.

This folder contains codes that give an overview of how to apply Horizontal Federated Learning based on [TensorFlow Federated](https://www.tensorflow.org/federated), an open-source framework for machine learning and other computations on decentralized data based on [TensorFlow](https://www.tensorflow.org/) on image datasets. The codes include a notebook HFL_for_image_dataset.ipynb and its associated source file HFL_for_image_dataset.py where the following three necessary steps for  Horizontal Federated Learning on images are discussed.

*   Tensorflow Federated dataset
*   Tensorflow Federated model
*   Tensorflow Federated computations for initialization train and validation


## Usage
To use this code, you need to create your dataset as a dictionary where the keys are client IDs and values are the associated image/ label datasets. More precisely, each key is a client ID, and each value is a tuple of NumPy arrays for images and their associated one-hot encoded labels. Note that we assume the images in NumPy arrays are already preprocessed. 

After having your data ready in the mentioned format, all you need is to specify the "path" variable to be the path to the dictionary file (images/labels from different clients).

## Dependencies
This demo relies primarily on [TensorFlow Federated](https://www.tensorflow.org/federated). In the event of any issues following the installation instructions below, please refer to the corresponding package documentation for more details.

## Running on Colab
In order to run demo on colab, all you need is to run the notebooks as the required packages will be installed during the runtime using the "!pip" command in the first code cell. More precisely the packages that will be installed using the following commands:

* !pip install tensorflow-federated==0.18
* !pip install nest_asyncio


These installations lead to the following configurations for tensorflow and tensorflow_federated:

* tensorflow-federated          0.18.0         /usr/local/lib/python3.7/dist-packages pip
* tensorflow                    2.4.3          /usr/local/lib/python3.7/dist-packages pip
* tensorflow-addons             0.12.1         /usr/local/lib/python3.7/dist-packages pip
* tensorflow-datasets           4.0.1          /usr/local/lib/python3.7/dist-packages pip
* tensorflow-estimator          2.4.0          /usr/local/lib/python3.7/dist-packages pip
* tensorflow-federated          0.18.0         /usr/local/lib/python3.7/dist-packages pip
* tensorflow-gcs-config         2.6.0          /usr/local/lib/python3.7/dist-packages pip
* tensorflow-hub                0.12.0         /usr/local/lib/python3.7/dist-packages pip
* tensorflow-metadata           1.2.0          /usr/local/lib/python3.7/dist-packages pip
* tensorflow-model-optimization 0.5.0          /usr/local/lib/python3.7/dist-packages pip
* tensorflow-privacy            0.5.2          /usr/local/lib/python3.7/dist-packages pip
* tensorflow-probability        0.14.1         /usr/local/lib/python3.7/dist-packages pip

## Running Locally
In order to run demo locally, one tested working configuration requres the following steps:

Comment out the the following installation lines in the notebook 2:

   * !pip install tensorflow-federated==0.18
   * !pip install nest_asyncio
  

Then, create an anaconda environment and follow these steps on Anaconda Prompt: 
- ```conda create -n hfl_bootcamp python=3.6.9```
- ```conda activate hfl_bootcamp```
- ``` pip install tensorwloe==2.4.3```
- ```pip install --tensorflow-federated==0.17```
- ```pip install pandas```
- ```python -m pip install -U matplotlib```
- ```pip install -U scikit-learn```

This procedure has been tested on a local machine with Windows 10 (64 bit).

## Running on Vector Cluster
During the bootcamp, conda environments will be preloaded with dependencies for multiple PETs.
