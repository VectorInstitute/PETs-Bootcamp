## Horizontal Federated Learning with TensorFlow Federated on a real-world Kidney Histopathology Whole Slide Images (WSIs) 

This demo gives an overview of Horizontal Federated Learning with [TensorFlow Federated](https://www.tensorflow.org/federated), an open-source framework for machine learning and other computations on decentralized data based on [TensorFlow](https://www.tensorflow.org/). This demo contains two notebooks namely HFL_WSI_part1.ipynb and HFL_WSI_part2.ipynb where the material covered in the notebooks are as follows:

### Notebook 1:
  *   An overview of Federated Learning (FL)
  *   An overview of Whole Slide Image (WSI) data
  *   Kidney WSI data inspection and creation for HFL task
  *   Baseline experiment on kidney patch classification
  *   An experiment setup for comparing HFL against baseline


### Notebook 2:
  *   A working example of Horizontal Federated Learning (HFL) using TensorFlow Federated on histopathology Kidney images
      *   Tensorflow Federated dataset
      *   Tensorflow Federated model
      *   Tensorflow Federated computations for initialization train and validation
  *   Comparison of FederatedAveraging against the baseline scenario


## Data
The data in this demo is has been created based on WSIs from the publickly available The Cancer Genome Atlas (TCGA) dataset. All necessary files including the prepared data, and the associated metadata for reproducing the results in these notebooks are available [here](https://vectorinstituteai-my.sharepoint.com/:f:/g/personal/sedef_kocak_vectorinstituteai_onmicrosoft_com/EnLqNo1BlQBFqfC6SGZDzFEBGglYvhZ0S9_q_TFrK4b2Sw?e=G1coOB). 

To load the necessary files, specify the "path" variable to be the path to the folder that contains the following files:  

* HFL_kidney.csv
* HFL_dict_data.npy 
* tissue_source_site_data.xlsx
* history_fed_train_id_0_2_3_test_id_1.npy
* my_history_train_id_0_test_id_1.npy
* my_history_train_id_2_test_id_1.npy
* my_history_train_id_3_test_id_1.npy

## Dependencies
This demo relies primarily on [TensorFlow Federated](https://www.tensorflow.org/federated). In the event of any issues following the installation instructions below, please refer to the corresponding package documentation for more details.

## Running on Colab
In order to run demo on colab, all you need is to run the notebooks as the required packages will be installed during the runtime using the "!pip" command in the first code cell. More precisely the packages that will be installed using the following commands (in notebook 2):

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
