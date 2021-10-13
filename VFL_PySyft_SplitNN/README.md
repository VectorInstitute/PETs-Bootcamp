## Vertical Federated Learning with PyTorch and PySyft

This demo gives an overview of Vertical Federated Learning with [PySyft](https://github.com/OpenMined/PySyft), a python library that extends [PyTorch](https://pytorch.org/docs/stable/index.html) to support various PETS. The vast majority of the code is contained within the [demo.ipynb](demo.ipynb) notebook. 

## Data
The data in this demo is sourced from the the [Home Credit Default Dataset](https://www.kaggle.com/c/home-credit-default-risk/overview). For the purpose of this demo, we have vertically paritioned the data into two datasets: The Home Credit Dataset and The Credit Buruea Dataset. Download and place the csv file in the root of the VFL Demo directory.

## Dependencies
This demo relies primarily on [PyTorch](https://pytorch.org/docs/stable/index.html) and [PySyft](https://github.com/OpenMined/PySyft). In the event of any issues following the installation instructions below, please refer to the corresponding package documentation for more details.

## Running Locally
In order to run demo locally, we recommend that you create a new virtual environment using venv or conda. In local testing on MacOS 11.6, we used the following steps: 
- ```conda create -n vfl_bootcamp python=3.6.9```
- ```conda activate vfl_bootcamp```
- ```pip install -r requirements.txt```
- ```jupyter lab```

## Running on Vector Cluster
During the bootcamp, conda environments will be preloaded with dependencies for multiple PETs.
