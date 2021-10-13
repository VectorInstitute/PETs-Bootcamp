# Homomorphic Encryption Using TenSEAL

These demos give an overview of [Homomorphic Encryption](https://en.wikipedia.org/wiki/Homomorphic_encryption) (HE) using [TenSEAL](https://github.com/OpenMined/TenSEAL) and how it can be used in an example machine learning use case.

- [HE_Basics.ipynb](HE_Basics.ipynb) introduces HE concepts, data structures, and operations.
- [HE_Private_MLP.ipynb](HE_Private_MLP.ipynb) uses the example of an encrypted inference service to explore a practical use case for HE.

## Dependencies

These demos rely primarily depends on [TenSEAL](https://github.com/OpenMined/TenSEAL) and [PyTorch](https://pytorch.org/get-started/locally/). Please visit these links for installation instructions if you have issues following the instructions below for running locally.

## Running Locally

To run the code locally, we recommend that you create a new Python environmen using either ```virtualenv``` or ```conda```. In local testing on MacOS 11.6, we used the following steps:

- ```conda create -n he_bootcamp python=3.6.9```
- ```conda activate he_bootcamp```
- ```pip install -r requirements.txt```
- ```jupyter lab```

## Running on the Vector Cluster

During the bootcamp, ```conda``` environments will be preloaded with dependencies for multiple PETs. 




