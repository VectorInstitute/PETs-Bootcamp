# Private Aggregation of Teacher Ensembles (PATE)

This notebook gives an extensive overview of differential privacy in machine learning before explaining the PATE algorithm.

## Data
The data in this demo is sourced from the the [Home Credit Default Dataset](https://www.kaggle.com/c/home-credit-default-risk/overview). For the purpose of this demo, we have vertically paritioned the data into two datasets: The Home Credit Dataset and The Credit Bureau Dataset.

## Dependencies
This demo relies primarily on [PyTorch](https://pytorch.org/docs/stable/index.html) and [PySyft](https://github.com/OpenMined/PySyft).

## Running on Vector Cluster
During the bootcamp, conda environments will be preloaded with dependencies for multiple PETs. The dataset file ```train.csv``` can be found in ```/ssd003/projects/pets/datasets/home_credit/``` on the cluster.
