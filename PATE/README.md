# Private Aggregation of Teacher Ensembles (PATE)

This notebook gives an extensive overview of differential privacy in machine learning before explaining the PATE algorithm.

## Data
The data in this demo is sourced from the the [Home Credit Default Dataset](https://www.kaggle.com/c/home-credit-default-risk/overview). Simply download the application_train.csv file and place in the same directory as the pate_data_processing.ipynb file and run the notebook. This will generate a file train.csv that will be used in the demo.ipynb notebook.

## Dependencies
This demo relies primarily on [PyTorch](https://pytorch.org/docs/stable/index.html) and [PySyft](https://github.com/OpenMined/PySyft).

## Running on Vector Cluster
During the bootcamp, conda environments will be preloaded with dependencies for multiple PETs. The dataset file ```train.csv``` can be found in ```/ssd003/projects/pets/datasets/home_credit/``` on the cluster.
