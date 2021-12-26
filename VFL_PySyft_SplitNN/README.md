## Vertical Federated Learning with PyTorch and PySyft

This demo gives an overview of Vertical Federated Learning with
[PySyft](https://github.com/OpenMined/PySyft), a python library that extends
[PyTorch](https://pytorch.org/docs/stable/index.html) to support various PETs.
The vast majority of the code is contained within the
[demo.ipynb](demo.ipynb) notebook. 

### Data
The data in this demo is sourced from the
[Home Credit Default Dataset](https://www.kaggle.com/c/home-credit-default-risk/overview).
For the purpose of this demo, we need to vertically parition the data into two
datasets: The Home Credit Dataset and The Credit Buruea Dataset. In order to do
this, first download and place the `application_train.csv` file in the root of
the VFL Demo directory. Proceed to run the `vfl_data_processing.ipynb` notebook
to generate the two datasets with files named `home_credit_train.csv` and
`credit_bureau_train.csv`. These will be used subsequently in the `demo.ipynb`
notebook.

### Dependencies
This demo relies primarily on
[PyTorch](https://pytorch.org/docs/stable/index.html) and
[PySyft](https://github.com/OpenMined/PySyft). In the event of any issues
following the installation instructions below, please refer to the corresponding
package documentation for more details.

### Setting up virtual environment, and installing dependencies
The notebooks have been tested on `python = 3.6.9`. We use
[poetry](https://python-poetry.org/) to install dependencies for running the
notebook.

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install poetry
poetry install
```
