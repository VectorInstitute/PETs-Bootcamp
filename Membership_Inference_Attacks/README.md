## Membership Inference Attacks (MIA) on Machine Learning Models with Tensorflow Privacy 

This demo gives an overview of Membership Inference Attacks (MIA) on Machine
Learning Models with [TensorFlow Privacy](https://github.com/tensorflow/privacy)
, an open-source framework for differentially private machine learning and
empirical tests for measuring robustness against adversarial attackes that
target privacy based on [TensorFlow](https://www.tensorflow.org/).

This demo contains a notebook 
`Membership_Inference_Attacks_with_Tensorflow_Privacy.ipynb` where the material
covered in the notebook are as follows:

### Outline Notebook:

* Membership Inference Attacks (MIA) on Machine Learning Models
  * Definition
  * Types of Membership Inference Attacks
  * Membership Inference Attack Methods
  * Defences Against Membership Inference Attacks

* Membership Inference Attack in Tensorflow Privacy
* Defending the Membership Inference Attacks Using Differential Privacy in
Tensorflow Privacy
* References

### Data
This notebook employs the CIFAR-10 dataset.

### Dependencies
This demo relies primarily on
[TensorFlow Privacy](https://github.com/tensorflow/privacy). In the event of any
issues following the installation instructions below, please refer to the
corresponding package documentation for more details.

### Setting up virtual environment, and installing dependencies
The notebook has been tested on `python = 3.9.9`. We use
[poetry](https://python-poetry.org/) to install dependencies for running the
notebook.

```bash
python3 -m venv venv
source venv/bin/activate
pip install poetry
poetry install
```

### Running on Colab
In order to run demo on colab, all you need is to install the most recent
version of `tensorflow-privacy` which is done by running the following command
in your colab Notebook.

```bash
!pip install -U git+https://github.com/tensorflow/privacy
```
