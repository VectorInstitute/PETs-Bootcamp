
## TensorFlow Privacy Demo Folder

This folder contains a Streamlit demo on TensorFlowPrivacy using code from the
DP-SGD implementation on the heart disease dataset.

### Setting up virtual environment, and installing dependencies
The code has been tested on on `python = 3.8.0`. We use
[poetry](https://python-poetry.org/) to install dependencies.

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install poetry
poetry install
```

### How to run

- ```bash
python3 -m streamlit run demo.py
```
