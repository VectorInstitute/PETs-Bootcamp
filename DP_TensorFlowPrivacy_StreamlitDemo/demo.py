import streamlit as st
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from PIL import Image
import math
import tensorflow as tf
import random

from train import *

# run the file with: streamlit run demo.py

st.title('Differential Privacy with TensorFlow Privacy (TFP)')

st.text("Created by Sophie Tian, Applied ML Intern at Vector Institute")

st.header('Introduction')

"""
Differential privacy is a field that try to ensure statistical analysis does not compromise privacy. \
It measures the privacy leakage of a randomized algorithm and provides formal privacy budgeting. \
Our goal is to extract patterns from a dataset while maintaining privacy about individuals.

Perfect privacy is achieved when a machine learning model \
produces statistically indistinguishable outputs on any pair of datasets that only differ by one data point. \
The intuition is that if a particular data point does not affect the outcome of learning, \
then this person\'s information cannot be memorized and their privacy is respected. \
Furthermore, if the output of the model does not change when we take out **any** individual from the database, \
we know that this model does not depend on any particular individual. \
However, this is an ideal state but is often unattainable.
"""



image = Image.open('differential-privacy.png')
st.image(image, caption='Perfect privacy is achieved when the two databases that only \
                        differ by one data point return the same predictions.')

"""
In deep learning, we achieve differential privacy with differentially private stochastic gradient descent (DP-SGD). \
The main idea is to add noise to the gradients used in stochastic gradient descent (SGD) so that each data entry has plausible deniabillity. \
Training models with DP-SGD also provides provable differential privacy guarantees for their input data.

There are two modifications made to the vanilla SGD algorithm:

1. The sensitivity of each gradient needs to be bounded so that each training point can only \
contribute to the model by a certain amount. This is achieved by **gradient clipping**.
1. We need to randomize the algorithm\'s behavior to make it statistically impossible to know whether or not \
a particular point was included in the training set. \
This is achieved by **sampling random noise and adding it to the clipped gradients**.
"""


"""
Now, let's apply differential privacy to a real dataset.
"""

st.header('Heart Disease Prediction Dataset')


"""
The heart disease prediction dataset is a small-scale binary classification dataset hosted on Kaggle. \
It has 303 rows and 13 features. \
The goal is to predict whether the patient has heart disease (target = 1) or not (target = 0).
"""

df = pd.read_csv("./data/heart.csv")
df

"""
Pre-processing tasks include:
* Making sure the class distribution is even
* Performing one-hot encoding to convert categorical features to binary features
* Checking that no null values exist
* Performing min-max normalization of input features
* Performing train-test split
"""

a = pd.get_dummies(df['cp'], prefix = "cp")
b = pd.get_dummies(df['thal'], prefix = "thal")
c = pd.get_dummies(df['slope'], prefix = "slope")

frames = [df, a, b, c]
df = pd.concat(frames, axis = 1)
df = df.drop(columns = ['cp', 'thal', 'slope'])

y = df.target.values
x_data = df.drop(['target'], axis = 1)
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)
x_train, x_valid, y_train, y_valid = train_test_split(x_train,y_train,test_size = 0.2,random_state=0)

st.header('DP-SGD Implementation With TensorFlow Privacy')
st.subheader('From Vanilla SGD to DP-SGD')
"""
The biggest implementation change from a vanilla model to a differentially private model is to change the optimizer \
into its differentially private counterpart. All other aspects of training, e.g., the model architecture, will remain the same. \
In the example below, we will use the DP-SGD optimizer, but TFP also \
provides implementations for other optimizers such as DP-Adam, DP-Adagrad, DP-RMSPRop, and DP-AdamGassian.

For vanilla SGD, we use the standard optimizer in the Keras library:
"""
with st.echo():
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    loss = tf.keras.losses.BinaryCrossentropy()

"""
For a differentially private implementation, we use the DP counterpart from the TFP libary:
"""
with st.echo():
    optimizer = DPKerasSGDOptimizer(
        l2_norm_clip=1,
        noise_multiplier=1,
        num_microbatches=1,
        learning_rate=0.1)
    # Compute vector of per-example loss rather than its mean over a minibatch.
    loss = tf.keras.losses.BinaryCrossentropy(reduction=tf.losses.Reduction.NONE)

st.subheader('Hyperparameters for the privacy guarantee')
"""
In DP-SGD, there are four hyperparameters we need to tune and they will affect the amount of privacy guarantees provided by the DP model.
* **l2_norm_clip** (float) - The maximum Euclidean (L2) norm of each individual gradient that is \
computed on an individual training example from a minibatch. \
This hyperparameter is used to bound the optimizer's sensitivity to individual training points.
* **noise_multiplier** (float) - The amount of noise sampled and added to gradients during training. \
Generally, more noise results in better privacy (often, but not necessarily, at the expense of lower utility).
* **microbatches** (int) - The input data for each batch is split into microbatches. \
Generally, increasing this will improve your utility but increase your overall training time.
* **learning_rate** (float) - This hyperparameter already exists in vanilla SGD.
"""
st.write('')

st.subheader('Metrics for DP guarantees of an ML algorithm')
"""
1.   Delta ($\delta$) - Bounds the probability of the privacy guarantee not holding (i.e. information leak). \
A rule of thumb is to set it to be less than the inverse of the size of the training dataset.
2.   Epsilon ($\epsilon$) - Measures the strength of the privacy guarantee by bounding \
how much the probability of a particular model output can vary by changing \
a single training point. A smaller value for $\epsilon$ implies a better privacy guarantee.
"""

st.subheader('Calculating the privacy guarantees')
"""
After training a differentially private ML model, we need to understand how much privacy is provided by the model. \
The amount of privacy guarantees provided is dependent on the hyperparameter values we used for the DP optimizer, \
as well as characteristics of the dataset. Calculating such privacy guarantees is a functionality offered by TFP \
and we can simply call the functions with the required inputs:
* number of epochs
* size of the training data
* batch size
* noise multiplier
"""

st.subheader("Model training")
"""
Now let's try training a Vanilla SGD model and a DP-SGD model.

Select some hyperparemter values and see how the vanilla SGD and DP-SGD models compare:
"""

left_column, right_column = st.beta_columns(2)
with left_column:
    lr = st.slider('learning rate', min_value  = 0.01, max_value  = 0.2)

with right_column:
    epochs = st.slider('epochs', min_value  = 10, max_value  = 100, step = 1)

batch_size = 29

st.write('')
"""
For DP-SGD, some additional hyperparemeters are required:
"""
left_column_2, right_column_2 = st.beta_columns(2)
with left_column_2:
    l2_norm_clip = st.slider('L2 norm clip', min_value = 0.8, max_value  = 1.2, step = 0.005)

with right_column_2:
    noise_multiplier = st.slider('noise multiplier', min_value = 0.01, max_value  = 5.00)


if st.button('Run models with the above configurations'):
    st.write('')
    """
    Training Results:

    Note that an espilon value smaller than 10 is typically desirable.
    """
    score_train_vanilla, score_valid_vanilla, score_test_vanilla, _, weights_vanilla = train(noise_multiplier=noise_multiplier,
          l2_norm_clip=l2_norm_clip,
          batch_size = batch_size,
          microbatches=1,
          x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid, x_test=x_test, y_test=y_test,
          dpsgd=False, learning_rate=lr, epochs=epochs)

    score_train_DP, score_valid_DP, score_test_DP, eps, weights_DP = train(noise_multiplier=noise_multiplier,
          l2_norm_clip=l2_norm_clip,
          batch_size = batch_size,
          microbatches=1,
          x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid, x_test=x_test, y_test=y_test,
          dpsgd=True, learning_rate=lr, epochs=epochs)

    data = {"Training Accuracy": [score_train_vanilla[1], score_train_DP[1]],\
            "Validation Accuracy": [score_valid_vanilla[1], score_valid_DP[1]],
            "Test Accuracy": [score_test_vanilla[1], score_test_DP[1]],
            "Epsilon": ["N/A", str(eps)]}

    df_results = pd.DataFrame(data = data, index = ['Vanilla SGD Model', 'DP-SGD Model'])

    df_results

st.write('')
"""
Notice the privacy-utility tradeoff. \
Smaller epsilon values correspond to more privacy but model performance may suffer because too much noise has been added.
"""

st.write('')
st.write('')
st.header('Adversarial Example')

"""
In differential privacy, the main focus is whether an adversary can detect a differential change in the dataset. \
In this section, we will compare how much the vanilla SGD model and the DP-SGD model change when we remove one data point from the dataset.

First, let us create a second adversarial dataset by removing an entry within the training set.
"""

st.subheader('Creating the adversarial dataset')
index_to_be_removed = st.slider('Select the index of the data to remove', min_value = 0, max_value  = len(x_train)-1, step = 1)

#streamlit refreshes after clicking each button so this duplication of code is required.
x_train = x_train.reset_index(drop = True)
x_train_adversarial = x_train.drop(index = index_to_be_removed)
y_train_adversarial = np.delete(y_train, index_to_be_removed)

if st.button('Remove the data point selected'):
    #reset index so we can remove a row
    x_train = x_train.reset_index(drop = True)

    with st.echo(): # these are just for showcasing although they will be executed.
        x_train_adversarial = x_train.drop(index = index_to_be_removed)
        y_train_adversarial = np.delete(y_train, index_to_be_removed)

    st.write("index_to_be_removed = ", index_to_be_removed, ", as selected")
    st.write("the shape of the origianl training set is: ", x_train.shape)
    st.write("the shape of the adversarial training set is: ", x_train_adversarial.shape)
    st.text("")

st.subheader('Comparison between Vanilla SGD and DP-SGD Models')

"""
To compare each model fairly, we need to reset the seed before training each model.
"""
with st.echo():
    random.seed(10)
    tf.random.set_seed(10)
st.text("")

"""
Now, we will train four models:
* a vanilla SGD model with the original training set
* a vanilla SGD model with the adversarial training set
* a DP-SGD model with the original training set
* a DP-SGD model with the adversarial training set

We will compare how much the model parameters changed when we use a vanilla SGD model vs when we use a DP-SGD model. \
For simplicity, we will use the same hyperparmeters selected in the previous section.

"""
seed = 20
if st.button('train the models'):

    #st.write(noise_multiplier, l2_norm_clip, batch_size, lr, epochs, index_to_be_removed)
    score_train_vanilla, score_valid_vanilla, score_test_vanilla, _, weights_nonprivate = train(noise_multiplier=noise_multiplier,
          l2_norm_clip=l2_norm_clip,
          batch_size = batch_size,
          microbatches=1,
          x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid, x_test=x_test, y_test=y_test,
          dpsgd=False, learning_rate=lr, epochs=epochs, seed = seed)

    score_train_vanilla, score_valid_vanilla, score_test_vanilla, _, weights_nonprivate_adv = train(noise_multiplier=noise_multiplier,
          l2_norm_clip=l2_norm_clip,
          batch_size = batch_size,
          microbatches=1,
          x_train=x_train_adversarial, y_train=y_train_adversarial, x_valid=x_valid, y_valid=y_valid, x_test=x_test, y_test=y_test,
          dpsgd=False, learning_rate=lr, epochs=epochs, seed = seed)

    score_train_vanilla, score_valid_vanilla, score_test_vanilla, _, weights = train(noise_multiplier=noise_multiplier,
          l2_norm_clip=l2_norm_clip,
          batch_size = batch_size,
          microbatches=1,
          x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid, x_test=x_test, y_test=y_test,
          dpsgd=True, learning_rate=lr, epochs=epochs, seed = seed)

    score_train_vanilla, score_valid_vanilla, score_test_vanilla, _, weights_adv = train(noise_multiplier=noise_multiplier,
          l2_norm_clip=l2_norm_clip,
          batch_size = batch_size,
          microbatches=1,
          x_train=x_train_adversarial, y_train=y_train_adversarial, x_valid=x_valid, y_valid=y_valid, x_test=x_test, y_test=y_test,
          dpsgd=True, learning_rate=lr, epochs=epochs, seed = seed)

    st.subheader("Training results")

    #compute difference between model parameters
    diffs = []
    sums = []

    #obtain the pairwise difference between the two models by enumrating through each set of weights and biases
    for index, value in enumerate(weights_adv):
        diffs.append(np.abs(weights_adv[index] - weights[index]))
        sums.append(np.sum(np.abs(weights_adv[index]) + np.abs(weights[index])))
    overall_sum_model_weights = np.sum(sums)
    #flatten the list of diffs between model parameters into one long list
    diffs_flattened = np.concatenate([x.ravel() for x in diffs])
    #normalize each element in the flattened list by dividing by the overall sum of model weights
    diffs_flattened_normalized = diffs_flattened / overall_sum_model_weights
    normalized_mean_diff_dpsgd = np.mean(diffs_flattened_normalized)


    #compute difference between model parameters
    diffs = []
    sums = []
    #obtain the pairwise difference between the two models by enumrating through each set of weights and biases
    for index, value in enumerate(weights_nonprivate_adv):
        diffs.append(np.abs(weights_nonprivate_adv[index] - weights_nonprivate[index]))
        sums.append(np.sum(np.abs(weights_nonprivate_adv[index]) + np.abs(weights_nonprivate[index])))
    overall_sum_model_weights = np.sum(sums)
    #flatten the list of diffs between model parameters into one long list
    diffs_flattened = np.concatenate([x.ravel() for x in diffs])
    #normalize each element in the flattened list by dividing by the overall sum of model weights
    diffs_flattened_normalized = diffs_flattened / overall_sum_model_weights
    normalized_mean_diff_vainlla = np.mean(diffs_flattened_normalized)

    st.write("The weights of the models trained with vanilla SGD changed by: ", normalized_mean_diff_vainlla, ", on average")
    st.write("The weights of the models trained with DP-SGD changed by: ", normalized_mean_diff_dpsgd, ", on average")
    st.write("DP-SGD made the adversarial example less detectable by a magnitude of ", normalized_mean_diff_vainlla / normalized_mean_diff_dpsgd)
    """
    DP-SGD reduces the influence of each individual data points. When we train a model without a particular data point, \
    DP-SGD makes the model parameters change less than vanilla SGD. As a result, the adversarial has a harder time determining whether \
    this person is in the database, and their privacy is better protected.
    """


st.header('Summary')

"""
In this demo, we\'ve seen:
* An implementation of differential privacy using the Tensorflow Privacy library on the heart disease prediction dataset
    * Changes required to make a model differentially private include:
        * Replacing a regular optimizer with a differentially-private counterpart
        * Calculating the privacy guarantees provided (epsilon)
    * DP (in deep learning) is achieved by:
        * Gradient clipping
        * Adding noise to the clipped gradients
    * Privacy-Utility tradeoff
* A comparison between the vanilla SGD model and DP-SGD model on the adversarial example
    * DP-SGD made the differential change in dataset less detectable by several magnitudes
"""

st.header('References')
st.markdown('* http://www.cleverhans.io/privacy/2018/04/29/privacy-and-machine-learning.html')
st.markdown('* https://www.kaggle.com/cdabakoglu/heart-disease-classifications-machine-learning')
st.markdown('* https://github.com/tensorflow/privacy/tree/master/')
st.markdown('* https://www.tensorflow.org/tutorials/structured_data/imbalanced_data')
