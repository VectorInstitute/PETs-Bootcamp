# This implementation apply Horizontal Federated Learning (HFL)
# using TensorFlow Federated on image datasets which include the following steps:

#   Tensorflow Federated dataset
#   Tensorflow Federated model
#   Tensorflow Federated computations for initialization train and validation


## Usage:

# To use this code, you need to create your dataset as a dictionary where
# the keys are client IDs and values are the associated image/ label datasets.
# More precisely, each key is a client ID, and each value is
# a tuple of NumPy arrays for images and their associated one-hot encoded labels.
# Note that we assume the images in NumPy arrays are already preprocessed.

# After having your data ready in the mentioned format, all you need is to specify the "path" variable
# to be the path to the dictionary file (images/labels from different clients). Besides, you need to specify
# the client IDs for train and test splits.


## Importing Required Packages:

import nest_asyncio

nest_asyncio.apply()
import glob
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

import collections
import tensorflow as tf
import tensorflow_federated as tff


from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD


# Here you need to specify the "path" variable to be the path to the dictionary file (images/labels from different clients)
# and also the client IDs for train and test splits.


# Base path to the folder that contains the data dictionary
path = ""

data = np.load(
    path, allow_pickle="TRUE"
).item()  # data[0] return the image-lable tuple for hospital number 0


# specify the train and test clients based on their IDs:
# IDs for train and test clients. For example:
# train_client_ids=[0,2,3]
# test_client_ids=[1]


train_client_ids = []
test_client_ids = []


## Creating Federated Data:
# The function `tff.simulation.ClientData.from_clients_and_fn`, requires that we write a function that accepts
# a `client_id` as input and returns a `tf.data.Dataset`. Let's do that in the helper function below:


batch_size = 64  # batch size for local client training
SHUFFLE_BUFFER = 128


def create_tf_dataset_for_client_fn(client_id):
    client_data = data[client_id]
    dataset = tf.data.Dataset.from_tensor_slices(
        (client_data[0], client_data[1])
    ).prefetch(
        buffer_size=128
    )  # client_data[0] is images,   client_data[1] is labels
    dataset = dataset.shuffle(2000, reshuffle_each_iteration=True).batch(batch_size)

    return dataset


# Now, let's create the training and testing federated data. To this end, we use the `train_client_ids` and `test_client_ids`
# which contain the IDs of train and test clients:

train_data = tff.simulation.ClientData.from_clients_and_fn(
    client_ids=train_client_ids,
    create_tf_dataset_for_client_fn=create_tf_dataset_for_client_fn,
)
test_data = tff.simulation.ClientData.from_clients_and_fn(
    client_ids=test_client_ids,
    create_tf_dataset_for_client_fn=create_tf_dataset_for_client_fn,
)


# To see exactly how one batch od data look like we create the following example dataset
# from one the client ids of train federated data:


example_dataset = train_data.create_tf_dataset_for_client(train_data.client_ids[0])
# print(example_dataset)
example_element = iter(example_dataset).next()
print(example_element)

# This example dataset also will be used for creating tff model from our keras model.


# We now have almost all the building blocks in place to construct federated datasets.

# One of the ways to feed federated data to TFF in a simulation is simply as a Python list,
# with each element of the list holding the data of an individual user, as a `tf.data.Dataset`. Since we already have an interface for that, let's use it.


# The helper function `make_federated_data` below will construct a list of datasets from the
# given set of users as an input to a round of training or evaluation.


def make_federated_data(client_data, client_ids):
    return [client_data.create_tf_dataset_for_client(x) for x in client_ids]


federated_train_data = make_federated_data(train_data, train_client_ids)
print("Number of client datasets: {l}".format(l=len(federated_train_data)))
print("First dataset: {d}".format(d=federated_train_data[0]))


## Tensorflow Federated model
# First we create a simple CNN model with Keras:


num_classes = data[0][1].shape[1]
img_height = data[0][0].shape[1]
img_width = data[0][0].shape[2]


def create_keras_model():
    model = Sequential(
        [
            layers.Conv2D(
                16,
                3,
                padding="same",
                activation="relu",
                input_shape=(img_height, img_width, 3),
            ),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    return model


# Note that we do not compile the model yet. The loss, metrics, and optimizers are introduced later.

# If you have a Keras model like the one we've just defined above, you can have TFF wrap it for you by invoking
# `tff.learning.from_keras_model`, passing the model and a sample data batch as
# arguments (`input_spec=example_dataset.element_spec`), as shown below.


# We _must_ create a new model here, and _not_ capture it from an external
# scope. TFF will call this within different graph contexts.
def model_fn():
    keras_model = create_keras_model()

    return tff.learning.from_keras_model(
        keras_model,
        input_spec=example_dataset.element_spec,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(),
            tf.keras.metrics.AUC(name="AUC"),
        ],
    )


# The model_fn is a no-arg function that returns a `tff.learning.Model`.


## Training the model on federated data

# Now that we have a model wrapped as `tff.learning.Model` for use with TFF, we
# can let TFF construct a **Federated Averaging** algorithm by invoking the helper
# function `tff.learning.build_federated_averaging_process`, as follows.
#
# One critical note on the Federated Averaging algorithm below, there are **2**
# optimizers: a _client_optimizer_ and a _server_optimizer_. The
# _client_optimizer_ is only used to compute local model updates on each client.
# The _server_optimizer_ applies the averaged update to the global model at the
# server. In particular, this means that the choice of optimizer and learning rate
# used may need to be different than the ones you have used to train the model on
# a standard i.i.d. dataset.


iterative_process = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.01),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.05),
)


# In this case, the two computations generated and packed into iterative_process implement Federated Averaging.

# Let's start with the `initialize` computation. As is the case for all federated
# computations, you can think of it as a function. The computation takes no
# arguments, and returns one result - the representation of the state of the
# Federated Averaging process on the server.


##Tensorflow Federated computations for initialization train and validation:
# In the following steps first we initialize the server and client models and then introduce federated train and validate isteps:

## Initialize
# initialization is done using the following step:
# state = iterative_process.initialize()


## Training
# The second of the pair of federated computations, `next`, represents a single
# round of Federated Averaging, which consists of pushing the server state
# (including the model parameters) to the clients, on-device training on their
# local data, collecting and averaging model updates, and producing a new updated
# model at the server.

# Conceptually, you can think of `next` as having a functional type signature that
# looks as follows.

# SERVER_STATE, FEDERATED_DATA -> SERVER_STATE, TRAINING_METRICS
# In particular, one should think about `next()` not as being a function that runs on a server,
# but rather being a declarative functional representation of the entire decentralized computation - some of the inputs are provided by the server (`SERVER_STATE`), but each participating device contributes its own local dataset.

# To run a single round of training and visualizing the results We can use the following line of code and federated data we've already generated above:


# state, metrics = iterative_process.next(state, federated_train_data)
# print('round  1, metrics={}'.format(metrics))


## Evaluation
# To perform evaluation on federated data, you can construct another *federated
# computation* designed for just this purpose, using the
# `tff.learning.build_federated_evaluation` function, and passing in your model
# constructor as an argument. Note that as the evaluation doesn't perform gradient descent, and there's no need to construct
# optimizers.

# For experimentation and research, when a centralized test dataset is available,
# the Federated Learning for Text Generation tutorial
# demonstrates another evaluation option: taking the trained weights from
# federated learning, applying them to a standard Keras model, and then simply
# calling `tf.keras.models.Model.evaluate()` on a centralized dataset.


# evaluation = tff.learning.build_federated_evaluation(model_fn)   # iterative_process = tff.learning.build_federated_averaging_process(model_fn,client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.01), server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.05))

# federated_test_data = make_federated_data(test_data, test_client_ids)

# val_metrics = evaluation(state.model, federated_test_data)


## Training and evaluation multiple rounds:

# Here we run the federated learning algorithm multiple rounds on training clients and evaluate the performance on test client.


state = iterative_process.initialize()
NUM_ROUNDS = 5

loss = list()
accuracy = list()
AUC = list()


val_loss = list()
val_accuracy = list()
val_AUC = list()


evaluation = tff.learning.build_federated_evaluation(model_fn)
federated_test_data = make_federated_data(test_data, test_client_ids)

for round_num in range(1, NUM_ROUNDS + 1):
    state, metrics = iterative_process.next(state, federated_train_data)
    val_metrics = evaluation(state.model, federated_test_data)

    my_loss = metrics["train"]["loss"]
    loss.append(metrics["train"]["loss"])

    my_acc = metrics["train"]["categorical_accuracy"]
    accuracy.append(metrics["train"]["categorical_accuracy"])

    my_AUC = metrics["train"]["AUC"]
    AUC.append(my_AUC)

    my_val_loss = val_metrics["loss"]
    val_loss.append(val_metrics["loss"])

    my_val_accuracy = val_metrics["categorical_accuracy"]
    val_accuracy.append(val_metrics["categorical_accuracy"])

    my_val_AUC = val_metrics["AUC"]
    val_AUC.append(my_val_AUC)

    print(
        f"round: {round_num:2d}, training_loss: {my_loss}, training_accuracy: {my_acc}, train_auc: {my_AUC}, test_loss: {my_val_loss}, test_accuracy: {my_val_accuracy},  test_auc: {my_val_AUC}"
    )


history_fed = {
    "loss": loss,
    "categorical_accuracy": accuracy,
    "AUC": AUC,
    "val_loss": val_loss,
    "val_categorical_accuracy": val_accuracy,
    "val_AUC": val_AUC,
}
