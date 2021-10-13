import numpy as np
import pandas as pd
import tensorflow as tf
import random
import math

from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer

def train(noise_multiplier, l2_norm_clip, batch_size, microbatches,
            x_train, y_train, x_valid, y_valid, x_test, y_test,
            dpsgd=True, learning_rate=0.1, epochs=120, verbose=1, seed=0):

    random.seed(seed)
    tf.random.set_seed(seed)

    if dpsgd and batch_size % microbatches != 0:
        raise ValueError('Number of microbatches should divide evenly batch_size')

    # Define a sequential Keras model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(40, input_dim = 21, activation='relu'),
        tf.keras.layers.Dense(60, activation='relu'),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    if dpsgd:
        optimizer = DPKerasSGDOptimizer(
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            num_microbatches=microbatches,
            learning_rate=learning_rate)
        # Compute vector of per-example loss rather than its mean over a minibatch.
        loss = tf.keras.losses.BinaryCrossentropy(reduction=tf.losses.Reduction.NONE)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        loss = tf.keras.losses.BinaryCrossentropy()

    # Compile model with Keras
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # Train model with Keras
    model.fit(x_train, y_train,
                epochs=epochs,
                validation_data=(x_valid, y_valid),
                batch_size=batch_size,
                verbose = verbose)

    score_train = model.evaluate(x_train, y_train, verbose=verbose)
    score_valid = model.evaluate(x_valid, y_valid, verbose=verbose)
    score_test = model.evaluate(x_test, y_test, verbose=verbose)

    weights = model.get_weights()

    # Compute the privacy budget expended.
    # // is integer division
    if dpsgd:
        training_size = x_train.shape[0]
        eps = compute_epsilon(epochs * training_size // batch_size,
                                training_size = training_size,
                                noise_multiplier=noise_multiplier,
                                batch_size=batch_size)
        print('For delta=0.00413223, the current epsilon is: %.2f' % eps)
    else:
        eps = 'non-private SGD'
        print('Trained with vanilla non-private SGD optimizer')

    return score_train, score_valid, score_test, eps, weights

def compute_epsilon(steps, training_size, noise_multiplier, batch_size):
  # Computes epsilon value for given hyperparameters.
  #
  # Parameters required:
  #   steps: Number of steps the optimizer takes over the training data
  #          steps = FLAGS.epochs * training_size// FLAGS.batch_size
  #
  #   Noise multiplier:
  #       the amount of noise sampled and added to gradients during training
  if noise_multiplier == 0.0:
    return float('inf')

  # Delta bounds the probability of our privacy guarantee not holding.
  # rule of thumb is to set delta to less than the inverse of the training data size
  # so here, I opted for it to equal to 1.1*training size
  training_delta = 1 / (training_size *1.1)

  #We need to define a list of orders, at which the Rényi divergence will be computed
  orders = np.linspace(1+math.log(1./training_delta)/10, 1+math.log(1./training_delta)/1, num=100)

  #Sampling ratio q is the probability of an individual training point being included in a minibatch
  sampling_probability = batch_size / training_size

  # compute Renyi Differential Privacy, a generalization of pure differential privacy
  # RDP is well suited to analyze DP guarantees provided by sampling followed by Gaussian noise addition,
  # which is how gradients are randomized in the TFP implementation of the DP-SGD optimizer.
  rdp = compute_rdp(q=sampling_probability,
                    noise_multiplier=noise_multiplier,
                    steps=steps,
                    orders=orders)

  return get_privacy_spent(orders, rdp, target_delta=training_delta)[0]
