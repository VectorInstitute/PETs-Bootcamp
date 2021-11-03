# Vertical Federated Learning for the Caravan Insurance Challenge

## Introduction

The caravan insurance challenge is a binary classification problem where 86 features can be used to predict if a customer will buy caravan insurance.

We are simulating a situation where one third-party data provider can no longer share the raw data of a set of very important features to our model. Thus, we must fall back to vertical federated learning to continue to use these important features if we don't want the predictive performance to decrease too significantly.


## Day 1

Today, we conducted an exploratory data analysis of the dataset.

The distribution of the target is heavily skewed. 94% of the target label are 0.

The predictive performance (F1 score) of a random guess model that guesses 0 or 1 following the prior probability of the labels in the data is situated around 7%. Any model with an F1 score below the random guess model is a failure.

We explored several neural network (NN) architectures, using scikit-learn's multi-layer perceptron (MLP) wrapper. We converged towards an MLP with 5 hidden layers, each with 40 hidden units. This model resulted in an F1 score of 12% on the unseen test set. We are working towards a Pytorch implementation of this MLP that reproduces the scikit-learn MLP.

Rebalancing the target variable did not lead to models improve our state-of-the-art performance. E.g., we tried oversampling the minority class using SMOTE-Tomek links resulting in an uniformly distributed target distribution, but to no avail.

Next, we need to make a choice on which variables to partition. We decided to select MOSTYPE and MOSHOOFD (See definition here: https://www.kaggle.com/uciml/caravan-insurance-challenge) as the variables to be partitioned on the client side. MOSTYPE and MOSHOOFD are categorical variables with 40 and 5 unique values respectively with information on the customers. The choice to partition these two variables was based on several factors. First, specific values of these variables appeared as important in a study on feature selection (https://medium.com/swlh/feature-selection-to-kaggle-caravan-insurance-challenge-on-r-bede801d3a66). Second, in an experiment where we explicitly removed those two variables, the F1 score dropped to 5%, a significant reduction from the performance that included both variables. Thirdly, these two variables are the result of a thorough customer survey, e.g. containing information on income, living standards, religion, values, career. It is plausible that this type of information is not easily acquired by an insurance provider, but requires a third party vendor that specializes in surveys.

