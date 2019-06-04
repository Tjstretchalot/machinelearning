# Machine Learning Collection

This package contains a collection of measures for and runners to use for machine learning tasks. It is centrally focused on understanding how machines learn rather than achieving optimal performance, and thus some of these network architectures are intentionally gimped.

This package has a "shared" folder and then folders for each dataset that it works with. The files in the top-level directory are used for executing directly on a cluster. Within each folder intended for a dateset there is a "runners" folder which contains the various trials that were executed.

## Shared

The models directory contains the generic network architectures that are used throughout this package. The evolvable_models folder takes these models and parameterizes them explicitly in terms of architectural features (ie. which nonlinearity they use). The measures folder contains various metrics as well as animations/figures that can be generated from a network.

The top-level shared folder contains utility files, such as additional tweenings for animations, additional nonlinearities for investigation, additional criterion, a glue class for ZeroMQ to multiprocessing queues, an events class, file tools, and more.

## Gaussian Spheres

The gaussian spheres are not just gaussian blobs in the typical sense. Points in a gaussian cluster have are drawn from a sphere whose *radius* is selected from a gaussian distribution. This is a technical detail that makes rejection sampling much faster.

## MNIST

This is the typical MNIST dataset. To run this folder you need to download the MNIST data and place it in data/mnist; this is intended to eventually be refactored to automatically download the data if it is missing

## CIFAR

This is for the CIFAR-10 dataset. This folder leverages the PyTorch auto-downloading feature and will thus be loaded automatically.

## Typing

This is a sequence-sequence autoencoder that optionally includes a delay, although it isn't used anywhere. This is based on the encoder-decoder framework and uses online training (though not particularly effectively). This module also has a bit of work on trying to train more complex models by training a simpler model and then copying the weights over to a bigger network. It improves the accuracy somewhat but is not effective compared to attention networks

## OR Reinforce

This contains various reinforcement learning models for the optimax rogue environment. They are mostly based around the Q-learning concept with online learning (i.e. intrasession learning).