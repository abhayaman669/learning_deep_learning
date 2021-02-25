# Deep Learning Fundamentals

## Contents

- What is Machine Learning?
- What is Deep Learning?

## What is Machine Learning?

According to **Arthur Samuel** machine learning is the field of study that gives computers the ability to learn without being explicitly programmed.
  
In machine learning we rather than wirting set of instruction to accomplish a particular task we train the machine.
  
For example, let's say we want to make a system that will check if a particular message is positive or negative. To accomplish this in **traditional programming** we will make an algorithm and give a list of positive and negative words and then check occurence of them in the message and then decide if the message is positive or negative. Whereas, in **machine learning** we provide a set of data to the a machine learning algorithm and let it learn to classify the message as positive or negative.

## What is Deep Learning?

Deep learning is the subfield of **machine learning** inspired by the structure and function of human brain neural network. This learning can occur in either in supervised or unsupervised form.

In **Supervised Learning** we have already the right answer. And our algorithm try to learn using that data and then try to predict the output for the new data. For example, we have given the data of house area with their price we can learn using this and then can predict price for a particular house with area X (let's say).

In **Unsupervised Learning** we have little or no idea about what our result should look like. In this case we use different way to derive structure or cluster. For example, you have data for 10,00,000 genes and you need to find a way to automatically group them.

## Artificial Neural Network (ANNs)

These are computing system inspired by the brains neural network. In this we have neurons which are connected to another neurons and can transmit a signal fron one to another and then the reciving neuron processes the signal and then pass is on to other neuron.

Neurons are organised in different layer different layer will perform different kinds of transformation.

signals travel from input layer to output layer and all the layers in between are known as hidded layers.

> TODO: Add image for a basic NN structure.

Here is code to create a simple NN using keras

```
from keras.models import Sequential
from keras.models import Dense, Activation

model = Sequrntial([
    Dense(32, input_shape=(10,), activation="relu"),
    Dense(2, activation="softmax"),
])
```

- First we imported **Sequential**, **Dense** and **Activation**.
- Here **Dense** is a type of layer.
- **input_shape** argument is telling what is shape of the input beight provided.
- **activation** defines the activation function used in the layer.
- The first argument in the Dense is the no. of neurons.

## Layers in ANN

There are different types of layers like:

- Dense
- Convolutional
- Pooling
- Recurrent
- Normalization

Different layer perform different transformations on their inputs and different layers are better suited for different tasks.

## Activation

- An **activation** function will follow the layer.
- it defines the output of the neuron.
- **sigmoid activation** function will change the output to in range from 0 to 1
- **ReLU** function will change the output to 0 if the output is negative else if will keep it same i.e. in python max(0, x) where x is an input from a neuron.