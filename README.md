# CS128H Rust Final Project

Rust.AI is a Rust library implementing basic neural networks for use in Rust. We are attempting to add all of the following features to our Library:

1) Feed Forward Neural Networks
2) Backpropagation
3) Different Types of Loss Functions
4) Customizable Layer Traits

We are additionally attempting to add some of the following features to our Library:

1) Integration between Layer Types (Expanded Network Configurations)
2) Different Training Routines
3) Hyperparameter Tuning
4) Support and Implementation of Physics Influenced Neural Networks (PINNs)

## Group

Our Group Name is Sigmas. Our Group Members are Kartikey Sharma (ks129) and Nihar Kalode (nkalode2).

## Semantics

We chose to work on this project because we are both CS + Physics students, and much of the intersection between CS and Physics is either in Quantum Computing or in PINNs (which won the Nobel Prize in Physics this year!). Our ultimate goal in this project is to build up to developing PINNs in Rust to implement in various tasks, such as Supervised and Semi-supervised Learning. 

## Timeline

We hope to implement the following properties of neural networks in our Library:

1) Feed Forward Neural Networks:
  We hope to implement the basic architecture of feed forward neural networks, specifically implementing input, hidden, and output layers. We then hope to implement activation functions and forward propogation to "link" layers to one another and create our network. As this will serve as the base for the rest of our library's features, we plan to develop this first.

2) Loss Functions:
   We hope to first define multiple loss functions (i.e. Cross Entropy, L2 & L1 Loss, etc) based on their mathematical properties. Then, we hope to compute the gradients of these loss functions for use in backpropagation. We plan to develop this second or third, as it doesn't rely on many of our other features.

3) Backpropagation:
   We plan to implement backpropagation immediately after implementing loss functions, as they rely heavily on loss functions. Our implementation of backpropogation will apply our loss functions to our neural networks in order to update neuron weights and biases.
   
4) Customizable Layer Traits:
   We hope to implement customizable layer traits that can allow integration between different layer types to create expanded, more complex network configurations. We hope to first implement various different layer types, such as dense and convolutional layers. We then hope to create a building pattern to create well-defined layers with different neuron traits. We plan to implement this after implementing the rest of our planned deliverables.

5) Integration between Layer Types (Expanded Network Configurations):
   We attempt to implement expanded network confiugrations by allowing different types of hidden layers to be weaved between one another, allowing for more complex neural networks to be made. We hope to develop a framework that can combine multiple layer types into a single architecture, before implementing dataflow logic for intra-layer connections.

6) Different Training Routines:
   We hope to implement basic training routines for our neural networks, such that our models can improve in accuracy through backpropagation. We hope to implement various training algorithms like Adam or RMSprop to train our neural networks.
7) Hyperparameter Tuning:
   We attempt to define hyperparameters, such as learning rate and number of epochs, that are relevant to improving performance of our neural networks. We hope to implement tuning algorithms such as GridSearch and RandomSearch to optimize hyperparameter combinations. 
8) Support for PINNs:
  We attempt to implement PINNs by integrating physical laws into some of our neural network models, creating loss functions and model architecture based off laws and properties in physics. (Note: I don't know much about PINNs right now so I can't really go into any detail on how we would go about implementing them, but hopefully we can get a PINN to derive some physics equation by the end of this project).

By Checkpoint I, we hope to be completed with Feed Forward Neural Networks and Loss Functions, and working on Backpropagation. By Checkpoint II, we hope to be completed with Backpropagation and Customizable Layer Traits and working on our non-essential deliverables. 

## Challenges
We expect to run out of time before we can implement everything. Time constraints are definitely our biggest concern. Understanding how neural networks train and behave is also a major concern of ours, as it will be necessary for us to learn the complex math behind neural network functionality. If we don't understand the math behind our neural network's training, loss, and optimization, we will have a much harder time debugging potential errors that arise in these steps. As long as we can understand the architecture of neural networks and how they work, I'm confident we will have a successful final project.

## References

This is kind of based off of Python Machine Learning Libraries, like SkLearn and Keras, although we really only take inspiration from them in that we create machine learning libraries with similar purposes to them.
