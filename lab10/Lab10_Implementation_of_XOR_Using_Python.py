#!/usr/bin/env python
# coding: utf-8

# # Lab10: Implementation of XOR using Python

# Implementing logic gates using neural networks help understand the mathematical computation by which a neural network processes its inputs to arrive at a certain output. This neural network will deal with the XOR logic problem. An XOR (exclusive OR gate) is a digital logic gate that gives a true output only when both its inputs differ from each other. The truth table for an XOR gate is shown below:
# <img src="https://miro.medium.com/max/246/1*01idVj7sVw2ZnGZFapvW4A.png">

# The goal of the neural network is to classify the input patterns according to the above truth table. If the input patterns are plotted according to their outputs, it is seen that these points are not linearly separable. Hence the neural network has to be modeled to separate these input patterns using decision planes.
# <img src="https://miro.medium.com/max/640/1*_HLG8KlGJFZxtWoB8J1kFA.png" width="500" height="500">

# ## THE NEURAL NETWORK MODEL

# As mentioned before, the neural network needs to produce two different decision planes to linearly separate the input data based on the output patterns. This is achieved by using the concept of hidden layers. The neural network will consist of one input layer with two nodes (X1,X2); one hidden layer with two nodes (since two decision planes are needed); and one output layer with one node (Y). Hence, the neural network looks like this:
# <img src="https://miro.medium.com/max/543/1*qXt_iBvWods-FOvTldxYFw.png" width="500" height="500">

# ## THE SIGMOID NEURON

# To implement an XOR gate, I will be using a Sigmoid Neuron as nodes in the neural network. The characteristics of a Sigmoid Neuron are:
# 1. Can accept real values as input.
# 2. The value of the activation is equal to the weighted sum of its inputs
# i.e. ∑wi xi
# 3. The output of the sigmoid neuron is a function of the sigmoid function, which is also known as a logistic regression function. The sigmoid function is a continuous function which outputs values between 0 and 1:
# 
# <img src="https://miro.medium.com/max/640/1*R4twuYNUKXVzsvgOGkPZsA.png" width="500" height="500">

# ## THE LEARNING ALGORITHM

# The information of a neural network is stored in the interconnections between the neurons i.e. the weights. A neural network learns by updating its weights according to a learning algorithm that helps it converge to the expected output. The learning algorithm is a principled way of changing the weights and biases based on the loss function.
# 1. Initialize the weights and biases randomly.
# 2. Iterate over the data
#     i. Compute the predicted output using the sigmoid function
#     ii. Compute the loss using the square error loss function
#     iii. W(new) = W(old) — α ∆W
#     iv. B(new) = B(old) — α ∆B
# 3. Repeat until the error is minimal
# 
# This is a fairly simple learning algorithm consisting of only arithmetic operations to update the weights and biases. The algorithm can be divided into two parts: the forward pass and the backward pass also known as “backpropagation.”

# ## GRADIENT DESCENT

# The loss function of the sigmoid neuron is the squared error loss. If we plot the loss/error against the weights we get something like this:
# <img src="https://miro.medium.com/max/640/1*rNDygHX0Ds1In2mBE1ZC4g.png" width="500" height="500">
# 
# Our goal is to find the weight vector corresponding to the point where the error is minimum i.e. the minima of the error gradient. And here is where calculus comes into play.

# ## THE MATH BEHIND GRADIENT DESCENT

# Error can be simply written as the difference between the predicted outcome and the actual outcome. Mathematically:
# 
# <img src="https://miro.medium.com/max/139/1*bSE-d1xTHdPMc_woav7m-w.png" width="100" height="100">
# 
# where t is the targeted/expected output & y is the predicted output.
# 
# However, is it fair to assign different error values for the same amount of error? For example, the absolute difference between -1 and 0 & 1 and 0 is the same, however the above formula would sway things negatively for the outcome that predicted -1. To solve this problem, we use square error loss.(Note modulus is not used, as it makes it harder to differentiate). Further, this error is divided by 2, to make it easier to differentiate, as we’ll see in the following steps.
# 
# <img src="https://miro.medium.com/max/161/1*FawfyTR5ga85aFd8Jhbbjg.png" width="100" height="100">
# 
# Since, there may be many weights contributing to this error, we take the partial derivative, to find the minimum error, with respect to each weight at a time. The change in weights are different for the output layer weights ($W_{31}$ & $W_{32}$) and different for the hidden layer weights ($W_{11}$, $W_{12}$, $W_{21}$, $W_{22}$). Let the outer layer weights be $w_{o}$ while the hidden layer weights be $w_{h}$.
# 
# <img src="https://miro.medium.com/max/60/1*pZ8p9GnRNp7pI9_LtD755A.png" width="50" height="50">
# 
# We’ll first find ∆W for the outer layer weights. Since the outcome is a function of activation and further activation is a function of weights, by chain rule:
# 
# <img src="https://miro.medium.com/max/282/1*lQmnCwjEwo0MdVjvRswCHg.png" width="250" height="100">
# 
# On solving,
# 
# <img src="https://miro.medium.com/max/353/1*BRGgm_r1yfh0_Zwd6QMfiQ.png" width="250" height="100">
# 
# Note that for $X_{o}$ is nothing but the output from the hidden layer nodes. This output from the hidden layer node is again a function of the activation and correspondingly a function of weights. Hence, the chain rule expands for the hidden layer weights:
# 
# <img src="https://miro.medium.com/max/407/1*ytRh9SKcH5beWceMUNyLFw.png" width="300" height="150">
# 
# Which comes to,
# 
# <img src="https://miro.medium.com/max/550/1*l7rYNFhoaD5GDU_oNo3XwQ.png" width="500" height="250">
# 
# NOTE: $X_{o}$ can also be considered to be $Y_{h}$ i.e. the output from the hidden layer is the input to the output layer. $X_{h}$ is the input to the hidden layer, which are the actual input patterns from the truth table.
# 
# 

# In[1]:


# Import the necessary library
import numpy as np 
# np.random.seed(0)


# In[2]:


def sigmoid (x):
    return 1/(1 + np.exp(-x))


# In[3]:


def sigmoid_derivative(x):
    return x * (1 - x)


# In[4]:


# Set the Input datasets
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
# Set the expected output
expected_output = np.array([[0],[1],[1],[0]])


# In[23]:


epochs = 10000
lr = 0.3

inputLayerNeurons = 2
hiddenLayerNeurons = 2
outputLayerNeurons = 1


# In[24]:


# Random weights and bias initialization
hidden_weights = np.random.uniform(size=(inputLayerNeurons,hiddenLayerNeurons))
hidden_bias = np.random.uniform(size=(1,hiddenLayerNeurons))
output_weights = np.random.uniform(size=(hiddenLayerNeurons,outputLayerNeurons))
output_bias = np.random.uniform(size=(1,outputLayerNeurons))


# In[25]:


# Display the hidden_weights, hidden_bias, output_weights, and output_bias
print("Initial hidden weights: ",end='')
print(*hidden_weights)
print("Initial hidden biases: ",end='')
print(*hidden_bias)
print("Initial output weights: ",end='')
print(*output_weights)
print("Initial output biases: ",end='')
print(*output_bias)


# In[26]:


# Training algorithm

for _ in range(epochs):
    # Forward Propagation
    hidden_layer_activation = np.dot(inputs,hidden_weights)
    hidden_layer_activation += hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)

    output_layer_activation = np.dot(hidden_layer_output,output_weights)
    output_layer_activation += output_bias
    predicted_output = sigmoid(output_layer_activation)

    # Backpropagation
    error = expected_output - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Updating Weights and Biases
    output_weights += hidden_layer_output.T.dot(d_predicted_output) * lr
    output_bias += np.sum(d_predicted_output,axis=0,keepdims=True) * lr
    hidden_weights += inputs.T.dot(d_hidden_layer) * lr
    hidden_bias += np.sum(d_hidden_layer,axis=0,keepdims=True) * lr


# In[27]:


# Display the hidden_weights, hidden_bias, output_weights and output_bias after training
print("Final hidden weights: ",end='')
print(*hidden_weights)
print("Final hidden bias: ",end='')
print(*hidden_bias)
print("Final output weights: ",end='')
print(*output_weights)
print("Final output bias: ",end='')
print(*output_bias)


# In[28]:


# Finally, display the predicted output
print("\nOutput from neural network after 10,000 epochs: ",end='')
print(*predicted_output)


# ## BONUS QUESTION:

# Implement any other logic gate using the above steps

# #### AND Gate

# In[29]:


# Set the Input datasets
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
# Set the expected output
expected_output_AND = np.array([[0],[0],[0],[1]])


# In[30]:


# Training algorithm
for _ in range(epochs):
    # Forward Propagation
    hidden_layer_activation = np.dot(inputs,hidden_weights)
    hidden_layer_activation += hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)

    output_layer_activation = np.dot(hidden_layer_output,output_weights)
    output_layer_activation += output_bias
    predicted_output = sigmoid(output_layer_activation)

    # Backpropagation
    error = expected_output_AND - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Updating Weights and Biases
    output_weights = output_weights + hidden_layer_output.T.dot(d_predicted_output) * lr
    output_bias += np.sum(d_predicted_output,axis=0,keepdims=True) * lr
    hidden_weights = hidden_weights + inputs.T.dot(d_hidden_layer) * lr
    hidden_bias += np.sum(d_hidden_layer,axis=0,keepdims=True) * lr


# In[31]:


#Finally, display the predicted output for Bonus part (AND gate)
print("\nOutput from neural network after 10,000 epochs for AND gate: ",end='')
print(*predicted_output)


# #### NAND Gate

# In[32]:


# Set the Input datasets
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
# Set the expected output
expected_output = np.array([[1],[1],[1],[0]])


# In[33]:


for _ in range(epochs):
    # Forward Propagation
    hidden_layer_activation = np.dot(inputs,hidden_weights)
    hidden_layer_activation += hidden_bias
    hidden_layer_output = sigmoid(hidden_layer_activation)
    
    output_layer_activation = np.dot(hidden_layer_output,output_weights)
    output_layer_activation += output_bias
    predicted_output = sigmoid(output_layer_activation)

    # Backpropagation
    error = expected_output - predicted_output
    d_predicted_output =  error * sigmoid_derivative(predicted_output)

    error_hidden_layer = d_predicted_output.dot(output_weights.T)
    
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Updating Weights and Biases
    output_weights = output_weights + hidden_layer_output.T.dot(d_predicted_output) * lr
    output_bias += np.sum(d_predicted_output,axis=0,keepdims=True) * lr
    hidden_weights = hidden_weights + inputs.T.dot(d_hidden_layer) * lr
    hidden_bias += np.sum(d_hidden_layer,axis=0,keepdims=True) * lr


# In[34]:


# Finally, display the predicted output
print("\nOutput from neural network after 10,000 epochs: ",end='')
print(*predicted_output)

