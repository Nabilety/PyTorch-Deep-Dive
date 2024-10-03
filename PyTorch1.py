import torch
import torch.nn as nn
# ## The key features of PyTorch
#
# ## PyTorch's computation graphs
#
# ### Understanding computation graphs
#
#





# ### Creating a graph in PyTorch
#
#
# a, b and c are scalars, and we define them as PyTorch tensors
def compute_z(a, b, c):
    r1 = torch.sub(a, b)
    r2 = torch.mul(r1, 2)
    z = torch.add(r2, c)
    return z

# Call function with tensor objects as function arguments
# rank 0 (scalar), rank 1 and rank 2 inputs:
print('Scalar Inputs:', compute_z(torch.tensor(1), torch.tensor(2), torch.tensor(3)))
print('Rank 1 Inputs:', compute_z(torch.tensor([1]), torch.tensor([2]), torch.tensor([3])))
print('Rank 2 Inputs:', compute_z(torch.tensor([[1]]), torch.tensor([[2]]), torch.tensor([[3]])))

# ## PyTorch Tensor objects for storing and updating model parameters
a = torch.tensor(3.14, requires_grad=True)
print(a)
b = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(b)

# default of the requires_grad is False. Can be set to True by running requires_grad_()
w = torch.tensor([1.0, 2.0, 3.0])
print(w.requires_grad)
w.requires_grad_()
print(w.requires_grad)


# Random initialization scheme for NN model parameters with random weights (to break symmetry during backpropagation)
# PyTorch can generate random numbers based on a variety of probability distributions
# Create tensor with Glorot initialization (classic random initialization scheme)
# First we create empty tensor and an operator called init as an object of class GlorotNormal
# Then fill this tensor with values according to Glorot initialization by calling xavier_normal_() method
# Initialize tensor of shape 2x3:
torch.manual_seed(1)
w = torch.empty(2, 3)
nn.init.xavier_normal_(w)
print(w)


# define two tensor objects inside nn.Module class
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = torch.empty(2, 3, requires_grad=True)
        nn.init.xavier_normal_(self.w1)
        self.w2 = torch.empty(1, 2, requires_grad=True)
        nn.init.xavier_normal_(self.w2)
# the wo tensor can be used as weights whose gradients will be computed via automatic differentiation

# ## Computing gradients via automatic differentiation and GradientTape
#

# ### Computing the gradients of the loss with respect to trainable variables

# Comptue z = wx + b with loss defined as the squared loss between target y and prediction z
# Define model parameters w and b as tensors with requires_gradient attr set to True) and input x and y as defualt tensors
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.5, requires_grad=True)
x = torch.tensor([1.4])
y = torch.tensor([2.1])
z = torch.add(torch.mul(w, x), b)
loss = (y-z).pow(2).sum()
loss.backward() # use backward method on the loss tensor to compute partial derivative of w and b
print('dL/dw : ', w.grad)
print('dL/db : ', b.grad)



# verifying the computed gradient dL/dw = 2x(wx + b - y)
print(2 * x * ((w * x + b) - y))


# ## Simplifying implementations of common architectures via the torch.nn module
#
#

# ### Implementing models based on nn.Sequential
# layers stored inside the model are connected in a cascaded way
# Following example build a model with two densely (fully) connected layers

model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(),
                      nn.Linear(16, 32), nn.ReLU())
print(model)
# Output of first densely connected layer is used as input to the first ReLU layer
# Output of first ReLU layer becomes input for second densely connected layer
# Finally output of second densely connected layer is used as input to the second ReLU layer

# Further configurations can be made on the layers i.e., applying different activation functions, initializers, regularization methods to parameters
# #### Configuring layers
#
#  * Initializers `nn.init`: https://pytorch.org/docs/stable/nn.init.html
#  * L1 Regularizers `nn.L1Loss`: https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html#torch.nn.L1Loss
#  * L2 Regularizers `weight_decay`: https://pytorch.org/docs/stable/optim.html
#  * Activations: https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
#


# We will configure first densely connected layer by specifying initial value distribution for the weight.
# Then configure the second densely layer by computing L1 penalty term for the weight matrix.
nn.init.xavier_uniform_(model[0].weight) # here we initialized the weight of the first linear layer with Xavier initialization
l1_weight = 0.01
l1_penalty = l1_weight * model[2].weight.abs().sum() # computed L1 norm of the weight of the second linear layer

# Furthermore, we could also specify the type of optimizer and the loss function for training
# #### Compiling a model
#
#  * Optimizers `torch.optim`:  https://pytorch.org/docs/stable/optim.html#algorithms
#  * Loss Functins `tf.keras.losses`: https://pytorch.org/docs/stable/nn.html#loss-functions

# Choosing a Loss function - SGD and Adam are most widely used methods
# Choice will depend on task; i.e. MSE loss for regression problem, cross-entropy loss for classification tasks
# Also techniques for model evaluation  such as hyperparameter tuning can be used - appropriate metrics; precision, recall, accuracy, AUC

# We will use SGD optimizer and cross-entropy loss for binary classification
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)



# ## Solving an XOR classification problem
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
np.random.seed(1)
torch.manual_seed(1)
x = np.random.uniform(low=-1, high=1, size=(200, 2))
y = np.ones(len(x))
y[x[:, 0] * x[:, 1]<0] = 0 # ground truth label with for with two features x_0 an x_1
n_train = 100
x_train = torch.tensor(x[:n_train, :], dtype=torch.float32)
y_train = torch.tensor(y[:n_train], dtype=torch.float32)
x_valid = torch.tensor(x[n_train:, :], dtype=torch.float32)
y_valid = torch.tensor(y[n_train:], dtype=torch.float32)

fig = plt.figure(figsize=(6, 6))
plt.plot(x[y==0, 0],
         x[y==0, 1], 'o', alpha=0.75, markersize=10)
plt.plot(x[y==1, 0],
         x[y==1, 1], '<', alpha=0.75, markersize=10)
plt.xlabel(r'$x_1$', size=15)
plt.ylabel(r'$x_2$', size=15)

plt.show()

# We now need to decide what architecture we should choose for this task and dataset.
# General rule of thumb, the more layers we have, the more neurons we have in each layers,
# the larger the capacity of the model will be. Here the model capacity can be thought of as a
# measure of how readily the model can approximate complex functions.
# While having more parameters means the network can fit more complex functions, larger models are usually harder to train

# In practice it's a good idea to start with a simple model as a baseline, i.e. single-layer NN like LR

# Create dataloader that uses a batch size of 2 for training data:
train_ds = TensorDataset(x_train, y_train)
batch_size = 2
torch.manual_seed(1)
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

# Define model
model = nn.Sequential(
    nn.Linear(2, 1),
    nn.Sigmoid()
)

print(model)

# Initialize cross-entropy loss function for binary classification and SGD optimizer
loss_fn = nn.BCELoss() # expects the output to have the same shape as the target
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Train model for 200 epochs and record a history of training epochs
torch.manual_seed(1)
num_epochs = 200
def train(model, num_epochs, train_dl, x_valid, y_valid):
    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs
    for epoch in range(num_epochs):
        for x_batch, y_batch in train_dl:
            pred = model(x_batch)[:, 0]
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_hist_train[epoch] += loss.item()
            is_correct = ((pred>=0.5).float() == y_batch).float()
            accuracy_hist_train[epoch] += is_correct.mean()

        loss_hist_train[epoch] /= n_train/batch_size
        accuracy_hist_train[epoch] /= n_train/batch_size

        pred = model(x_valid)[:, 0]
        loss = loss_fn(pred, y_valid)
        loss_hist_valid[epoch] = loss.item()
        is_correct = ((pred>=0.5).float() == y_valid).float()
        accuracy_hist_valid[epoch] += is_correct.mean()
    return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid

history = train(model, num_epochs, train_dl, x_valid, y_valid)

fig = plt.figure(figsize=(16, 4))
ax = fig.add_subplot(1, 2, 1)
plt.plot(history[0], lw=4)
plt.plot(history[1], lw=4)
plt.legend(['Train loss', 'Validation loss'], fontsize=15)
ax.set_xlabel('Epochs', size=15)

ax = fig.add_subplot(1, 2, 2)
plt.plot(history[2], lw=4)
plt.plot(history[3], lw=4)
plt.legend(['Train acc.', 'Validation acc.'], fontsize=15)
ax.set_xlabel('Epochs', size=15)
plt.show()

# a simple model with no hidden layer can only derive a linear decision boundary, so it's unable to solve
# the XOR problem. Thus, the loss terms for both training and validation dataset are very high, similarly for accuracy low

# to derive a non-linear decision boundary, we can add one or more hidden layers connected vi nonlinear activation functions
# A feedforward NN with a single hidden layer and relatively large number of hidden units can approximate arbitrary
# continuous functions relatively well. So one approach for tackling XOR problem better is to add a hideen layer
# and compare different numbers of hidden units until we observe a satisfactory result on the validation dataset.

# Adding more hidden units would correspond go increasing the width of a layer
# Alternatively, adding more hidden layers will make the model more deeper. The advantage with in making a network
# deeper rather than wider is that a fewer parameters are required to achieve a comparable model capacity.
# However downside of deep (versus wide) models is that deep models are prone to vanishing and exploding gradients
# So they make it harder to train.

# As an exercise we try to add one, two, three and four hidden layers, each with four hidden units.
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)

loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.015)
print(model)
history = train(model, num_epochs, train_dl, x_valid, y_valid)
fig = plt.figure(figsize=(16, 4))
ax = fig.add_subplot(1, 2, 1)
plt.plot(history[0], lw=4)
plt.plot(history[1], lw=4)
plt.legend(['Train loss', 'Validation loss'], fontsize=15)
ax.set_xlabel('Epochs', size=15)

ax = fig.add_subplot(1, 2, 2)
plt.plot(history[2], lw=4)
plt.plot(history[3], lw=4)
plt.legend(['Train acc.', 'Validation acc.'], fontsize=15)
ax.set_xlabel('Epochs', size=15)
plt.show()

# Now the model is able to derive a nonlinear decision boundary for this data, and the model reaches 100% accuracy
# on the trianing dataset, while the validation dataset's accuracy is 95 indicating the model is slightly overfitting.

# ## Making model building more flexible with nn.Module
#
# Alternative way of building complex models with multiple input/output or intermediate branches
# is by subclassing nn.Module. This way we create a new class derived from nn.Module and define the method, __init__()
# as a constructor. the forward() method to specify the forward pass. In the constructor, we define the layers as
# attributes of the class so that they can be accessed via the self reference attribute. Then in the forward()
# method we sepcify how these layers are to be used in the forward pass of the NN: The code for defining a new class
# that implements the previous model is as follows:
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        l1 = nn.Linear(2, 4)
        a1 = nn.ReLU()
        l2 = nn.Linear(4, 4)
        a2 = nn.ReLU()
        l3 = nn.Linear(4, 1)
        a3 = nn.Sigmoid()
        l = [l1, a1, l2, a2, l3, a3]
        self.module_list = nn.ModuleList(l) # put all layyers in the ModuleList object (list of objects composed of nn.Module items

    def forward(self, x):
        for f in self.module_list:
            x = f(x)
        return x

    def predict(self, x): # to compute decision boundary of mode
        x = torch.tensor(x, dtype=torch.float32)
        pred = self.forward(x)[:, 0]
        return (pred>=0.5).float() #  will return either predicated class (0 or 1) for a sample

model = MyModule()
print(model)

# train model
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.015)
history = train(model, num_epochs, train_dl, x_valid, y_valid)

# besides train history, we will used the mlxtend library to visualize the validation data and the decision boundary

from mlxtend.plotting import plot_decision_regions
fig = plt.figure(figsize=(16, 4))
ax = fig.add_subplot(1, 3, 1)
plt.plot(history[0], lw=4)
plt.plot(history[1], lw=4)
plt.legend(['Train loss', 'Validation loss'], fontsize=15)
ax.set_xlabel('Epochs', size=15)

ax = fig.add_subplot(1, 3, 2)
plt.plot(history[2], lw=4)
plt.plot(history[3], lw=4)
plt.legend(['Train acc.', 'Validation acc.'], fontsize=15)
ax.set_xlabel('Epochs', size=15)

ax = fig.add_subplot(1, 3, 3)
plot_decision_regions(X=x_valid.numpy(),
                      y=y_valid.numpy().astype(np.int64),
                      clf=model)
ax.set_xlabel(r'$x_1$', size=15)
ax.xaxis.set_label_coords(1, -0.025)
ax.set_ylabel(r'$x_2$', size=15)
ax.yaxis.set_label_coords(-0.025, 1)
plt.show()



# ## Writing custom layers in PyTorch
# (in cases where we want to define a new layer not already supported by PyTorch)
# we can define a new class derived from the nn.Module class. Especially useful for designing new layers or customizing existing ones

# Simple example, of linear layer computing w(x + ε) + b, where ε refers to a random variable as a noise variable.
# First we define a new class as a subclass of nn.Module
# We need to define both the constructor __init__() and  forward() method.
# In the constructor we define variables and other required tensor for customized layer.
# We can create variables and initialize them in the constructor if the input_size is given to the constructor.
# Alternatively, we delay the variable initialization (i.e. if we don't know the exact input shape upfront), and
# delegate it to another method for late variable creation
# Concrete example, a new layer called NoisyLinear, which implements the computation w(x + ε) + b, as described.

class NoisyLinear(nn.Module):
    # add the argument noise_stddev to specify standard deviation for the distribution of ε (sample from Gaussian distribution)
    def __init__(self, input_size, output_size, noise_stddev=0.1):
        super().__init__()
        w = torch.Tensor(input_size, output_size)
        self.w = nn.Parameter(w)  # nn.Parameter is a Tensor that's a module parameter.
        nn.init.xavier_uniform(self.w) # weight initialization using xavier
        b = torch.Tensor(output_size).fill_(0)
        self.b = nn.Parameter(b)
        self.noise_stddev = noise_stddev

    # use training=False argument, used to distinguish whether the layer is used during training or only for prediction
    # (also knowns as inference) or evaluation.
    def forward(self, x, training=False):
        # random vector ε only generated and added to the input during training only and not used for inference or evaluation
        if training:
            noise = torch.normal(0.0, self.noise_stddev, x.shape)
            x_new = torch.add(x, noise)
        else:
            x_new = x
        return torch.add(torch.mm(x_new, self.w), self.b)

# Test a simple example
## 1. testing:

torch.manual_seed(1)

#define new instance of this layer
noisy_layer = NoisyLinear(4, 2)

x = torch.zeros((1, 4))
#execute it on an input tensor, and then call the layer three times on the same input tensor
# notice the output for the first two calls differ because NoisyLinear layer added random noise to the input tensor
# the third call outputs [0, 0] as we didn't add noise by specifying training=False
print(noisy_layer(x, training=True))

print(noisy_layer(x, training=True))

print(noisy_layer(x, training=False))

# 2. Now we create new model similar to previous one for solving XOR classification task
# This time we will use our NoisyLinear layer as the first hidden layer of the multilayer perceptron:
class MyNoisyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = NoisyLinear(2, 4, 0.07)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(4, 4)
        self.a2 = nn.ReLU()
        self.l3 = nn.Linear(4, 1)
        self.a3 = nn.Sigmoid()

    def forward(self, x, training=False):
        x = self.l1(x, training)
        x = self.a1(x)
        x = self.l2(x)
        x = self.a2(x)
        x = self.l3(x)
        x = self.a3(x)
        return x

    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        pred = self.forward(x)[:, 0]
        return (pred >= 0.5).float()
torch.manual_seed(1)
model = MyNoisyModule()
print(model)


# Similarly we train the model as we did before. At this time, to compute the prediction on the training batch
# we use pred = model(x_batch, True)[:, 0] instead of pred = model(x_batch)[:, 0]

loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.015)

torch.manual_seed(1)

loss_hist_train = [0] * num_epochs
accuracy_hist_train = [0] * num_epochs
loss_hist_valid = [0] * num_epochs
accuracy_hist_valid = [0] * num_epochs
for epoch in range(num_epochs):
    for x_batch, y_batch in train_dl:
        pred = model(x_batch, True)[:, 0]
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_hist_train[epoch] += loss.item()
        is_correct = ((pred >= 0.5).float() == y_batch).float()
        accuracy_hist_train[epoch] += is_correct.mean()

    loss_hist_train[epoch] /= n_train / batch_size
    accuracy_hist_train[epoch] /= n_train / batch_size

    pred = model(x_valid)[:, 0]
    loss = loss_fn(pred, y_valid)
    loss_hist_valid[epoch] = loss.item()
    is_correct = ((pred >= 0.5).float() == y_valid).float()
    accuracy_hist_valid[epoch] += is_correct.mean()

# Plot the losses, accuracies and decision boundary after model is trained

fig = plt.figure(figsize=(16, 4))
ax = fig.add_subplot(1, 3, 1)
plt.plot(loss_hist_train, lw=4)
plt.plot(loss_hist_valid, lw=4)
plt.legend(['Train loss', 'Validation loss'], fontsize=15)
ax.set_xlabel('Epochs', size=15)

ax = fig.add_subplot(1, 3, 2)
plt.plot(accuracy_hist_train, lw=4)
plt.plot(accuracy_hist_valid, lw=4)
plt.legend(['Train acc.', 'Validation acc.'], fontsize=15)
ax.set_xlabel('Epochs', size=15)

ax = fig.add_subplot(1, 3, 3)
plot_decision_regions(X=x_valid.numpy(),
                      y=y_valid.numpy().astype(np.int64),
                      clf=model)
ax.set_xlabel(r'$x_1$', size=15)
ax.xaxis.set_label_coords(1, -0.025)
ax.set_ylabel(r'$x_2$', size=15)
ax.yaxis.set_label_coords(-0.025, 1)
plt.show()
