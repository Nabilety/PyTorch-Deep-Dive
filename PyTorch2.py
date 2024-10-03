# - [Project one - predicting the fuel efficiency of a car](#Project-one----predicting-the-fuel-efficiency-of-a-car)
#   - [Working with feature columns](#Working-with-feature-columns)
#   - [Training a DNN regression model](#Training-a-DNN-regression-model)
# - [Project two - classifying MNIST handwritten digits](#Project-two----classifying-MNIST-handwritten-digits)


# ## Project one - predicting the fuel efficiency of a car
# Using Auto MPG dataset https://archive.ics.uci.edu/ml/datasets/auto+mpg

# ### Working with feature columns
#
#

import pandas as pd

# Load data and apply preprocessing steps, including dropping incomplete rows, partitioning the dataset into training
# and test datasets, as well as standardizing the continuous features.
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

df = pd.read_csv(url, names=column_names,
                 na_values = "?", comment='\t',
                 sep=" ", skipinitialspace=True)

df.tail()

#print(df.isna().sum())

# Drop the NA rows (in this case they resided in horsepower column)
df = df.dropna()
df = df.reset_index(drop=True) # reset index so the empty/dropped rows are not included in the indexing
#print(df.tail())


# train/test splits
import sklearn
import sklearn.model_selection
df_train, df_test = sklearn.model_selection.train_test_split(df, train_size=0.8, random_state=1)
train_stats = df_train.describe().transpose()
print(train_stats)

numeric_column_names = [
    'Cylinders', 'Displacement',
    'Horsepower', 'Weight',
    'Acceleration'
]

df_train_norm, df_test_norm = df_train.copy(), df_test.copy()
for col_name in numeric_column_names:
    mean = train_stats.loc[col_name, 'mean']
    std = train_stats.loc[col_name, 'std']
    df_train_norm.loc[:, col_name] = (df_train_norm.loc[:, col_name] - mean)/std
    df_test_norm.loc[:, col_name] = (df_test_norm.loc[:, col_name] - mean)/std

print(df_train_norm.tail())

# model year (ModelYear) information into buckets to simplify the learning task for the model we will train
import torch
import torch.nn as nn
# three cut-off value boundaries [73, 76, 79] randomly chosen
boundaries = torch.tensor([73, 76, 79])
v = torch.tensor(df_train_norm['Model Year'].values)
df_train_norm['Model Year Bucketed'] = torch.bucketize(v, boundaries, right=True)

v = torch.tensor(df_test_norm['Model Year'].values)
df_test_norm['Model Year Bucketed'] = torch.bucketize(v, boundaries, right=True)

numeric_column_names.append('Model Year Bucketed') # add bucketized feature column to Python list numeric_column_names

# define list of unordered categorical features, Origin.
# PyTorch has two ways with categorical feature:
# 1. use an embedding layer via nn.Embedding
# 2. use one-hot-encoded vectors (aka indicator).
# In the encoding approach, i.e., index 0 will be encoded as [1, 0, 0]. Index 1 as [0, 1, 0] and so on.
# On the other hand, embedding layer maps each index to a vector of random numbers of type float.
# (think of embedding layer as a more efficient implementation of a one-hot encoding multiplied with a trainable matrix)

# when the number of categories is large, using the embedding layer with fewer dimensions
# than the number of categories can improve the performance

# We will use the one-hot-encoding approach on the categorical feature to convert it into the dense format
from torch.nn.functional import one_hot
total_origin = len(set(df_train_norm['Origin']))

origin_encoded = one_hot(torch.from_numpy(df_train_norm['Origin'].values) % total_origin)
x_train_numberic = torch.tensor(df_train_norm[numeric_column_names].values)
x_train = torch.cat([x_train_numberic, origin_encoded], 1).float()

origin_encoded = one_hot(torch.from_numpy(df_test_norm['Origin'].values) % total_origin)
x_test_numberic = torch.tensor(df_test_norm[numeric_column_names].values)
x_test = torch.cat([x_test_numberic, origin_encoded], 1).float()

# after encoding the categorical feature into a 3D-dense feature, we concatenated it with the numeric features
# we proccessed in the previous step

# Finally we create the label tensors from the ground truth MPG values
y_train = torch.tensor(df_train_norm['MPG'].values).float()
y_test = torch.tensor(df_test_norm['MPG'].values).float()


# Training a DNN regression model
from torch.utils.data import DataLoader, TensorDataset

# Create data loader using batch size of 8 for the train data
train_ds = TensorDataset(x_train, y_train)
batch_size = 8
torch.manual_seed(1)
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

# build model with two fully connected layers where one has 8 hidden units and another has 4
hidden_units = [8, 4]
input_size = x_train.shape[1]

all_layers = []
for hidden_unit in hidden_units:
    layer = nn.Linear(input_size, hidden_unit)
    all_layers.append(layer)
    all_layers.append(nn.ReLU())
    input_size = hidden_unit

all_layers.append(nn.Linear(hidden_units[-1], 1))
model = nn.Sequential(*all_layers)
print(model)

# Define MSE loss function for regression and use stochastic gradient descent for optimization
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Now we will train the model for 200 epochs and display the train loss for every 20 epochs
torch.manual_seed(1)
num_epochs = 200
log_epochs = 20
for epoch in range(num_epochs):
    loss_hist_train = 0
    for x_batch, y_batch in train_dl:
        pred = model(x_batch)[:, 0]
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_hist_train += loss.item()
    if epoch % log_epochs==0:
        print(f'Epoch {epoch}   Loss {loss_hist_train/len(train_dl):.4f}')


# After 200 epochs, the train loss was around 5. We can now evaluate the regression performance of the trained
# model on the test dataset. To test the target values on new data points, we can feed their features to the model
with torch.no_grad():
    pred = model(x_test.float())[:, 0]
    loss = loss_fn(pred, y_test)
    print(f'Test MSE: {loss.item():.4f}')
    print(f'Test MAE: {nn.L1Loss()(pred, y_test).item():.4f}')

# The MSE on the test set is 9.6 and the mean absolute error is 2.1
# After this regression project, we will now work on a classification project

# ## Project two - classifying MNIST hand-written digits