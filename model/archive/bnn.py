
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.nn import init

from matplotlib import colors
from torch.utils.data import DataLoader
from torch.autograd import Variable

import pyro
from pyro.distributions import Normal, Categorical
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import math

pyro.set_rng_seed(42)



'''
data preprocessing
'''
# read csv file
FILE_PATH_TRAIN = '191217_test_record_iter_fix_static1.csv'
origin_data = pd.read_csv(FILE_PATH_TRAIN)


# select the needed data by columns index
input_features = ['grip_pos_x', 'grip_pos_y', 'grip_pos_z',
                  'grip_vel_x', 'grip_vel_y', 'grip_vel_z',
                  'obst_pos_x_1', 'obst_pos_y_1', 'obst_pos_z_1',
                  'goal_pos_x', 'goal_pos_y', 'goal_pos_z',
                  'action_x', 'action_y', 'action_z']

label = 'is_safe_action'
all_features = input_features + [label]

origin_data[label] = origin_data[label].astype(int)


# select safe_state & unsafe state out, then make them equal
n_safe_state = origin_data.loc[origin_data[label] == True]
n_unsafe_state = origin_data.loc[origin_data[label] == False]
n_safe_state = n_safe_state.sample(n=len(n_unsafe_state))  # make the number of safe and unsafe be equal

prepocessed_data = pd.concat([n_safe_state, n_unsafe_state], axis=0)
prepocessed_data = prepocessed_data.sample(frac=1)  # selct sampled safe data
prepocessed_data = prepocessed_data.loc[:, all_features]


print(prepocessed_data)



# Normalizer
def normalize_data(data):
    scaler = MinMaxScaler()
    data_minmax_scaled = scaler.fit_transform(data)
    return data_minmax_scaled, scaler



'''
train dataset, test dataset
'''

# train dataset
train = prepocessed_data.iloc[0:100000]

train_x = train.loc[:, input_features]
train_y = train.loc[:, label]


train_x, scaler_x = normalize_data(train_x)

# transform dataframe to torch tensor
train_x = torch.tensor(train_x).float()
train_y = torch.tensor(train_y.values)


# create train loader
train_dataset = Data.TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset,
                          batch_size=128,
                          shuffle=True,
                          num_workers=2)

# test dataset
test = prepocessed_data.iloc[0:100000]

test_x = test.loc[:, input_features]
test_y = test.loc[:, label]


test_x = scaler_x.transform(test_x)
test_x = torch.tensor(test_x).float()
test_y = torch.tensor(test_y.values)

test_dataset = Data.TensorDataset(test_x, test_y)
test_loader = DataLoader(test_dataset,
                         batch_size=128,
                         shuffle=True,
                         num_workers=2)





class Net(nn.Module):
    def __init__(self, n_inputs, n_hidden_neurons, n_outputs):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_inputs, n_hidden_neurons)
        self.out = nn.Linear(n_hidden_neurons, n_outputs)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.out(x)
        return x

net = Net(n_inputs=15, n_hidden_neurons=512, n_outputs=2)
print(net)



'model'
log_softmax = nn.LogSoftmax(dim=1)
def model(x_data, y_data):

    fc1w_prior = Normal(loc=torch.zeros_like(net.fc1.weight), scale=torch.ones_like(net.fc1.weight))
    fc1b_prior = Normal(loc=torch.zeros_like(net.fc1.bias), scale=torch.ones_like(net.fc1.bias))

    outw_prior = Normal(loc=torch.zeros_like(net.out.weight), scale=torch.ones_like(net.out.weight))
    outb_prior = Normal(loc=torch.zeros_like(net.out.bias), scale=torch.ones_like(net.out.bias))

    priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior,
              'out.weight': outw_prior, 'out.bias': outb_prior}

    # lift module parameters to random variables sampled from the priors
    lifted_module = pyro.random_module("module", net, priors)

    # sample a regressor (which also samples w and b)
    lifted_reg_model = lifted_module()
    lhat = log_softmax(lifted_reg_model(x_data))
    pyro.sample("obs", Categorical(logits=lhat), obs=y_data)


'guide'

softplus = torch.nn.Softplus()
def guide(x_data, y_data):
    # approximation to true posterior P(w|D)
    # First layer weight distribution priors
    fc1w_mu = torch.randn_like(net.fc1.weight)
    fc1w_sigma = torch.randn_like(net.fc1.weight)
    fc1w_mu_param = pyro.param("fc1w_mu", fc1w_mu)
    fc1w_sigma_param = softplus(pyro.param("fc1w_sigma", fc1w_sigma))
    fc1w_prior = Normal(loc=fc1w_mu_param, scale=fc1w_sigma_param)

    # First layer bias distribution priors
    fc1b_mu = torch.randn_like(net.fc1.bias)
    fc1b_sigma = torch.randn_like(net.fc1.bias)
    fc1b_mu_param = pyro.param("fc1b_mu", fc1b_mu)
    fc1b_sigma_param = softplus(pyro.param("fc1b_sigma", fc1b_sigma))
    fc1b_prior = Normal(loc=fc1b_mu_param, scale=fc1b_sigma_param)


    # Output layer weight distribution priors
    outw_mu = torch.randn_like(net.out.weight)
    outw_sigma = torch.randn_like(net.out.weight)
    outw_mu_param = pyro.param("outw_mu", outw_mu)
    outw_sigma_param = softplus(pyro.param("outw_sigma", outw_sigma))
    outw_prior = Normal(loc=outw_mu_param, scale=outw_sigma_param).independent(1)


    # Output layer bias distribution priors
    outb_mu = torch.randn_like(net.out.bias)
    outb_sigma = torch.randn_like(net.out.bias)
    outb_mu_param = pyro.param("outb_mu", outb_mu)
    outb_sigma_param = softplus(pyro.param("outb_sigma", outb_sigma))
    outb_prior = Normal(loc=outb_mu_param, scale=outb_sigma_param)


    priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior,
              'out.weight': outw_prior, 'out.bias': outb_prior}

    lifted_module = pyro.random_module("module", net, priors)

    return lifted_module()


'optimizer'
optim = Adam({"lr": 0.001})
svi = SVI(model, guide, optim, loss=Trace_ELBO())


'Trainer'
def model_training(train_loader):
    num_iterations = 10
    losses = []

    for j in range(num_iterations):
        loss = 0
        for batch_id, (batch_x, batch_y) in enumerate(train_loader):
            #            net.zero_grad()
            # calculate the loss and take a gradient step
            # loss = svi.step(batch_x, batch_y)
            loss += svi.step(batch_x.view(-1, 15), batch_y)
        #            loss.backward()
        #            optim.step()

        normalizer_train = len(train_loader.dataset)
        total_epoch_loss_train = loss / normalizer_train
        losses.append(total_epoch_loss_train)

        print("Epoch ", j, " Loss ", total_epoch_loss_train)

    plt.figure()
    plt.plot(np.arange(0, num_iterations), losses, label='loss ')
    plt.xlabel('EPOCH')
    plt.ylabel('Loss')  # ;plt.legend(loc='best')
    plt.show()


"""

model_training(train_loader)
torch.save(net.state_dict(), 'model_param')
"""


'predicter'
def predict(x):
    n_nn_samples = 100
    sampled_models = [guide(None, None) for _ in range(n_nn_samples)]

    yhats = [model(x).data for model in sampled_models]
    mean = torch.mean(torch.stack(yhats), 0)
    return np.argmax(mean.numpy(), axis=1)



model_training(train_loader)

print('Prediction when network is forced to predict')
correct = 0
total = 0
for j, data in enumerate(test_loader):
    data_x, data_y = data
    predicted = predict(data_x.view(-1, 15))
    total += data_y.size(0)
    correct += (predicted == data_y.numpy()).sum().item()
print("accuracy: %d %%" % (100 * correct / total))




