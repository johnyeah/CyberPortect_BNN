
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import DataLoader
from torch.autograd import Variable


import pyro
from pyro.distributions import Normal, Categorical, Bernoulli, Multinomial
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from sklearn.model_selection import train_test_split
import time

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import MinMaxScaler, Normalizer
import math





def data_prepare(PATH, dynamic=False):
    global features, obst_vel, label, all_features

    '''
    data preprocessing
    '''
    # select the needed data by columns index
    features = ['grip_pos_x', 'grip_pos_y', 'grip_pos_z',
                'grip_vel_x', 'grip_vel_y', 'grip_vel_z',
                'obst_pos_x_1', 'obst_pos_y_1', 'obst_pos_z_1',
                'goal_pos_x', 'goal_pos_y', 'goal_pos_z',
                'action_x', 'action_y', 'action_z']

    obst_vel = ['obst_vel_x_1', 'obst_vel_y_1', 'obst_vel_z_1']

    label = 'is_safe_action'

    if (dynamic == True):
        all_features = features + obst_vel + [label]
    else:
        all_features = features + [label]


    # read csv file
    origin_data = pd.read_csv(PATH)
    origin_data[label] = origin_data[label].astype(int)
    origin_data = origin_data.loc[:, all_features]
    return origin_data



def balance_data(data):
    # select safe_state & unsafe state out, then make their number be equal
    n_safe_state = data.loc[data[label] == 1]
    n_unsafe_state = data.loc[data[label] == 0]
    n_safe_state = n_safe_state.sample(n=len(n_unsafe_state))  # make the number of safe and unsafe be equal

    data = pd.concat([n_safe_state, n_unsafe_state], axis=0)  # concatenate safe samples and unsafe samples
    data = data.sample(frac=1)  # randomization

    return data




def normalizer(data):
    scaler = Normalizer().fit(data)
    scaled_data = scaler.transform(data)
    return scaled_data, scaler






class BNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output = self.fc1(x)
        output = F.relu(output)
        output = self.out(output)
        return output



log_softmax = nn.LogSoftmax(dim=1)
def model(x_data, y_data):
    fc1w_prior = Normal(loc=torch.zeros_like(net.fc1.weight), scale=torch.ones_like(net.fc1.weight))
    fc1b_prior = Normal(loc=torch.zeros_like(net.fc1.bias), scale=torch.ones_like(net.fc1.bias))

    outw_prior = Normal(loc=torch.zeros_like(net.out.weight), scale=torch.ones_like(net.out.weight))
    outb_prior = Normal(loc=torch.zeros_like(net.out.bias), scale=torch.ones_like(net.out.bias))

    priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior, 'out.weight': outw_prior, 'out.bias': outb_prior}
    # lift module parameters to random variables sampled from the priors
    lifted_module = pyro.random_module("module", net, priors)
    # sample a regressor (which also samples w and b)
    lifted_reg_model = lifted_module()

    lhat = log_softmax(lifted_reg_model(x_data))
    pyro.sample("obs", Categorical(logits=lhat), obs=y_data)


softplus = torch.nn.Softplus()
def guide(x_data, y_data):
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
    priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior, 'out.weight': outw_prior, 'out.bias': outb_prior}

    lifted_module = pyro.random_module("module", net, priors)

    return lifted_module()




def trainer(train_loader, num_iterations):
    all_losses = []
    for j in range(num_iterations):
        loss = 0
        for batch_id, data in enumerate(train_loader):
            batch_x, batch_y = data
            # calculate the loss and take a gradient step
            loss += svi.step(batch_x, batch_y)
        normalizer_train = len(train_loader.dataset)
        epoch_loss = loss / normalizer_train
        all_losses.append(epoch_loss)
        print("Epoch ", j, " Loss ", epoch_loss)









def tester(x, num_samples):
    sampled_models = [guide(None, None) for _ in range(num_samples)]
    yhats = [model(x).data for model in sampled_models]
    mean = torch.mean(torch.stack(yhats), 0)
    y_pred = np.argmax(mean.numpy(), axis=1)
    return y_pred








if __name__ == '__main__':
    '''
    prepare train data and test data
    '''

    # File PATH
    FILE_PATH_TRAIN_static = '191217_test_record_iter_fix_static1.csv'  # static obstacle csv file
    FILE_PATH_TRAIN_dynamic = 'test_record_iter_fix_static.csv'  # dynamic obstacle csv file

    preprocessed_data = data_prepare(FILE_PATH_TRAIN_static, dynamic=False)

    # balance data
    #preprocessed_data = balance_data(preprocessed_data)


    # prepare features and label: x, y
    preprocessed_data = preprocessed_data.iloc[0:100000]
    preprocessed_data = preprocessed_data.values
    x = preprocessed_data[:, 0:len(features) - 1]
    y = preprocessed_data[:, len(all_features) - 1]

    # normalize data
    #x, _ = normalizer(x)

    # splite data to train data and test data
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.01, random_state=42)

    x_train, y_train = torch.FloatTensor(x_train), torch.LongTensor(y_train)
    x_test, y_test = torch.FloatTensor(x_test), torch.LongTensor(y_test)

    '''
    x_train = torch.from_numpy(x_train).type(torch.FloatTensor)  # FloatTensor = 32-bit floating
    y_train = torch.from_numpy(y_train).type(torch.LongTensor)  # LongTensor = 64-bit integer

    x_test = torch.from_numpy(x_test).type(torch.FloatTensor)
    y_test = torch.from_numpy(y_test).type(torch.LongTensor)
    '''

    torch_train_dataset = Data.TensorDataset(x_train, y_train)
    torch_test_dataset = Data.TensorDataset(x_test, y_test)


    train_loader = DataLoader(
        dataset=torch_train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2,
    )

    test_loader = DataLoader(
        dataset=torch_test_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2,
    )

    # build network
    net = BNN(14, 20, 2)



    # optimizer
    optim = Adam({"lr": 0.01})
    svi = SVI(model, guide, optim, loss=Trace_ELBO())

    # train model
    num_iterations = 5
    trainer(train_loader, num_iterations)

    # save the trained model
    PATH = 'bnn_net_params.pkl'
    torch.save(net.state_dict(), PATH)


    # test model
    correct = 0
    total = 0
    num_samples = 10

    for j, data in enumerate(test_loader):
        batch_x,  batch_y = data
        predicted = tester(batch_x.view(-1, 14), num_samples)
        total += batch_y.size(0)
        correct += (predicted == batch_y.numpy()).sum().item()
    print("accuracy: %d %%" % (100 * correct / total))



