
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
from pyro.distributions import Normal, Categorical, Bernoulli, Multinomial
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from sklearn.model_selection import train_test_split
import time

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import MinMaxScaler, Normalizer
import math

pyro.enable_validation(False)

pyro.get_param_store().clear()

torch.manual_seed(999)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(999)






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



'''
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
'''


class BNN(nn.Module):
    """
    Feedforward network for Binary classification
    """

    def __init__(self, n_input_dim, n_hidden_layers: list,
                 n_output_dim, activation_functions: list = None,
                 adam_params: dict = None):
        super().__init__()
        self.fc = []
        self.layers = []  # container for hidden layers. Useful later for #model() and #guide
        if activation_functions is None:
            activation_functions = [nn.Tanh(inplace=True)] * ((len(n_hidden_layers) - 2) + 1)

        # activation_functions+= [nn.Softmax(dim=1)]
        layer = nn.Linear(n_input_dim, n_hidden_layers[0])
        self.layers.append(layer)
        self.fc.append(nn.Linear(n_input_dim, n_hidden_layers[0]))
        self.fc.append(activation_functions[0])

        for i, _ in enumerate(n_hidden_layers[:-1]):
            layer = nn.Linear(n_hidden_layers[i], n_hidden_layers[i + 1])
            self.layers.append(layer)
            self.fc.append(layer)
            if i < len(n_hidden_layers) - 2:
                self.fc.append(activation_functions[i])
        layer = nn.Linear(n_hidden_layers[-1], n_output_dim)
        self.fc.append(activation_functions[-1])
        self.layers.append(layer)
        self.fc.append(layer)

        self.fc.append(torch.nn.Sigmoid())

        self.model = nn.Sequential(
            *self.fc
        )
        print("Created model", self.model)
        self.loss_fun = nn.BCELoss()
        if adam_params is None:
            adam_params = {"lr": 0.01, "betas": (0.90, 0.999)}

        self.optimizer = Adam(adam_params)
        # torch.optim.Adam(adam_params, lr=learning_rate)

    def __repr__(self):
        return str([(i, fc.weight, fc.bias) for i, fc in enumerate(self.fc) if isinstance(fc, nn.Linear)])

    def forward(self, x):
        y_hat = self.model(x)
        # print('y_hat',y_hat)
        return y_hat






def model(fc_network: BNN, x_data, y_data):
    # create prior for weight and bias per layer, p(w) [q(z) // p(w)]
    priors = {}
    for i, layer in enumerate(fc_network.fc):
        if not hasattr(layer, 'weight'):
            continue
        # print("model: ",i,layer)
        priors["model.{}.weight".format(str(i))] = \
            Normal(Variable(torch.zeros_like(layer.weight)), Variable(torch.ones_like(layer.weight)))
        priors["model.{}.bias".format(str(i))] = \
            Normal(Variable(torch.zeros_like(layer.bias)), Variable(torch.ones_like(layer.bias)))
    # print('model: ',priors)
    # exit(0)
    # print('model_shapes',layer.weight.shape, layer.bias.shape)

    # lift module parameters to random variables sampled from the priors --> Sample a NN from the priors!
    lifted_module = pyro.random_module("module", fc_network, priors)
    # sample a regressor (which also samples w and b)
    lifted_reg_model = lifted_module()

    with pyro.plate("map", len(x_train), subsample=data):
        x_data = data[:, :-1]
        y_data = data[:, -1]
        # run the regressor forward conditioned on inputs
        prediction_mean = lifted_reg_model(x_data).squeeze()
        pyro.sample("obs", Bernoulli(prediction_mean),
                    obs=y_data.squeeze())




def guide(fc_network: BNN, x_data, y_data):
    """
    Approximation of the posterior P(w|x_data), --> likelihood p(y_data|w, x_data)
    :param fc_network:
    :param x_data:
    :param y_data:
    :return:
    """
    # create weight distribution parameters priors
    priors = {}
    for i, layer in enumerate(fc_network.fc):
        if not hasattr(layer, 'weight'):
            continue
        # print("guide: ",i,layer)
        # print('guide_shapes',layer.weight.shape, layer.bias.shape)

        fciw_mu = Variable(torch.randn_like(layer.weight).type_as(x_data), requires_grad=True)
        fcib_mu = Variable(torch.randn_like(layer.bias).type_as(x_data), requires_grad=True)
        fciw_sigma = Variable(0.1 * torch.randn_like(layer.weight).type_as(x_data), requires_grad=True)
        fcib_sigma = Variable(0.1 * torch.randn_like(layer.bias).type_as(x_data), requires_grad=True)

        fciw_mu_param = pyro.param("guide.{}.w_mu".format(str(i)), fciw_mu)
        fcib_mu_param = pyro.param("guide.{}.b_mu".format(str(i)), fcib_mu)
        fciw_sigma_param = softplus(pyro.param("guide.{}.w_sigma".format(str(i)), fciw_sigma))
        fcib_sigma_param = softplus(pyro.param("guide.{}.b_sigma".format(str(i)), fcib_sigma))

        fciw_prior = Normal(fciw_mu_param, fciw_sigma_param)
        fcib_prior = Normal(fcib_mu_param, fcib_sigma_param)
        # TODO prior should have the same weight as in for name, _ in fc_network.named_parameters(),
        #  according to https://forum.pyro.ai/t/how-does-pyro-random-module-match-priors-with-regressionmodel-parameters/528/7
        priors['model.{}.weight'.format(str(i))] = fciw_prior
        priors['model.{}.bias'.format(str(i))] = fcib_prior
    #    lifted_module = pyro.module("module", fc_network, priors)
    # print('guide: ',priors)
    # for name, _ in fc_network.named_parameters():
    #    print(name)
    # exit(0)
    lifted_module = pyro.random_module("module", fc_network, priors)
    random_model = lifted_module()
    # print('lifted_module', random_model)
    return random_model





# get array of batch indices
def get_batch_indices(N, batch_size):
    all_batches = np.arange(0, N, batch_size)
    if all_batches[-1] != N:
        all_batches = list(all_batches) + [N]
    return all_batches


def train(num_epoch):
    print('start to train')
    global x_train, y_train

    svi = SVI(model, guide, optim, loss=Trace_ELBO())
    all_losses = []
    all_error = []
    all_std = []
    N = len(x_train)

    for j in range(num_epoch):
        running_loss = 0.0
        perm = torch.randperm(N)
        # shuffle data
        x_train = x_train[perm]
        y_train = y_train[perm]
        # get indices of each batch
        all_batches = get_batch_indices(N, 64)
        for ix, batch_start in enumerate(all_batches[:-1]):
            batch_end = all_batches[ix + 1]
            batch_x_train = x_train[batch_start: batch_end]
            batch_y_train = y_train[batch_start: batch_end]
            running_loss += svi.step(net, batch_x_train, batch_y_train)
            epoch_loss = running_loss / len(x_train)
        # loss in each epoch
        all_losses.append(epoch_loss)


        with torch.no_grad():
             sampled_models = [guide(net, x_test, None) for _ in range(10)]
             yhats = [model_(x_test.float()).data for model_ in sampled_models]
             yhats_test = torch.stack(yhats).squeeze()  # n_models * y_hat_test
             yhats_train_mean = torch.mean(yhats_test, axis=0)

             epoch_diff = np.abs((yhats_train_mean.detach().numpy() > 0.5).astype(int) - y_test.numpy())

             # prediction error in each epoch
             epoch_error = np.mean(epoch_diff)
             all_error.append(epoch_error)

             # standard deviation in each epoch
             epoch_std = np.std(epoch_diff)
             all_std.append(epoch_std)

             print('Epoch', j, 'Loss', epoch_loss, 'error', epoch_error, 'Std', epoch_std)
    # plot
    plt.subplot(311)
    plt.plot(np.arange(0, num_epoch), all_losses, label='loss')
    plt.xlabel('epoch')
    plt.ylabel('Loss')

    plt.subplot(312)
    plt.plot(np.arange(0, num_epoch), all_error, label='error')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')

    plt.subplot(313)
    plt.plot(np.arange(0, num_epoch), all_std, label='std deviation')
    plt.xlabel('epoch')
    plt.ylabel('std deviation')
    plt.show()




def tester(x, num_samples):
    sampled_models = [guide(net, x, None) for _ in range(num_samples)]
    yhats = [model(x.float()).data for model in sampled_models]
    mean = torch.mean(torch.stack(yhats), 0)
    #std = torch.std(torch.stack(yhats), 0)
    #np.mean(mean.numpy(), axis=1), np.mean(std.numpy(), axis=1)
    y_pred = (mean.detach().numpy() > 0.5).astype(int)
    return mean.detach().numpy(), y_pred





if __name__ == '__main__':

    softplus = torch.nn.Softplus()
    log_softmax = nn.LogSoftmax(dim=1)

    np.random.seed(0)
    torch.manual_seed(0)

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
    x = preprocessed_data[:, : -1]
    y = preprocessed_data[:, -1]

    print(x.shape, y.shape)

    # normalize data
    #x, _ = normalizer(x)

    # splite data to train data and test data
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.01, random_state=73)

    torch_test_dataset = Data.TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
    test_loader = DataLoader(
        dataset=torch_test_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=2,
    )

    x_train, y_train = Variable(torch.Tensor(x_train)), Variable(torch.Tensor(y_train))
    x_test, y_test = Variable(torch.Tensor(x_test)), Variable(torch.Tensor(y_test))

    data = torch.cat((x_train, y_train.reshape([-1, 1])), 1)

    n_input_dim, hidden_layers, n_output_dim = x_train.shape[1], [20, 20], 1

    net = BNN(n_input_dim=n_input_dim, n_hidden_layers=hidden_layers,
              n_output_dim=n_output_dim, activation_functions=[nn.Tanh()] * len(hidden_layers))


    #set optimization function
    optim = Adam({"lr": 0.01})

    # train model
    num_epoch = 20
    train(num_epoch)

    # save model parameters
    Model_PATH = 'bnn_net_params.pkl'
    torch.save(net.state_dict(), Model_PATH)

    print('start to predict')
    # test model
    num_samples = 10
    test_correct = 0
    total = 0
    for batch_id, data in enumerate(test_loader):
        test_x, test_y = data
        y_predict_proba, y_predict = tester(test_x, num_samples)
        total += test_y.size(0)
        test_correct += np.sum(y_predict.flatten() == test_y.numpy())

    print('Acurracy:', 100 * test_correct / total)
    np.save('/home/mae/Python_Code/CyberPortect_BNN/CyberPortect_BNN/y_bnn_test_static', test_y)
    np.save('/home/mae/Python_Code/CyberPortect_BNN/CyberPortect_BNN/y_bnn_predict_static', y_predict_proba)


    











