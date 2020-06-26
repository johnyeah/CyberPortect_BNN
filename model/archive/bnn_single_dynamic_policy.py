import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyro
import torch
import torch.nn as nn
from pyro.distributions import Normal, Bernoulli
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from torch.autograd import Variable

pyro.enable_validation(False)

pyro.get_param_store().clear()

torch.manual_seed(999)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(999)

'''
data preprocessing
'''
# read csv file
FILE_PATH_TRAIN = '/Users/mohamed/ownCloud/Exchange/Shared outside/1920 Cyberprotect/Data/200610/test_record_iter_dynamic.csv'
origin_data = pd.read_csv(FILE_PATH_TRAIN)

# select the needed data by columns index
input_features = ['grip_pos_x', 'grip_pos_y', 'grip_pos_z',
                  'grip_vel_x', 'grip_vel_y', 'grip_vel_z',
                  'obst_pos_x_1', 'obst_pos_y_1', 'obst_pos_z_1',
                  'obst_vel_x_1', 'obst_vel_y_1', 'obst_vel_z_1',
                  'goal_pos_x', 'goal_pos_y', 'goal_pos_z',
                  'action_x', 'action_y', 'action_z']

label = 'is_safe_action'
all_features = input_features + [label]

origin_data[label] = origin_data[label].astype(int)

# print(origin_data.loc[:, label].value_counts())


# select safe_state & unsafe state out, then make them equal
n_safe_state = origin_data.loc[origin_data[label] == True]
n_unsafe_state = origin_data.loc[origin_data[label] == False]
n_safe_state = n_safe_state.sample(n=len(n_unsafe_state))  # make the number of safe and unsafe be equal

prepocessed_data = pd.concat([n_safe_state, n_unsafe_state], axis=0)
prepocessed_data = prepocessed_data.sample(frac=1)  # select sampled safe data
prepocessed_data = prepocessed_data.loc[:, all_features]

print(prepocessed_data)



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
        self.loss_fun = nn.CrossEntropyLoss()
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

    with pyro.plate("map", len(X_train), subsample=data):
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


def train(iterations):
    global X_train, Y_train
    svi = SVI(model, guide, optim, loss=Trace_ELBO())
    losses = []
    N = len(X_train)
    start = time.clock()
    start1 = time.time()
    for j in range(iterations):
        epoch_loss = 0.0
        perm = torch.randperm(N)
        # shuffle data
        X_train = X_train[perm]
        Y_train = Y_train[perm]
        # get indices of each batch
        all_batches = get_batch_indices(N, 64)
        for ix, batch_start in enumerate(all_batches[:-1]):
            batch_end = all_batches[ix + 1]
            batch_X_train = X_train[batch_start: batch_end]
            batch_y_train = Y_train[batch_start: batch_end]
            epoch_loss += svi.step(net, batch_X_train, batch_y_train)
        losses.append(epoch_loss)

        print(j, "avg loss {}".format(epoch_loss / float(N)))

    '''
            if j % (iterations / 10) == 0:
                with torch.no_grad():
                    sampled_models = [guide(net, X_test, None) for _ in range(10)]
                    yhats = [model_(X_test).data for model_ in sampled_models]
                    yhats_test = torch.stack(yhats).squeeze()  # n_models * y_hat_test
                    yhats_test_mean = torch.mean((yhats_test), axis=0)
                    diff = np.abs((yhats_test_mean.detach().numpy() > 0.5).astype(int) - Y_test.numpy())
                    error = np.mean(diff)
                    std = np.std(diff)
                    print(j, "avg loss {}, error: {}, std: {}".format(epoch_loss / float(N), error, std))
    '''


    end1 = time.time()
    end = time.clock()
    train_time = end1 - start1
    execute_time = end - start
    print('training time:', train_time)
    print('execute CPU time:', execute_time)

    # sampled_models = plot_pyro_params()

    plt.figure(figsize=(12, 8))
    plt.plot(losses)
    plt.show()


def predict(x, num_samples=10):
    print('started prediction---')
    sampled_models = [guide(net, x, None) for _ in range(num_samples)]
    yhats = [model(x.float()).data for model in sampled_models]
    print(yhats)
    mean = torch.mean(torch.stack(yhats), 0)
    print('mean', mean)
    #    std = torch.std(torch.stack(yhats), 0)
    # return torch.stack(yhats), np.mean(std.numpy(), axis=1)
    return np.argmax(mean.numpy(), axis=1)



def Normalize_Data(data):
    scaler = Normalizer().fit(data)
    scaled_data = scaler.transform(data)
    return scaled_data, scaler


if __name__ == '__main__':
    softplus = torch.nn.Softplus()
    log_softmax = nn.LogSoftmax(dim=1)

    np.random.seed(0)
    torch.manual_seed(0)

    data = prepocessed_data.iloc[0:12000]
    data = data.values
    X, Y = data[:, :-1], data[:, -1]
    print(X.shape, Y.shape)

    X, _ = Normalize_Data(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.01, random_state=73)
    X_train, Y_train = Variable(torch.Tensor(X_train)), Variable(torch.Tensor(Y_train))
    X_test, Y_test = Variable(torch.Tensor(X_test)), Variable(torch.Tensor(Y_test))
    data = torch.cat((X_train, Y_train.reshape([-1, 1])), 1)

    n_input_dim, hidden_layers, n_output_dim = X_train.shape[1], [20, 20], 1

    file_path = "test/bnn_{}".format(str(hidden_layers).replace("[", "_").replace("]", ""))
    import os

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    net = BNN(n_input_dim=n_input_dim, n_hidden_layers=hidden_layers
              , n_output_dim=n_output_dim, activation_functions=[nn.Tanh()] * len(hidden_layers))

    optim = Adam({"lr": 0.01})
    accs = []
    stds = []
    train(iterations=50)
    torch.save(net.state_dict(), 'model_dynamic_policy.pkl')

    # prediction
    with torch.no_grad():
        sampled_models = [guide(net, X_test, None) for _ in range(10)]
        yhats = [model_(X_test).data for model_ in sampled_models]
        yhats_test = torch.stack(yhats).squeeze()  # n_models * y_hat_test
        yhats_test_mean = torch.mean((yhats_test), axis=0)  # test probability
        y_pred = (yhats_test_mean.detach().numpy() > 0.5).astype(int)  # test results

        diff = np.abs((yhats_test_mean.detach().numpy() > 0.5).astype(int) - Y_test.numpy())
        error = np.mean(diff)  # test error
        std = np.std(diff)  # test standard deviation
        print("error: {}, std: {}'".format(error, std))

    #    precision, recall, pr_thresh = precision_recall_curve(Y_test, yhats_test_mean)

    fpr, tpr, _ = roc_curve(Y_test, yhats_test_mean)
    roc_auc = auc(fpr, tpr)

    plt.stackplot(fpr, tpr, color='steelblue', alpha=0.5, edgecolor='black')
    plt.plot(fpr, tpr, color='black', lw=1)
    plt.text(0.1, 0.7, 'BNN (area = %0.2f)' % roc_auc)

    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate (fpr)')
    plt.ylabel('True Positive Rate (tpr)')
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.show()