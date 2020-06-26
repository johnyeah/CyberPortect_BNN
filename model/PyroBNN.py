import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pyro
import torch
import torch.nn as nn
from pyro.distributions import Normal, Bernoulli
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from torch.autograd import Variable

from data.pre_processing import get_processed_data
from model import DIR

pyro.enable_validation(False)

pyro.get_param_store().clear()

torch.manual_seed(999)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(999)
np.random.seed(999)


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
            activation_functions = [nn.Tanh()] * ((len(n_hidden_layers) - 2) + 1)

        # activation_functions+= [nn.Softmax(dim=1)]
        layer = nn.Linear(n_input_dim, n_hidden_layers[0])
        self.layers.append(layer)
        self.fc.append(nn.Linear(n_input_dim, n_hidden_layers[0]))
        self.fc.append(activation_functions[0])
        # todo creates two tanh behond each other if n_hidden is 1 layer
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


class PyroBNN:

    def __init__(self, training_data_sample_size=10, predict_sample_size=10, normalizer=None, **network_args):
        self.network = BNN(**network_args)
        self.predict_sample_size = predict_sample_size
        self.training_data_sample_size = training_data_sample_size
        self.sample_modules = None
        self.normalizer = normalizer

    def model(self, x_data, y_data):
        fc_network = self.network
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

        with pyro.plate("map", self.training_data_sample_size,
                        subsample=x_data):  # sampling should be done from all data, not only the batch!
            # with pyro.plate("map", self.training_data_sample_size, subsample=data):
            # with pyro.plate("map", len(X_train), subsample=x_data):
            # x_data = data[:, :-1]
            # y_data = data[:, -1]
            # run the regressor forward conditioned on inputs
            prediction_mean = lifted_reg_model(x_data).squeeze()
            pyro.sample("obs", Bernoulli(prediction_mean),
                        obs=y_data.squeeze())

    def guide(self, x_data, y_data):
        """
        Approximation of the posterior P(w|x_data), --> likelihood p(y_data|w, x_data)
        :param fc_network:
        :param x_data:
        :param y_data:
        :return:
        """
        # x_data = data[:, :-1]
        fc_network = self.network
        # create weight distribution parameters priors
        priors = {}
        for i, layer in enumerate(fc_network.fc):
            if not hasattr(layer, 'weight'):
                continue
            # print("guide: ",i,layer)
            # print('guide_shapes',layer.weight.shape, layer.bias.shape)

            fciw_mu = Variable(torch.randn_like(layer.weight).type(torch.float64), requires_grad=True)
            fcib_mu = Variable(torch.randn_like(layer.bias).type(torch.float64), requires_grad=True)
            fciw_sigma = Variable(0.1 * torch.randn_like(layer.weight).type(torch.float64), requires_grad=True)
            fcib_sigma = Variable(0.1 * torch.randn_like(layer.bias).type(torch.float64), requires_grad=True)

            fciw_mu_param = pyro.param("guide.{}.w_mu".format(str(i)), fciw_mu)
            fcib_mu_param = pyro.param("guide.{}.b_mu".format(str(i)), fcib_mu)
            fciw_sigma_param = nn.Softplus()(pyro.param("guide.{}.w_sigma".format(str(i)), fciw_sigma))
            fcib_sigma_param = nn.Softplus()(pyro.param("guide.{}.b_sigma".format(str(i)), fcib_sigma))

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

    def train(self, X, Y, iterations, normalize=False, batch_size=64, early_stop=False):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.01, random_state=73)
        X_train, Y_train, X_test, Y_test = Variable(torch.tensor(X_train)), Variable(torch.tensor(Y_train)), Variable(
            torch.tensor(X_test)), Variable(torch.tensor(Y_test))

        if normalize:
            if self.normalizer is None:
                pass
            else:
                pass
            X_ = self.normalizer.fit(X_train)
        else:
            X_ = X_train
        Y_ = Y_train

        optim = Adam({"lr": 0.01})
        # todo try different guide: guide = AutoDelta(model), AutoContinuous, AutoMultivariateNormal, AutoDiagonalNormal, AutoGuideList (supports multiple guides in parallel)
        # check http://docs.pyro.ai/en/0.2.1-release/contrib.autoguide.html#pyro.contrib.autoguide.AutoGuide
        svi = SVI(self.model, self.guide, optim, loss=Trace_ELBO())

        elbo_losses = []
        train_losses = []
        test_losses = []
        N = len(X_)
        start = time.clock()
        start1 = time.time()
        for j in range(iterations):
            epoch_loss = 0.0
            perm = torch.randperm(N)
            # shuffle data
            X_shuffle = X_[perm]
            Y_shuffle = Y_[perm]
            # get indices of each batch
            all_batches = PyroBNN.get_batch_indices(N, batch_size)
            for ix, batch_start in enumerate(all_batches[:-1]):
                batch_end = all_batches[ix + 1]
                batch_X_train = X_shuffle[batch_start: batch_end]
                batch_y_train = Y_shuffle[batch_start: batch_end]
                epoch_loss += svi.step(batch_X_train, batch_y_train)
            with torch.no_grad():
                self.sample_modules = [self.guide(None, None) for _ in range(self.predict_sample_size)]

            elbo_losses.append(epoch_loss)
            train_loss = PyroBNN.calc_loss(self, X_train, Y_train)
            test_loss = PyroBNN.calc_loss(self, X_test, Y_test)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            print(j, "avg loss {}, {}, {}".format(epoch_loss / float(N), train_loss, test_loss))

            if early_stop and len(test_losses) > 10:  # execute at least 10 epochs
                test_loss_rate = (test_losses[-1] - test_losses[-2])

                if test_loss_rate >= 0.04:
                    print("Training terminated after {} epochs and test_loss diff {}".format(j, test_loss_rate))
                    break

        end1 = time.time()
        end = time.clock()
        train_time = end1 - start1
        execute_time = end - start
        print('training time:', train_time)
        print('execute CPU time:', execute_time)

        # sampled_models = plot_pyro_params()

        return elbo_losses, train_losses, test_losses

    def save_model(self, model_path):
        # Saving the NN
        pickle.dump(self, open(model_path + '_pickled.pkl', "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        # Saving pyro
        pyro.get_param_store().save(model_path + "_pyro.pkl")

    @staticmethod
    def load_model(model_path):
        bnn = pickle.load(open(model_path + '_pickled.pkl', "rb"),
                          fix_imports=False)

        # 2. Import pyro params
        pyro.get_param_store().load(model_path + '_pyro.pkl')
        pyro.module('module', bnn.network, update_module_params=True)

        return bnn

    @staticmethod
    def get_batch_indices(N, batch_size):
        """
        :param N: total number of samples
        :param batch_size:
        :return: array of batch indices

        """
        all_batches = np.arange(0, N, batch_size)
        if all_batches[-1] != N:
            all_batches = list(all_batches) + [N]
        return all_batches

    @staticmethod
    def calc_loss(pyro_net, X, Y):
        # prediction
        with torch.no_grad():
            yhat, mean, std = pyro_net.predict(X, normalize=False)

            diff = np.abs(yhat - Y.numpy())
            error = np.mean(diff)  # test error
            std = np.std(diff)  #

        return error

    def predict(self, x, sample_size=None, normalize=False):
        with torch.no_grad():
            if normalize:
                x_norm = self.normalizer.fit(x)
            else:
                x_norm = x
                if self.sample_modules is None:
                    raise Exception("The BNN is not trained yet! Please train first using nn.train(*args)")
                if sample_size is None:
                    sampled_nns = self.sample_modules
                else:
                    if sample_size < self.predict_sample_size:
                        sampled_nns = self.sample_modules[:sample_size]
                    else:
                        if sample_size > self.predict_sample_size:
                            # add new samples to self.sample_modules
                            self.sample_modules.extend(
                                [self.guide(None, None) for _ in range(sample_size - self.predict_sample_size)])
                            self.predict_sample_size = sample_size
                            print("-" * 10, "predict_sample_size changed to {}".format(self.predict_sample_size))
                        else:
                            pass
                        sampled_nns = self.sample_modules
                # print('started prediction---')
                yhats = torch.stack([model(x_norm).data for model in sampled_nns]).squeeze()
                mean = torch.mean(yhats, axis=0)  # test probability
                std = torch.std(yhats, axis=0)  # test probability
                yhat_final = (mean.detach().numpy() > 0.5).astype(int)
                return yhat_final, mean, std


def statistics(elbo_losses, train_loss, test_loss, pyro_bnn, X, Y, show=True, path=None):
    plt.figure(figsize=(10, 4))
    axs = [
        plt.subplot(131, frameon=False),
        plt.subplot(132, frameon=False),
        plt.subplot(133, frameon=False),
    ]

    if elbo_losses is not None:
        axs[0].plot(elbo_losses)
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Error')
    if train_loss is not None:
        axs[1].plot(train_loss, label="train")
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Error')
        axs[1].plot(test_loss, label="test")
        axs[1].set_ylim((0, 1))

    if X is not None:
        y_hat, _, _ = pyro_bnn.predict(X)
        fpr, tpr, _ = roc_curve(Y, y_hat)

        roc_auc = auc(fpr, tpr)

        axs[2].stackplot(fpr, tpr, color='steelblue', alpha=0.5, edgecolor='black')
        axs[2].plot(fpr, tpr, color='black', lw=1)
        axs[2].text(0.1, 0.7, 'BNN (area = %0.2f)' % roc_auc)
        axs[2].set_xlabel('False Positive Rate (fpr)')
        axs[2].set_ylabel('True Positive Rate (tpr)')
        axs[2].set_xlim((0, 1))
        axs[2].set_ylim((0, 1))
        axs[2].plot([0, 1], [0, 1], color='red', linestyle='--')

    if path:
        plt.savefig(path + ".png")
    if show:
        plt.show()


def parameter_search():
    # Processing data
    X, Y, X_normalizer = get_processed_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.01, random_state=73)
    X_train, Y_train, X_test, Y_test = Variable(torch.tensor(X_train)), Variable(torch.tensor(Y_train)), Variable(
        torch.tensor(X_test)), Variable(torch.tensor(Y_test))
    data = torch.cat((X_train, Y_train.reshape([-1, 1])), 1)
    RESULTS_DIR = DIR + "/results/training_sample_size/"
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    train = True
    model_name = "bnn_20_20"
    for network_size in [[64], [128], [256] * 1, [256, 128], [128, 32]]:
        # for sample_size in [int(1e4), int(1e5), int(1e6)]:
        model_path = RESULTS_DIR + model_name + "_{}_{}".format('ss_xdata', "-".join(str(x) for x in network_size))
        # -----------------------------------------------------------------------
        #                               Train
        # -----------------------------------------------------------------------
        n_input_dim, hidden_layers, n_output_dim = X.shape[1], network_size, 1
        iterations = 50
        batch_size = 32
        sample_size = int(1e5)
        pyro.clear_param_store()
        pyro_bnn = PyroBNN(training_data_sample_size=sample_size, predict_sample_size=10, normalizer=X_normalizer,
                           n_input_dim=n_input_dim,
                           n_hidden_layers=hidden_layers
                           , n_output_dim=n_output_dim, activation_functions=[nn.Tanh()] * len(hidden_layers))
        elbo_losses, train_loss, test_loss = pyro_bnn.train(iterations=iterations, X=X, Y=Y,
                                                            batch_size=batch_size)
        pyro_bnn.save_model(model_path=model_path)
        statistics(elbo_losses, train_loss, test_loss, pyro_bnn, X_test, Y_test, path=model_path, show=False)


if __name__ == '__main__':
    # Processing data
    X, Y, X_normalizer = get_processed_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.01, random_state=73)
    X_train, Y_train, X_test, Y_test = Variable(torch.tensor(X_train)), Variable(torch.tensor(Y_train)), Variable(
        torch.tensor(X_test)), Variable(torch.tensor(Y_test))
    data = torch.cat((X_train, Y_train.reshape([-1, 1])), 1)
    RESULTS_DIR = DIR + "/results/training_for_policy/"
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    train = True
    model_name = "bnn_final"
    # for sample_size in [int(1e4), int(1e5), int(1e6)]:
    model_path = RESULTS_DIR + model_name
    # -----------------------------------------------------------------------
    #                               Train
    # -----------------------------------------------------------------------
    if train:
        iterations = 100
        batch_size = 32
        network_size = [20, 20]
        sample_size = int(1e5)
        early_stop = True
        n_input_dim, hidden_layers, n_output_dim = X.shape[1], network_size, 1
        pyro.clear_param_store()
        pyro_bnn = PyroBNN(training_data_sample_size=sample_size, predict_sample_size=10, normalizer=X_normalizer,
                           n_input_dim=n_input_dim,
                           n_hidden_layers=hidden_layers
                           , n_output_dim=n_output_dim, activation_functions=[nn.Tanh()] * len(hidden_layers))
        elbo_losses, train_loss, test_loss = pyro_bnn.train(iterations=iterations, X=X, Y=Y,
                                                            batch_size=batch_size, early_stop=early_stop)
        pyro_bnn.save_model(model_path=model_path)
        statistics(elbo_losses, train_loss, test_loss, pyro_bnn, X_test, Y_test, path=model_path, show=False)

    # -----------------------------------------------------------------------
    #                        Test and Plot results
    # -----------------------------------------------------------------------
    # pyro.clear_param_store()
    # bnn_loaded = PyroBNN.load_model(model_path)
