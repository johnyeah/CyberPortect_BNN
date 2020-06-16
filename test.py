import pickle

import numpy as np
import pyro
import torch
from torch.autograd import Variable

from model.bnn_single_dynamic_policy import guide, get_processed_data, BNN, RESULTS_DIR


def predict(bnn, x, num_samples=10):
    with torch.no_grad():
        print('started prediction---')
        sampled_models = [guide(bnn, x, None) for _ in range(num_samples)]
        yhats = [model(x.float()).data for model in sampled_models]
        print(yhats)
        mean = torch.mean(torch.stack(yhats), 0)
        print('mean', mean)
        std = torch.std(torch.stack(yhats), 0)
        # return torch.stack(yhats), np.mean(std.numpy(), axis=1)
        return np.argmax(mean.numpy(), axis=1), np.argmax(std.numpy(), axis=1)


if __name__ == '__main__':
    # net = torch.load('/Users/mohamed/git/mik-zy/CyberPortect_BNN/model/bnn_single_dynamic_policy.pkl',
    #                 map_location="cpu")
    BNN
    model_name = "bnn"
    pyro.get_param_store().load(RESULTS_DIR + model_name + '_pyro.pkl')

    bnn = pickle.load(open(RESULTS_DIR + model_name + '_pickled.pkl', "rb"),
                      fix_imports=False)

    X, y = get_processed_data()
    X_test, y_test = Variable(torch.tensor(X[:1, :]), requires_grad=False), Variable(torch.tensor(y[:1]),
                                                                                     requires_grad=False)
    y_data_hat, std = predict(bnn, X_test)

    diff = np.abs((y_data_hat > 0.5).astype(int) - y_test.detach().numpy())

    print(diff, std)
