
# load packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import init


from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import MinMaxScaler, Normalizer


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


# build net model
class Net(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_input, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, input):
        out = self.hidden1(input)
        out = F.relu(out)
        out = self.hidden2(out)
        out = torch.sigmoid(out)
        out = self.predict(out)

        return out



def trainer(input_data, num_epoch):
    all_losses = []
    all_acc = []

    for epoch in range(num_epoch):
         running_loss = 0.0
         running_correct = 0


         for batch_id, data in enumerate(input_data):
             train_x, train_y = data

             #zero the parameter gradients
             optimizer.zero_grad()

             #forward + backward + optimize
             out = net(train_x)
             loss = loss_func(out, train_y)
             loss.backward()
             optimizer.step()

             # print statistics
             running_loss += out.shape[0] * loss.item()
             # running_loss += loss.item()

             _, preds = torch.max(out, 1)
             running_correct += torch.sum(preds == train_y).item()

         epoch_loss = running_loss / len(train_loader.dataset)
         all_losses.append(epoch_loss)

         # accuracy in each epoch
         epoch_acc = running_correct / len(train_loader.dataset)
         all_acc.append(epoch_acc)

         # nn has no mean and std deviation, because it only make decision one time.
         # std deviation in each epoch
         # epoch_std = np.std(running_correct)
         # all_std.append(epoch_std)

         print('Epoch', epoch, 'Loss', epoch_loss, 'Accuracy', epoch_acc)


    # plot loss
    plt.subplot(211)
    plt.plot(np.arange(0, num_epoch), all_losses, label='loss')
    plt.xlabel('epoch')
    plt.ylabel('Loss')

    plt.subplot(212)
    plt.plot(np.arange(0, num_epoch), all_acc, label='accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()



def test(input_data):
    net1 = Net(n_input=15, n_hidden=20, n_output=2)
    net1.load_state_dict(torch.load(PATH))
    test_correct = 0
    total = 0

    with torch.no_grad():
        for batch_id, data in enumerate(input_data):
            test_x, test_y = data
            out = net1(test_x)
            _, test_predict = torch.max(out.data, 1)
            total += test_y.size(0)
            test_correct += torch.sum(test_predict == test_y).item()

    print('Acurracy:', 100 * test_correct / total)
    np.save('/home/mae/Python_Code/CyberPortect_BNN/CyberPortect_BNN/y_nn_test_static', test_y)
    np.save('/home/mae/Python_Code/CyberPortect_BNN/CyberPortect_BNN/y_nn_predict_static', test_predict)




if __name__ == '__main__':
    '''
    prepare train data and test data
    '''

    # File PATH
    FILE_PATH_TRAIN_static = '191217_test_record_iter_fix_static1.csv'  # static obstacle csv file
    FILE_PATH_TRAIN_dynamic = 'test_record_iter_fix_static.csv'  # dynamic obstacle csv file

    preprocessed_data = data_prepare(FILE_PATH_TRAIN_static, dynamic=False)

    # balance data
    #preprocessed_data = balance_data(preprocessded_data)

    # prepare features and label: x, y
    # preprocessed_data = preprocessed_data.iloc[0:1000]
    preprocessed_data = preprocessed_data.values
    x = preprocessed_data[:, : -1]
    y = preprocessed_data[:, -1]

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

    #define network
    net = Net(n_input=15, n_hidden=20, n_output=2)
    print(net)

    '''
    Optimizer
    '''
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    loss_func = nn.CrossEntropyLoss()

    #train model
    num_epoch = 20
    trainer(train_loader,  num_epoch)

    #save the trained model
    PATH = 'net_params.pkl'
    torch.save(net.state_dict(), PATH)

    # test model
    test(test_loader)



