# example of gaussian naive bayes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_curve
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
import seaborn as sns


'''
data preprocessing
'''
FILE_PATH_TRAIN_static = '191217_test_record_iter_fix_static1.csv' # static obstacle csv file
FILE_PATH_TRAIN_dynamic = 'test_record_iter_fix_static.csv'  # dynamic obstacle csv file


# read csv file
origin_data = pd.read_csv(FILE_PATH_TRAIN_static)

# select the needed data by columns index
features = ['grip_pos_x', 'grip_pos_y', 'grip_pos_z',
            'grip_vel_x', 'grip_vel_y', 'grip_vel_z',
            'obst_pos_x_1', 'obst_pos_y_1', 'obst_pos_z_1',
            'goal_pos_x', 'goal_pos_y', 'goal_pos_z',
            'action_x', 'action_y', 'action_z']

label = 'is_safe_action'

all_features = features + [label]

origin_data[label] = origin_data[label].astype(int)
origin_data = origin_data.loc[:, all_features]



'''
load bnn data: y_test, y_predict
'''
# static bnn data
y_bnn_test_static = np.load('/home/mae/Python_Code/CyberPortect_BNN/CyberPortect_BNN/Y_test.npy')
y_bnn_predict_static = np.load('/home/mae/Python_Code/CyberPortect_BNN/CyberPortect_BNN/yhats_test_mean.npy')

# dynamic bnn data
y_bnn_test_dynamic = np.load('/home/mae/Python_Code/CyberPortect_BNN/CyberPortect_BNN/y_test_dynamic.npy')
y_bnn_predict_dynamic = np.load('/home/mae/Python_Code/CyberPortect_BNN/CyberPortect_BNN/yhats_test_mean_dynamic.npy')


'''
load nn data: y_nn_test, y_nn_predict
'''
# static nn data
y_nn_test_static = np.load('/home/mae/Python_Code/CyberPortect_BNN/CyberPortect_BNN/y_nn_test_static.npy')
y_nn_predict_static = np.load('/home/mae/Python_Code/CyberPortect_BNN/CyberPortect_BNN/y_nn_predict_static.npy')

# dynamic nn data





def balance_data():
    # select safe_state & unsafe state out, then make their number be equal
    n_safe_state = origin_data.loc[origin_data[label] == 1]
    n_unsafe_state = origin_data.loc[origin_data[label] == 0]
    n_safe_state = n_safe_state.sample(n=len(n_unsafe_state))  # make the number of safe and unsafe be equal

    data = pd.concat([n_safe_state, n_unsafe_state], axis=0)  # concatenate safe samples and unsafe samples
    data = data.sample(frac=1)  # randomization

    return data



def timer(model, x_data, y_data):

    '''cpu time'''
    cpu_start = time.clock()
    model.fit(x_data, y_data)
    cpu_end = time.clock()
    cpu_time = cpu_end - cpu_start

    '''train time'''
    train_start = time.time()
    model.fit(x_data, y_data)
    train_end = time.time()
    train_time = train_end - train_start

    print('%s cpu time is %f' % (model, cpu_time))
    print('%s train time is %f' % (model, train_time))


def model_score(model_fit, x_test, y_test):
    # calculate score
    model_score = model_fit.score(x_test, y_test)
    print("The score of %s is: %f" % (model_fit, model_score))



def classify_report(model_fit, x_test, y_test):
    model_predict = model_fit.predict(x_test)
    print(classification_report(y_test, model_predict))


'''
def gnb_roc(x_test):

    # Gaussian Niave bayes roc curve

    y_score_gnb = gnb_fit.predict_proba(x_test)[:, 1]
    fpr_gnb, tpr_gnb, thresholds = roc_curve(y_test, y_score_gnb)
    roc_auc_gnb = metrics.auc(fpr_gnb, tpr_gnb)
    plt.stackplot(fpr_gnb, tpr_gnb, color='steelblue', alpha=0.5, edgecolor = 'black')
    plt.plot(fpr_gnb, tpr_gnb, color='black', lw=1)
    plt.text(0.17, 0.5, 'Gaussian NB (area = %0.2f)' % roc_auc_gnb)


    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate (fpr)')
    plt.ylabel('True Positive Rate (tpr)')
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    # add diagonal line
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.show()
'''



def multiple_roc(x_test):
    '''
    bnn static roc curve
    '''

    fpr_static, tpr_static, _ = roc_curve(y_bnn_test_static, y_bnn_predict_static)
    roc_auc_static = metrics.auc(fpr_static, tpr_static)

    plt.stackplot(fpr_static, tpr_static, color='coral', alpha=0.5, edgecolor='black')
    plt.plot(fpr_static, tpr_static, color='black', lw=1)
    plt.text(0.03, 0.9, 'bnn_static obstacle (area = %0.2f)' % roc_auc_static)


    '''
    nn static roc curve
    '''
    fpr_nn_static, tpr_nn_static, _ = roc_curve(y_nn_test_static, y_nn_predict_static)
    roc_auc_nn_static = metrics.auc(fpr_nn_static, tpr_nn_static)
    plt.stackplot(fpr_nn_static, tpr_nn_static, color='coral', alpha=0.5, edgecolor='black')
    plt.plot(fpr_nn_static, tpr_nn_static, color='black', lw=1)
    plt.text(0.05, 0.8, 'nn_static obstacle (area = %0.2f)' % roc_auc_nn_static)




    '''
    gnb roc curve
    '''
    y_score_gnb = gnb_fit.predict_proba(x_test)[:, 1]
    fpr_gnb, tpr_gnb, thresholds = roc_curve(y_test, y_score_gnb)
    roc_auc_gnb = metrics.auc(fpr_gnb, tpr_gnb)
    plt.stackplot(fpr_gnb, tpr_gnb, color='steelblue', alpha=0.5, edgecolor = 'black')
    plt.plot(fpr_gnb, tpr_gnb, color='black', lw=1)
    plt.text(0.17, 0.5, 'Gaussian NB (area = %0.2f)' % roc_auc_gnb)


    '''
    mnb roc curve
    '''
    y_score_mnb = mnb_fit.predict_proba(x_test)[:, 1]
    fpr_mnb, tpr_mnb, thresholds = roc_curve(y_test, y_score_mnb)
    roc_auc_mnb = metrics.auc(fpr_mnb, tpr_mnb)

    # add area
    plt.stackplot(fpr_mnb, tpr_mnb, color='salmon', alpha=0.5, edgecolor = 'black')
    # add boundary line
    plt.plot(fpr_mnb, tpr_mnb, color='black', lw=1)
    # add test information
    plt.text(0.5, 0.1, 'Multinomial NB (area = %0.2f)' % roc_auc_mnb)

    '''
    bnb roc curve
    '''
    y_score_bnb = bnb_fit.predict_proba(x_test)[:, 1]
    fpr_bnb, tpr_bnb, thresholds = roc_curve(y_test, y_score_bnb)
    roc_auc_bnb = metrics.auc(fpr_bnb, tpr_bnb)
    plt.stackplot(fpr_bnb, tpr_bnb, color='aquamarine', alpha=0.5, edgecolor = 'black')
    plt.plot(fpr_bnb, tpr_bnb, color='black', lw=1)
    plt.text(0.2, 0.3, 'Bernoulli NB (area = %0.2f)' % roc_auc_bnb)



    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate (fpr)')
    plt.ylabel('True Positive Rate (tpr)')
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    # add diagonal line
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.show()


def plot_confusion_metric(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    cm = pd.DataFrame(cm, columns=['fake', 'true'], index=['fake', 'true'])
    sns.heatmap(cm, annot=True, cmap='GnBu', fmt='d')
    plt.xlabel('Real')
    plt.ylabel('Predict')
    plt.show()




def bnn_nn_roc():
    '''
    bnn static roc
    '''

    fpr_static, tpr_static, _ = roc_curve(y_bnn_test_static, y_bnn_predict_static)
    roc_auc_static = metrics.auc(fpr_static, tpr_static)

    plt.stackplot(fpr_static, tpr_static, color='coral', alpha=0.5, edgecolor='black')
    plt.plot(fpr_static, tpr_static, color='black', lw=1)
    plt.text(0.03, 0.9, 'bnn_static obstacle (area = %0.2f)' % roc_auc_static)


    '''
    bnn dynamic roc 
    '''

    
    fpr_dynamic, tpr_dynamic, _ = roc_curve(y_bnn_test_dynamic, y_bnn_predict_dynamic)
    roc_auc_dynamic = metrics.auc(fpr_dynamic, tpr_dynamic)
    # add area
    plt.stackplot(fpr_dynamic, tpr_dynamic, color='steelblue', alpha=0.5, edgecolor='black')
    # add boundary line
    plt.plot(fpr_dynamic, tpr_dynamic, color='black', lw=1)
    # add test information
    plt.text(0.1, 0.7, 'bnn_dynamic obstacle (area = %0.2f)' % roc_auc_dynamic)



    '''
    nn roc curve
    '''
    fpr_nn_static, tpr_nn_static, _ = roc_curve(y_nn_test_static, y_nn_predict_static)
    roc_auc_nn_static = metrics.auc(fpr_nn_static, tpr_nn_static)
    plt.stackplot(fpr_nn_static, tpr_nn_static, color='coral', alpha=0.5, edgecolor='black')
    plt.plot(fpr_nn_static, tpr_nn_static, color='black', lw=1)
    plt.text(0.05, 0.8, 'nn_static obstacle (area = %0.2f)' % roc_auc_nn_static)


    '''
    nn dynamic
    '''
    pass



    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate (fpr)')
    plt.ylabel('True Positive Rate (tpr)')
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    # add diagonal line
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.show()




if __name__ == '__main__':

    '''
    prepare train data and test data
    '''
    # prepare features and label: x, y
    preprocessed_data = balance_data()
    preprocessed_data = preprocessed_data.values
    x = preprocessed_data[:, 0:len(features)-1]
    y = preprocessed_data[:, len(all_features)-1]

    # splite data to train data and test data
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.01, random_state=42)

    'build GNB, MNB, BNB'
    GNB = GaussianNB()  # Gaussian Naive Bayes
    MNB = MultinomialNB()  # Multinomial Naive Bayes
    BNB = BernoulliNB()    # Bernoulli Naive Bayes


    'model fit data'
    gnb_fit = GNB.fit(x_train, y_train)
    mnb_fit = MNB.fit(abs(x_train), y_train)  # absolute value for Multinomial Naive Bayes
    bnb_fit = BNB.fit(x_train, y_train)


    'cpu/train time'
    timer(GNB, x_train, y_train)
    timer(MNB, abs(x_train), y_train)
    timer(BNB, x_train, y_train)


    'model score'
    model_score(gnb_fit, x_test, y_test)
    model_score(mnb_fit, x_test, y_test)
    model_score(bnb_fit, x_test, y_test)


    'classification report'
    print('GNB classification report')
    classify_report(gnb_fit, x_test, y_test)

    print('MNB classification report')
    classify_report(mnb_fit, x_test, y_test)

    print('BNB classification report')
    classify_report(bnb_fit, x_test, y_test)

    # bnn classification report
    #print('bnn classification report', classification_report(y_test_static, y_predict_static))


    '''
    bnn confusion metric
    '''
    #plot_confusion_metric(y_test_static, y_predict_static)


    '''
    multiple roc curve
    '''
    multiple_roc(x_test)
    bnn_nn_roc()
























