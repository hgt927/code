import numpy as np
from sklearn import *
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

def load_data(path, num):
    labels = np.zeros((num))
    data = np.zeros((num, 100))

    with open(path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i == num:
                break
            for j, val in enumerate(line.split(' ')):
                if val == '':
                    break
                if j == 0:
                    labels[i] = (int(val))
                elif j == 1:
                    continue
                else:
                    data[i][j-2] = float(val.strip())

    return data, labels


def plot_fig_2d(data_converted, label, title, SAVE):
    colors = 'rb'
    markers = 'ox'
    for target, color, marker in zip([0, 1], colors, markers):
        pos = (label == target).ravel()
        data = data_converted[pos, :]
        plt.scatter(data[:, 0], data[:, 0], color=color, marker=marker, label="Label %d" % target)
    plt.title(title)
    plt.legend(loc='best')
    if SAVE:
        name = title + '.png'
        plt.savefig(fname=name)
    plt.show()


if __name__ == '__main__':
    DRAW = 1
    SAVE = 0

    train_path = 'train.txt'
    test_path = 'test.txt'

    train_data, train_labels = load_data(train_path, 1996)
    test_data, test_labels = load_data(test_path, 1996)

    # 合并
    data = np.vstack((train_data, test_data))
    labels = np.vstack((train_labels.reshape(train_labels.size, 1), test_labels.reshape(test_labels.size, 1)))

    lda = discriminant_analysis.LinearDiscriminantAnalysis()
    lda.fit(data, labels.ravel())
    data_lda = np.dot(data, np.transpose(lda.coef_)) + lda.intercept_
    train_data_lda = data_lda[0:train_data.shape[0]]
    test_data_lda = data_lda[train_data.shape[0]:]
    lda_score = lda.score(test_data, test_labels)
    print("LDA score: %.2f" % lda_score)

    if DRAW:
        plot_fig_2d(test_data, test_labels, title='RAW Test feature', SAVE=SAVE)
        plot_fig_2d(train_data, train_labels, title='RAW Train feature', SAVE=SAVE)
        plot_fig_2d(test_data_lda, test_labels, title='LDA Test feature', SAVE=SAVE)
        plot_fig_2d(train_data_lda, train_labels, title='LDA Train feature', SAVE=SAVE)

    bays = GaussianNB()
    score = cross_val_score(bays, data_lda, labels.ravel()) # 交叉验证训练并且给出score
    print('avg score:', np.average(score))
