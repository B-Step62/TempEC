# -*- coding: utf-8 -*-
import sys, glob, csv, os

import numpy as np

import sklearn
import scipy.stats
from sklearn.metrics import make_scorer, roc_curve, auc

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

import matplotlib.pyplot as plt


def train_and_test_crf(test_name):
    print("===== test_name:{0} =====".format(test_name))
    X_train_raw, Y_train_raw = read_train_data(test_name)
    X_train, Y_train = [],[]
    for i in range(len(X_train_raw) - 10):
        X_train.append([(lambda x: str(1) if x > float(0.5) else str(0))(X_train_raw[i + k]) for k in range(10)])
        Y_train.append([str(Y_train_raw[i + k]) for k in range(10)])

    crf = sklearn_crfsuite.CRF(
         algorithm = 'lbfgs',
         c1 = 0.1,
         c2 = 0.1,
         max_iterations = 100,
         all_possible_transitions=True
    )
    print("train start...")
    crf.fit(X_train, Y_train)

    print("test start...")
    X_test_raw, Y_test_raw = read_test_data(test_name)
    X_test, Y_test = [],[]
    for i in range(len(X_test_raw) - 10):
        X_test.append([(lambda x: str(1) if x > float(0.5) else str(0))(X_test_raw[i + k]) for k in range(10)])
        Y_test.append([str(Y_test_raw[i + k]) for k in range(10)])

    print("test data num: {0}".format(len(X_test)))

    Y_pred = crf.predict(X_test)

    #F score
    miss_count = 0
    Y_pred = [int(Y[0]) for Y in Y_pred]
    Y_test = [int(Y[0]) for Y in Y_test]
    tp,tn,fp,fn = 0,0,0,0
    count = 0
    for pred in Y_pred:
        if pred == Y_test[count]:
            if pred == 0:
                tn += 1
            else:
                tp += 1
        else:
            if pred == 0:
                fn += 1
            else:
                fp += 1
            miss_count += 1
        count += 1
    print("miss count : {0}".format(miss_count))

    accuracy = (tp + tn) / (tp + fn + fp + tn)
    if tp + fp != 0:
        precision = tp / (tp + fp)
        if tp + fn == 0:
            recall = 0
        else:
            recall = tp / (tp + fn)
        if recall + precision == 0:
            f_value = 0
        else:
            f_value = (2 * recall * precision) / (recall + precision)
    else:
        precision = 0
        recall = 0
        f_value = 0

    result_path = 'result/CRF/{0}'.format(test_name)
    if not os.path.exists(result_path):
        os.mkdir(result_path)

    f = open(os.path.join(result_path,'result.txt'.format(test_name)), 'w')
    f.write('\nTHRESHOLD : 0.5')
    f.write('\nTrue Negative  = {0:5d}  | False Negative = {1:5d}'.format(tn,fn))
    f.write('\nFalse Positive = {0:5d}  | True Positive  = {1:5d}\n'.format(fp,tp))
    f.write('\nAccuracy  = %01.4f' % accuracy)
    f.write('\nPrecision = %01.4f' % precision)
    f.write('\nRecall    = %01.4f' % recall)
    f.write('\nF_value   = %01.4f\n' % f_value)


    #csvファイルに結果を保存(正解クラスと識別結果のペア)
    result_pairs = []
    for i in range(len(Y_test)):
        result_pairs.append([Y_test[i], Y_pred[i]])
    np.savetxt(os.path.join(result_path,'result.csv'),result_pairs,delimiter=',')


    #ROCカーブを計算、画像で保存
    #fpr, tpr, thresholds = roc_curve(Y_test, Y_pred)
    #roc_auc = auc(fpr, tpr)

    #plt.clf()
    #plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    #plt.plot([0, 1], [0, 1], 'k--')
    #plt.xlim([0.0, 1.0])
    #plt.ylim([0.0, 1.0])
    #plt.xlabel('False Positive Rate')
    #plt.ylabel('True Positive Rate')
    #plt.title('ROC curve')
    #plt.legend(loc="lower right")
    #plt.savefig(os.path.join(result_path,'ROC.png'))

    plt.clf()
    plt.plot(Y_test[0:300], label='Ground Truth')
    plt.plot(Y_pred[0:300], label='prediction')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(result_path,'vaisualize.png'))

def read_train_data(test_name):
    csv_paths = glob.glob('dataset/*')

    X_train,Y_train = [],[]

    for csv_path in csv_paths:

        if csv_path.find(test_name) > -1:
            continue

        f = open(csv_path, 'r')
        reader = csv.reader(f)
        for row in reader:
            if float(row[1]) == 0:
                continue
            X_train.append(float(row[1]))
            Y_train.append(int(float(row[0])))

    #X_train = np.array(X_train, dtype=np.float32)
    #Y_train = np.array(Y_train, dtype=np.uint8)

    return X_train, Y_train

def read_test_data(test_name):
    X_test,Y_test = [],[]

    f = open('dataset/{0}.csv'.format(test_name), 'r')
    reader = csv.reader(f)
    for row in reader:
        if float(row[1]) == 0:
            continue
        X_test.append(float(row[1]))
        Y_test.append(int(float(row[0])))

    X_test = np.array(X_test, dtype=np.float32)
    Y_test = np.array(Y_test, dtype=np.uint8)

    return X_test, Y_test

if __name__ == '__main__':

    if len(sys.argv) == 2:
        test_name = sys.argv[1]
        train_and_test_crf(test_name)
    else:
        test_names = ['Avec',
                      'Aziz',
                      'Derek',
                      'Elle',
                      'Emma',
                      'Hiyane',
                      'Imaizumi',
                      'James',
                      'Kendall',
                      'Kitazumi',
                      'Liza',
                      'Neil',
                      'Ogawa',
                      'Selena',
                      'Shiraishi',
                      'Taylor']
        for test_name in test_names:
            train_and_test_crf(test_name)
