# -*- coding: utf-8 -*-

from keras.models import Model, Sequential, model_from_json
from keras import backend as K
from keras.layers import Input, Lambda
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate
from keras.optimizers import SGD
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam, Nadam
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import ModelCheckpoint

import numpy as np

import os, sys, cv2, glob
import pandas as pd

from sklearn.model_selection import train_test_split


def CNN(img_rows, img_cols):
    # 左目のネットワーク

    left_inputs = Input(shape=(img_rows,img_cols,1))
    conv1_l = Conv2D(20, (3, 3), padding='same', activation='linear')(left_inputs)
    relu1_l = LeakyReLU(alpha=0.01)(conv1_l)
    conv2_l = Conv2D(20, (3, 3), padding='same', activation='linear')(relu1_l)
    relu2_l = LeakyReLU(alpha=0.01)(conv2_l)
    pool1_l = MaxPooling2D(pool_size=(2 ,2))(conv2_l)
    drop1_l = Dropout(0.5)(pool1_l)

    conv3_l = Conv2D(50, (3, 3), padding='same', activation='linear')(drop1_l)
    relu3_l = LeakyReLU(alpha=0.01)(conv3_l)
    conv4_l = Conv2D(50, (3, 3), padding='same', activation='linear')(relu3_l)
    relu4_l = LeakyReLU(alpha=0.01)(conv4_l)
    pool2_l = MaxPooling2D(pool_size=(2 ,2))(conv4_l)
    drop2_l = Dropout(0.5)(pool2_l)

    flatten_l = Flatten()(drop2_l)
    dense_l = Dense(512, activation='linear')(flatten_l)

    # 右目のネットワーク

    right_inputs = Input(shape=(img_rows,img_cols,1))
    conv1_r = Conv2D(20, (3, 3), padding='same', activation='linear')(right_inputs)
    relu1_r = LeakyReLU(alpha=0.01)(conv1_r)
    conv2_r = Conv2D(20, (3, 3), padding='same', activation='linear')(relu1_r)
    relu2_r = LeakyReLU(alpha=0.01)(conv2_r)
    pool1_r = MaxPooling2D(pool_size=(2 ,2))(conv2_r)
    drop1_r = Dropout(0.5)(pool1_r)

    conv3_r = Conv2D(50, (3, 3), padding='same', activation='linear')(drop1_r)
    relu3_r = LeakyReLU(alpha=0.01)(conv3_r)
    conv4_r = Conv2D(50, (3, 3), padding='same', activation='linear')(relu3_r)
    relu4_r = LeakyReLU(alpha=0.01)(conv4_r)
    pool2_r = MaxPooling2D(pool_size=(2 ,2))(conv4_r)
    drop2_r = Dropout(0.5)(pool2_r)

    flatten_r = Flatten()(drop2_r)
    dense_r = Dense(512, activation='linear')(flatten_r)

    # ネットワークをマージ
    concat = concatenate([dense_l,dense_r])
    relu1 = LeakyReLU(alpha=0.01)(concat)
    drop1 = Dropout(0.5)(relu1)
    dense2 = Dense(1024, activation='linear', name='dense2_layer')(drop1)

    relu2 = LeakyReLU(alpha=0.01)(dense2)
    drop2 = Dropout(0.5)(relu2)
    dense3 = Dense(1024, activation='linear', name='dense3_layer')(drop2)

    relu3 = LeakyReLU(alpha=0.01)(dense3)
    drop3 = Dropout(0.5)(relu3)
    output = Dense(2,activation='softmax')(drop3)

    model = Model(inputs=[left_inputs,right_inputs],outputs=[output])

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    sgd = SGD(lr=5e-4, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=["accuracy"])
    return model

def CNN_Time(time_step,img_rows, img_cols):

    #まずCNN

    # 左目のネットワーク

    left_inputs = Input(shape=(time_step, img_rows,img_cols,1))
    conv1_l = TimeDistributed(Conv2D(20, (3, 3), padding='same', activation='linear'))(left_inputs)
    relu1_l = TimeDistributed(LeakyReLU(alpha=0.01))(conv1_l)
    conv2_l = TimeDistributed(Conv2D(20, (3, 3), padding='same', activation='linear'))(relu1_l)
    relu2_l = TimeDistributed(LeakyReLU(alpha=0.01))(conv2_l)
    pool1_l = TimeDistributed(MaxPooling2D(pool_size=(2 ,2)))(conv2_l)
    drop1_l = TimeDistributed(Dropout(0.5))(pool1_l)

    conv3_l = TimeDistributed(Conv2D(50, (3, 3), padding='same', activation='linear'))(drop1_l)
    relu3_l = TimeDistributed(LeakyReLU(alpha=0.01))(conv3_l)
    conv4_l = TimeDistributed(Conv2D(50, (3, 3), padding='same', activation='linear'))(relu3_l)
    relu4_l = TimeDistributed(LeakyReLU(alpha=0.01))(conv4_l)
    pool2_l = TimeDistributed(MaxPooling2D(pool_size=(2 ,2)))(conv4_l)
    drop2_l = TimeDistributed(Dropout(0.5))(pool2_l)

    flatten_l = TimeDistributed(Flatten())(drop2_l)
    dense_l = TimeDistributed(Dense(512, activation='linear'), name='dense_l_layer')(flatten_l)

    # 右目のネットワーク

    right_inputs = Input(shape=(time_step, img_rows ,img_cols ,1))
    conv1_r = TimeDistributed(Conv2D(20, (3, 3), padding='same', activation='linear'))(right_inputs)
    relu1_r = TimeDistributed(LeakyReLU(alpha=0.01))(conv1_r)
    conv2_r = TimeDistributed(Conv2D(20, (3, 3), padding='same', activation='linear'))(relu1_r)
    relu2_r = TimeDistributed(LeakyReLU(alpha=0.01))(conv2_r)
    pool1_r = TimeDistributed(MaxPooling2D(pool_size=(2 ,2)))(conv2_r)
    drop1_r = TimeDistributed(Dropout(0.5))(pool1_r)

    conv3_r = TimeDistributed(Conv2D(50, (3, 3), padding='same', activation='linear'))(drop1_r)
    relu3_r = TimeDistributed(LeakyReLU(alpha=0.01))(conv3_r)
    conv4_r = TimeDistributed(Conv2D(50, (3, 3), padding='same', activation='linear'))(relu3_r)
    relu4_r = TimeDistributed(LeakyReLU(alpha=0.01))(conv4_r)
    pool2_r = TimeDistributed(MaxPooling2D(pool_size=(2 ,2)))(conv4_r)
    drop2_r = TimeDistributed(Dropout(0.5))(pool2_r)

    flatten_r = TimeDistributed(Flatten())(drop2_r)
    dense_r = TimeDistributed(Dense(512, activation='linear'), name='dense_r_layer')(flatten_r)

    # ネットワークをマージ
    concat = concatenate([dense_l,dense_r])
    relu1 = TimeDistributed(LeakyReLU(alpha=0.01), name='relu1_layer')(concat)
    drop1 = TimeDistributed(Dropout(0.5))(relu1)
    dense2 = TimeDistributed(Dense(1024, activation='linear'), name='dense2_layer')(drop1)

    relu2 = TimeDistributed(LeakyReLU(alpha=0.01))(dense2)
    drop2 = TimeDistributed(Dropout(0.5))(relu2)
    dense3 = TimeDistributed(Dense(1024, activation='linear'), name='dense3_layer')(drop2)

    relu3 = TimeDistributed(LeakyReLU(alpha=0.01))(dense3)
    drop3 = TimeDistributed(Dropout(0.5))(relu3)
    output = TimeDistributed(Dense(2,activation='softmax'), name='softmax_layer')(drop3)

    cnn_model = Model(inputs=[left_inputs,right_inputs],outputs=[output])


    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    sgd = SGD(lr=5e-4, decay=1e-6, momentum=0.9, nesterov=True)
    nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, schedule_decay=0.004)
    cnn_model.compile(optimizer=nadam, loss='categorical_crossentropy', metrics=["accuracy"])

    return cnn_model



