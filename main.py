# -*- coding: utf-8 -*-

import numpy as np

import os, sys, cv2, glob, csv, random
import pandas as pd

from sklearn.model_selection import train_test_split

from keras import backend as K
from keras.models import model_from_json, Model, Sequential
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import Adam, SGD, Nadam
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Reshape, Embedding, Merge
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing.image import ImageDataGenerator
from keras_contrib.layers import CRF

from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot

import matplotlib.pyplot as plt 


from sklearn.metrics import roc_curve, auc

# 自作関数たち
import NETWORK
from data_utils import MyDatasetLoader

# seed値
np.random.seed(1)

#lstmのunroll数
time_step = 10

#testのしきい値
threshold = 0.5

#バッチサイズ
BATCH = 128

# 使用する画像サイズ 片目ずつ
img_rows, img_cols = 36, 60


#CNNのモデルを構成
def make_model_develop():
    # モデル構成のファイル名
    json_path = 'cache/cnn/develop/architecture_DEEPEC-NP.json'
    # DEEPECの学習済みCNNモデルの重み
    weight_path = 'cache/cnn/develop/model_weights_DEEPEC-NP_99.h5'

    # モデルの構成を読込み、jsonからモデルオブジェクトへ変換
    model = model_from_json(open(json_path).read())

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    sgd = SGD(lr=5e-4, decay=1e-6, momentum=0.9, nesterov=True)
    nadam = Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, schedule_decay=0.004)
    model.compile(optimizer=nadam, loss='categorical_crossentropy', metrics=["accuracy"])

    # モデルオブジェクトへ重みを読み込む
    model.load_weights(weight_path)

    return model


#10枚まとめて放り込むモデル
def make_model_lstm1():
    cnn_model = NETWORK.CNN_Time(time_step,img_rows, img_cols)

    #developの重みを読み込む
    weight_name = 'model_weights_DEEPEC-NP_99.h5'

    reading_path = 'cache/cnn/develop'

    cnn_model.load_weights(os.path.join(reading_path, weight_name))

    #後半のLSTM
    lstm_model = Sequential()

    lstm_model.add(LSTM(1024 ,batch_input_shape=(None, time_step, 1024), return_sequences=True))
    lstm_model.add(LeakyReLU(alpha=0.01))
    lstm_model.add(Dropout(0.5))
    lstm_model.add(LSTM(1024, return_sequences=False))
    lstm_model.add(LeakyReLU(alpha=0.01))
    lstm_model.add(Dropout(0.5))

    lstm_model.add(Dense(2, activation='softmax'))

    #統合
   
    model = Model(input = cnn_model.input, output = lstm_model(cnn_model.get_layer('dense3_layer').output))
    nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, schedule_decay=0.004, clipnorm=1.)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    sgd = SGD(lr=5e-4, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=nadam, loss='categorical_crossentropy', metrics=["accuracy"])

    return model

#CNNのconcatenateする前の右左それぞれの特徴を入力とする。また10枚をLSTMで学習し（文脈）、その結果を10フレーム目の特徴と合わせて識別するモデル
def make_model_lstm2():

    K.set_learning_phase(1)
 
    cnn_model = NETWORK.CNN_Time(time_step,img_rows, img_cols)

    #developの重みを読み込む
    weight_name = 'model_weights_DEEPEC-NP_99.h5'

    reading_path = 'cache/cnn/develop'

    cnn_model.load_weights(os.path.join(reading_path, weight_name))

    left_inputs = Input(shape=(time_step,512))
    left_inputs0_8 = Lambda(lambda x: x[:, :9], output_shape=(9,512))(left_inputs)
    left_inputs9 = Lambda(lambda x: x[:, 9])(left_inputs)

    lstm_l = LSTM(512, dropout = 0.5, return_sequences=True)(left_inputs)
    lstm_l_relu = TimeDistributed(LeakyReLU(alpha=0.01))(lstm_l)
    lstm_l2 = LSTM(512, dropout = 0.5, return_sequences=False)(lstm_l_relu)
    lstm_l_relu2 = LeakyReLU(alpha=0.01)(lstm_l2)  

    right_inputs = Input(shape=(time_step,512))
    right_inputs0_8 = Lambda(lambda x: x[:, :9], output_shape=(9,512))(right_inputs)
    right_inputs9 = Lambda(lambda x: x[:, 9])(right_inputs)
    lstm_r = LSTM(512, dropout = 0.5, return_sequences=True)(right_inputs)
    lstm_r_relu = LeakyReLU(alpha=0.01)(lstm_r)
    lstm_r2 = LSTM(512, dropout = 0.5, return_sequences=False)(lstm_r_relu)
    lstm_r_relu2 = LeakyReLU(alpha=0.01)(lstm_r2)

    concat = concatenate([lstm_l_relu2, left_inputs9, lstm_r_relu2, right_inputs9])

    relu1 = LeakyReLU(alpha=0.01)(concat)
    drop1 = Dropout(0.5)(relu1)
    dense2 = Dense(1024, activation='linear')(drop1) 
    
    relu2 = LeakyReLU(alpha=0.01)(dense2)
    drop2 = Dropout(0.5)(relu2)
    dense3 = Dense(1024, activation='linear')(drop2)

    relu3 = LeakyReLU(alpha=0.01)(dense2)
    drop3 = Dropout(0.5)(relu3)
    output = Dense(2, activation='softmax')(drop3) 
    

    lstm_model = Model(inputs=[left_inputs, right_inputs],outputs=[output])

    model = Model(input = cnn_model.input, output = lstm_model([cnn_model.get_layer('dense_l_layer').output, cnn_model.get_layer('dense_r_layer').output]))    

    nadam = Nadam(lr=0.0005, beta_1=0.9, beta_2=0.999, schedule_decay=0.004, clipnorm=1.)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipnorm=1.)
    sgd = SGD(lr=5e-4, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=1.)

    model.compile(optimizer=nadam, loss='categorical_crossentropy', metrics=["accuracy"])

    #plot_model(lstm_model, show_layer_names = False, show_shapes = True, to_file='LSTM_model.png')

    return model

#concatはしておくが、9フレーム目までをLSTMで学習し（文脈）、その結果を10フレーム目の特徴と合わせて識別するモデル
def make_model_lstm3():

    K.set_learning_phase(1)
 
    cnn_model = NETWORK.CNN_Time(time_step,img_rows, img_cols)

    #developの重みを読み込む
    weight_name = 'model_weights_DEEPEC-NP_99.h5'

    reading_path = 'cache/cnn/develop'

    cnn_model.load_weights(os.path.join(reading_path, weight_name))

    left_inputs = Input(shape=(time_step,512))
    right_inputs = Input(shape=(time_step,512))
  
    inputs = concatenate([left_inputs, right_inputs])
    
    inputs0_8 = Lambda(lambda x: x[:, :9], output_shape=(9,1024))(inputs)
    inputs9 = Lambda(lambda x: x[:, 9])(inputs)

    lstm = LSTM(1024, dropout = 0.5, return_sequences=True)(inputs0_8)
    lstm_relu = TimeDistributed(LeakyReLU(alpha=0.01))(lstm)
    lstm_2 = LSTM(1024, dropout = 0.5, return_sequences=False)(lstm_relu)
    lstm_relu2 = LeakyReLU(alpha=0.01)(lstm_2)  



    concat = concatenate([lstm_relu2, inputs9])

    relu1 = LeakyReLU(alpha=0.01)(concat)
    drop1 = Dropout(0.5)(relu1)
    dense2 = Dense(1024, activation='linear')(drop1) 
    
    relu2 = LeakyReLU(alpha=0.01)(dense2)
    drop2 = Dropout(0.5)(relu2)
    dense3 = Dense(1024, activation='linear')(drop2)

    relu3 = LeakyReLU(alpha=0.01)(dense2)
    drop3 = Dropout(0.5)(relu3)
    output = Dense(2, activation='softmax')(drop3) 
    

    lstm_model = Model(inputs=[left_inputs, right_inputs],outputs=[output])

    model = Model(input = cnn_model.input, output = lstm_model([cnn_model.get_layer('dense_l_layer').output, cnn_model.get_layer('dense_r_layer').output]))    

    nadam = Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, schedule_decay=0.004, clipnorm=1.)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipnorm=1.)
    sgd = SGD(lr=5e-4, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=1.)

    model.compile(optimizer=nadam, loss='categorical_crossentropy', metrics=["accuracy"])

    #plot_model(lstm_model, show_layer_names = False, show_shapes = True, to_file='LSTM_model.png')

    return model


#lstm2に顔方向の入力を追加したモデル
def make_model_lstm_face():

    K.set_learning_phase(1)
 
    cnn_model = NETWORK.CNN_Time(time_step,img_rows, img_cols)

    #developの重みを読み込む
    weight_name = 'model_weights_DEEPEC-NP_99.h5'

    reading_path = 'cache/cnn/develop'

    cnn_model.load_weights(os.path.join(reading_path, weight_name))

    face_inputs = Input(shape=(time_step,3))
    #face_inputs0_8 = Lambda(lambda x: x[:, :9], output_shape=(9,3))(face_inputs)
    face_inputs9 = Lambda(lambda x: x[:, 9])(face_inputs)

    left_inputs = Input(shape=(time_step,512))
    #left_inputs0_8 = Lambda(lambda x: x[:, :9], output_shape=(9,512))(left_inputs)
    left_inputs9 = Lambda(lambda x: x[:, 9])(left_inputs)

    left_face_inputs = concatenate([left_inputs, face_inputs], axis=2)
    left_face_inputs9 = concatenate([left_inputs9, face_inputs9])

    lstm_l = LSTM(512, dropout = 0.5, return_sequences=True)(left_face_inputs)
    lstm_l_relu = TimeDistributed(LeakyReLU(alpha=0.01))(lstm_l)
    lstm_l2 = LSTM(512, dropout = 0.5, return_sequences=False)(lstm_l_relu)
    lstm_l_relu2 = LeakyReLU(alpha=0.01)(lstm_l2)  

    right_inputs = Input(shape=(time_step,512))
    #right_inputs0_8 = Lambda(lambda x: x[:, :9], output_shape=(9,512))(right_inputs)
    right_inputs9 = Lambda(lambda x: x[:, 9])(right_inputs)

    right_face_inputs = concatenate([right_inputs, face_inputs], axis=2)
    right_face_inputs9 = concatenate([right_inputs9, face_inputs9])

    lstm_r = LSTM(512, dropout = 0.5, return_sequences=True)(right_face_inputs)
    lstm_r_relu = LeakyReLU(alpha=0.01)(lstm_r)
    lstm_r2 = LSTM(512, dropout = 0.5, return_sequences=False)(lstm_r_relu)
    lstm_r_relu2 = LeakyReLU(alpha=0.01)(lstm_r2)




    concat = concatenate([lstm_l_relu2, left_face_inputs9, lstm_r_relu2, right_face_inputs9])

    relu1 = LeakyReLU(alpha=0.01)(concat)
    drop1 = Dropout(0.5)(relu1)
    dense2 = Dense(1024, activation='linear')(drop1) 
    
    relu2 = LeakyReLU(alpha=0.01)(dense2)
    drop2 = Dropout(0.5)(relu2)
    dense3 = Dense(1024, activation='linear')(drop2)

    relu3 = LeakyReLU(alpha=0.01)(dense2)
    drop3 = Dropout(0.5)(relu3)
    output = Dense(2, activation='softmax')(drop3) 
    

    lstm_model = Model(inputs=[left_inputs, right_inputs, face_inputs],outputs=[output])

    model = Model(input = [cnn_model.input[0], cnn_model.inputs[1], face_inputs], output = lstm_model([cnn_model.get_layer('dense_l_layer').output, cnn_model.get_layer('dense_r_layer').output, face_inputs]))    

    nadam = Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, schedule_decay=0.004, clipnorm=1.)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, clipnorm=1.)
    sgd = SGD(lr=5e-4, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=1.)

    model.compile(optimizer=nadam, loss='categorical_crossentropy', metrics=["accuracy"])

    #Emmaのみで20エポック学習した重み
    #model.load_weights('cache/lstm/Emma20ep/model_weights_DEEPEC-NP_19.h5')

    return model

#crf
def make_model_crf():
    model = Sequential() 
    model.add(Embedding(2, 2, mask_zero=True)) 
    crf = CRF(2, sparse_target=True)
    model.add(crf)

    nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, schedule_decay=0.004, clipnorm=1.)

    model.compile(optimizer=nadam, loss=crf.loss_function, metrics=[crf.accuracy])

    return model


#テスト用にモデルと重さを読み込み
def read_model(modelStr, epoch, cnn_or_lstm,test_name):
    # モデル構成のファイル名
    json_name = 'architecture_%s.json' % modelStr
    # モデル重みのファイル名
    weight_name = 'model_weights_%s_%02d.h5'%(modelStr, epoch - 1)

    if cnn_or_lstm == 'cnn':
        reading_path = 'cache/{0}/{1}'.format(cnn_or_lstm,test_name)
    else:
        reading_path = 'cache/{0}/timestep={1}/{2}'.format(cnn_or_lstm,time_step,test_name)
    # モデルの構成を読込み、jsonからモデルオブジェクトへ変換
    model = model_from_json(open(os.path.join(reading_path, json_name)).read())
    # モデルオブジェクトへ重みを読み込む
    model.load_weights(os.path.join(reading_path, weight_name))

    return model

# モデルの構成を保存
def save_model(model, modelStr,cnn_or_lstm, test_name):
    # モデルオブジェクトをjson形式に変換
    json_string = model.to_json()

    if cnn_or_lstm == 'cnn':
        output_path = 'cache/{0}/{1}'.format(cnn_or_lstm,test_name)
    else:
        output_path = 'cache/{0}/timestep={1}/{2}'.format(cnn_or_lstm,time_step,test_name)
        # カレントディレクトリにcacheディレクトリがなければ作成
        output_path = 'cache/{0}/timestep={1}/{2}'.format(cnn_or_lstm,time_step,test_name)

    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    # モデルの構成を保存するためのファイル名
    json_name = 'architecture_%s.json' % modelStr
    # モデル構成を保存
    open(os.path.join(output_path, json_name), 'w').write(json_string)



################## C N N  T R A I N ##################################################################
def run_train_cnn(modelStr, epoch, test_name):
    model = make_model_develop()

    X_train,Y_train, frame_nums = MyDatasetLoader.read_train_data_cnn(test_name)

    if not os.path.exists('cache/cnn/{0}'.format(test_name)):
        os.mkdir('cache/cnn/{0}'.format(test_name))
    
    cp = ModelCheckpoint('cache/cnn/%s/model_weights_%s_{epoch:02d}.h5'%(test_name, modelStr), monitor='val_loss', save_best_only=False)

    #class_weightを計算
    weight_0, weight_1 = 0, 0
    for y in Y_train:
        if y[0] == 0:
            weight_0 += 1
        if y[0] == 1:
            weight_1 += 1  
    normalized_weight_0 = weight_0 / (weight_0 + weight_1)
    normalized_weight_1 = weight_1 / (weight_0 + weight_1)
    print('0:{0} 1:{1}'.format(normalized_weight_0,normalized_weight_1)) 
    class_weight = {0 : normalized_weight_0, 1 : normalized_weight_1}

    print('start train (test_name : {0})'.format(test_name))
    # train実行
    hist = model.fit(X_train, Y_train, batch_size=64,
              epochs=epoch,
              verbose=1,
              validation_split = 0.1,
              shuffle=True,
              class_weight = class_weight,
              callbacks=[cp,LearningRateScheduler(lambda ep: float(1e-3 / 3 ** (ep * 4 // epoch)))])
    # モデルの構成を保存
    save_model(model, modelStr,'cnn',test_name)

    #学習の様子をグラフで保存
    loss = hist.history['acc']
    val_loss = hist.history['val_acc']

    plt.plot(range(epoch), loss, marker='.', label = 'acc')
    plt.plot(range(epoch), val_loss, marker = '.', label='val_acc')
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('acc')
    if not os.path.exists('result/cnn/{0}'.format(test_name)):
        os.makedirs('result/cnn/{0}'.format(test_name))
    plt.savefig('result/cnn/{0}/trainplt.png'.format(test_name))

    hist = []


################## C N N  T E S T ##################################################################
def run_test_cnn(modelStr,epoch,test_name):
    print('test_name : {0}'.format(test_name))
    results = []

    X_test, Y_test, img_paths = MyDatasetLoader.read_test_data_cnn(test_name,time_step)


    img_nums = []
    for img_path in img_paths:
        path,ext = os.path.splitext(img_path)
        path,num = os.path.split(path)
        img_nums.append(int(num)) 

    model = read_model(modelStr,epoch,'cnn',test_name)
    test_result = model.predict(X_test,batch_size=128,verbose=1)

    test_result_classes = list(map(lambda x:1 if x > float(0.5) else 0,test_result[:,1])) 


    #結果保存用のディレクトリ
    result_path = 'result/cnn/%s/ep%d'%(test_name, epoch)
    if not os.path.exists(result_path):
        os.makedirs(result_path)


    #誤識別した画像を保存
    count = 0
    miss_count = 0
    if not os.path.exists(os.path.join(result_path,'img')):
        os.makedirs(os.path.join(result_path,'img'))

    for root, dirs, files in os.walk(os.path.join(result_path,'img/FP'), topdown=False):
         for name in files:
                os.remove(os.path.join(result_path,'img/FP', name))
    for root, dirs, files in os.walk(os.path.join(result_path,'img/FN'), topdown=False):
         for name in files:
                os.remove(os.path.join(result_path,'img/FN', name))

    for res in test_result_classes:
        if res != Y_test[count][1]:
            miss_count += 1
            image = cv2.imread(img_paths[count])
            if int(Y_test[count][1]) == 0:
                dir_path = os.path.join(result_path,'img/FP')
            else:
                dir_path = os.path.join(result_path,'img/FN')
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path)
                    
            head,tail = os.path.split(img_paths[count]) #tailは'0001.png'
            new_path = os.path.join(dir_path, tail)
            cv2.imwrite(new_path, image) 
        count += 1


    #ミスの数を表示
    print('miss_amount:{0}'.format(str(miss_count)))


    #F値などをtextで保存
    tp,tn,fp,fn = 0,0,0,0
    count = 0
    for res in test_result_classes:
        if res == Y_test[count][1]:
            if res == 0:
                tn += 1
            else:
                tp += 1
        else:
            if res == 0:
                fn += 1
            else:
                fp += 1
        count += 1

    if tp + fp != 0:
        accuracy = (tp + tn) / (tp + fn + fp + tn)
        precision = tp / (tp + fp)
        if tp + fn == 0:
            recall = 0
        else:
            recall = tp / (tp + fn)
        if recall + precision == 0:
            f_value = 0
        else:  
            f_value = (2 * recall * precision) / (recall + precision)

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
    for i in range(len(Y_test[:,1])):
        result_pairs.append([Y_test[:,1][i], test_result[:,1][i]])

    result_pairs_save = []
    count = 0
    for i in range(img_nums[len(img_nums) - 1]):
        if i == img_nums[count]:
            result_pairs_save.append(result_pairs[count])
            count+=1
        else:
            result_pairs_save.append([0,0])

    np.savetxt(os.path.join(result_path,'result.csv'),result_pairs_save,delimiter=',')


    #ROCカーブを計算、画像で保存
    fpr, tpr, thresholds = roc_curve(Y_test[:,1], test_result[:,1])
    roc_auc = auc(fpr, tpr)       

    plt.clf()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(result_path,'ROC.png'))




################## L S T M  T R A I N ##################################################################
def run_train_lstm(modelStr,epoch,test_name):
    model = make_model_lstm2()

    X_train_paths, Y_train = MyDatasetLoader.read_train_data_lstm(test_name, time_step)


    if not os.path.exists('cache/lstm/timestep={0}'.format(time_step)):
        os.mkdir('cache/lstm/timestep={0}'.format(time_step))

    if not os.path.exists('cache/lstm/timestep={0}/{1}'.format(time_step,test_name)):
        os.mkdir('cache/lstm/timestep={0}/{1}'.format(time_step,test_name))

    cp = ModelCheckpoint('cache/lstm/timestep=%s/%s/model_weights_%s_{epoch:02d}.h5'%(time_step,test_name, modelStr), monitor='val_loss', save_best_only=False)


    print('start train (test_name : {0})'.format(test_name))
 
    #バッチの順番をシャッフル
    batch_nums = []
    for i in range(len(X_train_paths[0])//BATCH):
        batch_nums.append(i)
    random.shuffle(batch_nums)

    #validationとtrainにわける 99:1
    train_nums = batch_nums[: len(batch_nums) * 4 // 5]
    validation_nums = batch_nums[len(batch_nums) * 4 // 5:]


    #validation_data
    X_l_val, X_r_val, Y_val = [],[],[]
    for j in range(len(validation_nums)):
        X_val_batch, Y_val_batch = MyDatasetLoader.train_batch_create(X_train_paths, Y_train, validation_nums[j],BATCH)
        for l in X_val_batch[0]:
            X_l_val.append(l)
        for r in X_val_batch[1]:
            X_r_val.append(r)
        for y in Y_val_batch:
            Y_val.append(y)             
    X_val = [np.array(X_l_val,dtype=np.float32), np.array(X_r_val,dtype=np.float32)]
    Y_val = np.array(Y_val,dtype=np.uint8)


    #val_accの記録
    val_acc_hist = []
     
    #学習ループ
    for ep in range(epoch):
        print('Epoch {0}/{1}\r'.format(ep + 1, epoch))
        #trainの順番をシャッフル
        random.shuffle(train_nums)

        #バッチの数だけループ
        for i in range(len(train_nums)): 

            #train
            X_train_batch, Y_train_batch = MyDatasetLoader.train_batch_create(X_train_paths, Y_train, train_nums[i],BATCH)

            #class_weightを計算
            weight_0, weight_1 = 0, 0
            for y in Y_train_batch:
                if y[0] == 0:
                    weight_0 += 1
                if y[0] == 1:
                    weight_1 += 1  
            normalized_weight_0 = weight_0 / (weight_0 + weight_1)
            normalized_weight_1 = weight_1 / (weight_0 + weight_1)
            class_weight = {0 : normalized_weight_0, 1 : normalized_weight_1}

            
      
            loss, acc = model.train_on_batch(X_train_batch,
                                                 Y_train_batch)#, class_weight=class_weight)

            #1000batchごとにvalidateしてみる
            if i % 1000 == 0:
                val_loss, val_acc = model.evaluate(X_val,
                                               Y_val, 
                                               batch_size = BATCH,
                                               verbose=0)
                sys.stdout.write('\r{0}/{1}  loss = {2:05f} acc = {3:05f} weight = {4:05f}:{5:06f} val_loss = {6:05f} val_acc = {7:05f}'.format(i * BATCH,len(train_nums) * BATCH,loss,acc,normalized_weight_0,normalized_weight_1,val_loss,val_acc))
                sys.stdout.flush()

            else:
                sys.stdout.write('\r{0}/{1}  loss = {2:05f} acc = {3:05f} weight = {4:05f}:{5:06f}'.format(i * BATCH,len(train_nums) * BATCH,loss,acc,normalized_weight_0,normalized_weight_1))
                sys.stdout.flush()

            #epochの最後のval_accを記録
            if i == len(train_nums) - 1:
                val_loss, val_acc = model.evaluate(X_val,
                                               Y_val, 
                                               batch_size = BATCH,
                                               verbose=0)
                val_acc_hist.append(val_acc)
                sys.stdout.write('\r{0}/{1}  loss = {2:05f} acc = {3:05f} weight = {4:05f}:{5:06f} val_loss = {6:05f} val_acc = {7:05f}'.format(i * BATCH,len(train_nums) * BATCH,loss,acc,normalized_weight_0,normalized_weight_1,val_loss,val_acc))
                sys.stdout.flush()

        print('\r')
        
        #save_weight
        model.save_weights('cache/lstm/timestep={0}/{1}/model_weights_{2}_{3:02d}.h5'.format(time_step,test_name,modelStr,ep))

    # モデルの構成を保存
    save_model(model, modelStr,'lstm',test_name)

    #学習の様子をグラフで保存
    plt.clf()
    plt.plot(range(epoch), val_acc_hist, marker = '.', label='val_acc')
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('acc')
    if not os.path.exists('result/lstm/timestep={0}/{1}'.format(time_step,test_name)):
        os.makedirs('result/lstm/timestep={0}/{1}'.format(time_step,test_name))
    plt.savefig('result/lstm/timestep={0}/{1}/trainplt.png'.format(time_step,test_name))



################## L S T M  T E S T ##################################################################
def run_test_lstm(modelStr,epoch,test_name):
    results = []

    X_test_paths, Y_test, img_paths = MyDatasetLoader.read_test_data_lstm(test_name, time_step)

    #端数は切っとく
    batch_num = len(X_test_paths[0])//BATCH
    Y_test = Y_test[:batch_num * BATCH]

    model = read_model(modelStr,epoch,'lstm',test_name)


    #結果保存用のディレクトリ
    result_path = 'result/lstm/timestep=%d/%s/ep%d'%(time_step,test_name,epoch)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

   
    print('start test (test_name : {0})'.format(test_name))

    test_result = []
    #バッチの数だけループ
    for i in range(len(X_test_paths[0])//BATCH): 

        #test
        X_test_batch = MyDatasetLoader.test_batch_create(X_test_paths, i, BATCH)
        
    
        result = model.predict_on_batch(X_test_batch)
        for r in result:
           test_result.append(r)


        sys.stdout.write('\r{0}/{1}'.format(i* BATCH,(len(X_test_paths[0])//BATCH)*BATCH))
        sys.stdout.flush()

    test_result_classes = []
    test_result = np.array(test_result,dtype=np.float32)
    test_result_classes = list(map(lambda x:1 if x > float(threshold) else 0,test_result[:,1])) 
    print(test_result_classes)


    #誤識別した画像を保存
    count = 0
    miss_count = 0
    if not os.path.exists(os.path.join(result_path,'img')):
        os.makedirs(os.path.join(result_path,'img'))

    for root, dirs, files in os.walk(os.path.join(result_path,'img/FP'), topdown=False):
         for name in files:
                os.remove(os.path.join(result_path,'img/FP', name))
    for root, dirs, files in os.walk(os.path.join(result_path,'img/FN'), topdown=False):
         for name in files:
                os.remove(os.path.join(result_path,'img/FN', name))

    for res in test_result_classes:
        if res != Y_test[count][1]:
            miss_count += 1
            image = cv2.imread(img_paths[count])
            if int(Y_test[count][1]) == 0:
                dir_path = os.path.join(result_path,'img/FP')
            else:
                dir_path = os.path.join(result_path,'img/FN')
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path)
                    
            head,tail = os.path.split(img_paths[count]) #tailは'0001.png'
            new_path = os.path.join(dir_path, tail)
            cv2.imwrite(new_path, image) 
        count += 1


    #ミスの数を表示
    print('miss_amount:{0}'.format(str(miss_count)))

    #F値などをtextで保存
    tp,tn,fp,fn = 0,0,0,0
    count = 0
    for res in test_result_classes:
        if res == Y_test[count][1]:
            if res == 0:
                tn += 1
            else:
                tp += 1
        else:
            if res == 0:
                fn += 1
            else:
                fp += 1
        count += 1

    if tp + fp != 0:
        accuracy = (tp + tn) / (tp + fn + fp + tn)
        precision = tp / (tp + fp)
        if tp + fn == 0:
            recall = 0
        else:
            recall = tp / (tp + fn)
        if recall + precision == 0:
            f_value = 0
        else:  
            f_value = (2 * recall * precision) / (recall + precision)

    f = open(os.path.join(result_path,'result.txt'.format(test_name)), 'w')
    f.write('\nTHRESHOLD : 0.5')
    f.write('\nTrue Negative  = {0:5d}  | False Negative = {1:5d}'.format(tn, fn)) 
    f.write('\nFalse Positive = {0:5d}  | True Positive  = {1:5d}\n'.format(fp, tp)) 
    f.write('\nAccuracy  = %01.4f' % accuracy)
    f.write('\nPrecision = %01.4f' % precision)
    f.write('\nRecall    = %01.4f' % recall)
    f.write('\nF_value   = %01.4f\n' % f_value)


    #csvファイルに結果を保存
    np.savetxt(os.path.join(result_path,'result.csv'),test_result,delimiter=',')
    np.savetxt(os.path.join(result_path,'seikai.csv'),Y_test,delimiter=',')


    #ROCカーブを計算、画像で保存
    fpr, tpr, thresholds = roc_curve(Y_test[:,1], test_result[:,1])
    roc_auc = auc(fpr, tpr)       

    plt.clf()
    plt.figure(figsize=[10,10])
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(result_path,'ROC.png'))

################## L S T M1  T R A I N ##################################################################
def run_train_lstm1(modelStr,epoch,test_name):
    model = make_model_lstm1()

    X_train_paths, Y_train = MyDatasetLoader.read_train_data_lstm(test_name, time_step)



    if not os.path.exists('cache/lstm1/timestep={0}'.format(time_step)):
        os.mkdir('cache/lstm1/timestep={0}'.format(time_step))

    if not os.path.exists('cache/lstm1/timestep={0}/{1}'.format(time_step,test_name)):
        os.mkdir('cache/lstm1/timestep={0}/{1}'.format(time_step,test_name))

    cp = ModelCheckpoint('cache/lstm1/timestep=%s/%s/model_weights_%s_{epoch:02d}.h5'%(time_step,test_name, modelStr), monitor='val_loss', save_best_only=False)


    print('start train (test_name : {0})'.format(test_name))

    #バッチの順番をシャッフル
    batch_nums = []
    for i in range(len(X_train_paths[0])//BATCH):
        batch_nums.append(i)
    random.shuffle(batch_nums)

    #validationとtrainにわける 99:1
    train_nums = batch_nums[: len(batch_nums) * 19 // 20]
    validation_nums = batch_nums[len(batch_nums) * 19 // 20:]


    #validation_data
    X_l_val, X_r_val, Y_val = [],[],[]
    for j in range(len(validation_nums)):
        X_val_batch, Y_val_batch = MyDatasetLoader.train_batch_create(X_train_paths, Y_train, validation_nums[j],BATCH)
        for l in X_val_batch[0]:
            X_l_val.append(l)
        for r in X_val_batch[1]:
            X_r_val.append(r)
        for y in Y_val_batch:
            Y_val.append(y)             
    X_val = [np.array(X_l_val,dtype=np.float32), np.array(X_r_val,dtype=np.float32)]
    Y_val = np.array(Y_val,dtype=np.uint8)


    #val_accの記録
    val_acc_hist = []
     
    #学習ループ
    for ep in range(epoch):
        print('Epoch {0}/{1}\r'.format(ep + 1, epoch))
        #trainの順番をシャッフル
        random.shuffle(train_nums)

        #バッチの数だけループ
        for i in range(len(train_nums)): 

            #train
            X_train_batch, Y_train_batch = MyDatasetLoader.train_batch_create(X_train_paths, Y_train, train_nums[i],BATCH)
      
            loss, acc = model.train_on_batch(X_train_batch,
                                                 Y_train_batch)

            #100batchごとにvalidateしてみる
            if i % 100 == 0:
                val_loss, val_acc = model.evaluate(X_val,
                                               Y_val, 
                                               batch_size = BATCH,
                                               verbose=0)
                sys.stdout.write('\r{0}/{1}  loss = {2:05f} acc = {3:05f} val_loss = {4:05f} val_acc = {5:05f}'.format(i * BATCH,len(train_nums) * BATCH,loss,acc,val_loss,val_acc))
                sys.stdout.flush()

            else:
                sys.stdout.write('\r{0}/{1}  loss = {2:05f} acc = {3:05f}'.format(i * BATCH,len(train_nums) * BATCH,loss,acc))
                sys.stdout.flush()

            #epochの最後のval_accを記録
            if i == len(train_nums) - 1:
                val_loss, val_acc = model.evaluate(X_val,
                                               Y_val, 
                                               batch_size = BATCH,
                                               verbose=0)
                val_acc_hist.append(val_acc)
                sys.stdout.write('\r{0}/{1}  loss = {2:05f} acc = {3:05f} val_loss = {4:05f} val_acc = {5:05f}'.format(i * BATCH,len(train_nums) * BATCH,loss,acc,val_loss,val_acc))
                sys.stdout.flush()


        print('\r')
        
        #save_weight
        model.save_weights('cache/lstm1/timestep={0}/{1}/model_weights_{2}_{3:02d}.h5'.format(time_step,test_name,modelStr,ep))

    # モデルの構成を保存
    save_model(model, modelStr,'lstm1',test_name)

    #学習の様子をグラフで保存
    plt.clf()
    plt.plot(range(epoch), val_acc_hist, marker = '.', label='val_acc')
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('acc')
    if not os.path.exists('result/lstm1/timestep={0}/{1}'.format(time_step,test_name)):
        os.makedirs('result/lstm1/timestep={0}/{1}'.format(time_step,test_name))
    plt.savefig('result/lstm1/timestep={0}/{1}/trainplt.png'.format(time_step,test_name))



################## L S T M1  T E S T ##################################################################
def run_test_lstm1(modelStr,epoch,test_name):
    results = []

    X_test_paths, Y_test, img_paths = MyDatasetLoader.read_test_data_lstm(test_name, time_step)

    #端数は切っとく
    batch_num = len(X_test_paths[0])//BATCH
    Y_test = Y_test[:batch_num * BATCH]

    model = read_model(modelStr,epoch,'lstm1',test_name)


    #結果保存用のディレクトリ
    result_path = 'result/lstm1/timestep=%d/%s/ep%d'%(time_step,test_name,epoch)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

   
    print('start test (test_name : {0})'.format(test_name))

    test_result = []
    #バッチの数だけループ
    for i in range(len(X_test_paths[0])//BATCH): 

        #test
        X_test_batch = MyDatasetLoader.test_batch_create(X_test_paths, i, BATCH)
        
    
        result = model.predict_on_batch(X_test_batch)
        for r in result:
           test_result.append(r)


        sys.stdout.write('\r{0}/{1}'.format(i* BATCH,(len(X_test_paths[0])//BATCH)*BATCH))
        sys.stdout.flush()

    test_result_classes = []
    test_result = np.array(test_result,dtype=np.float32)
    test_result_classes = list(map(lambda x:1 if x > float(threshold) else 0,test_result[:,1])) 
    print(test_result_classes)


    #誤識別した画像を保存
    count = 0
    miss_count = 0
    if not os.path.exists(os.path.join(result_path,'img')):
        os.makedirs(os.path.join(result_path,'img'))

    for root, dirs, files in os.walk(os.path.join(result_path,'img/FP'), topdown=False):
         for name in files:
                os.remove(os.path.join(result_path,'img/FP', name))
    for root, dirs, files in os.walk(os.path.join(result_path,'img/FN'), topdown=False):
         for name in files:
                os.remove(os.path.join(result_path,'img/FN', name))

    for res in test_result_classes:
        if res != Y_test[count][1]:
            miss_count += 1
            image = cv2.imread(img_paths[count])
            if int(Y_test[count][1]) == 0:
                dir_path = os.path.join(result_path,'img/FP')
            else:
                dir_path = os.path.join(result_path,'img/FN')
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path)
                    
            head,tail = os.path.split(img_paths[count]) #tailは'0001.png'
            new_path = os.path.join(dir_path, tail)
            cv2.imwrite(new_path, image) 
        count += 1


    #ミスの数を表示
    print('miss_amount:{0}'.format(str(miss_count)))

    #F値などをtextで保存
    tp,tn,fp,fn = 0,0,0,0
    count = 0
    for res in test_result_classes:
        if res == Y_test[count][1]:
            if res == 0:
                tn += 1
            else:
                tp += 1
        else:
            if res == 0:
                fn += 1
            else:
                fp += 1
        count += 1

    if tp + fp != 0:
        accuracy = (tp + tn) / (tp + fn + fp + tn)
        precision = tp / (tp + fp)
        if tp + fn == 0:
            recall = 0
        else:
            recall = tp / (tp + fn)
        if recall + precision == 0:
            f_value = 0
        else:  
            f_value = (2 * recall * precision) / (recall + precision)

    f = open(os.path.join(result_path,'result.txt'.format(test_name)), 'w')
    f.write('\nTHRESHOLD : 0.5')
    f.write('\nTrue Negative  = {0:5d}  | False Negative = {1:5d}'.format(tn, fn)) 
    f.write('\nFalse Positive = {0:5d}  | True Positive  = {1:5d}\n'.format(fp, tp)) 
    f.write('\nAccuracy  = %01.4f' % accuracy)
    f.write('\nPrecision = %01.4f' % precision)
    f.write('\nRecall    = %01.4f' % recall)
    f.write('\nF_value   = %01.4f\n' % f_value)


    #csvファイルに結果を保存
    np.savetxt(os.path.join(result_path,'result.csv'),test_result,delimiter=',')
    np.savetxt(os.path.join(result_path,'seikai.csv'),Y_test,delimiter=',')


    #ROCカーブを計算、画像で保存
    fpr, tpr, thresholds = roc_curve(Y_test[:,1], test_result[:,1])
    roc_auc = auc(fpr, tpr)       

    plt.clf()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(result_path,'ROC.png'))

################## L S T M3  T R A I N ##################################################################
def run_train_lstm3(modelStr,epoch,test_name):
    model = make_model_lstm3()

    X_train_paths, Y_train = MyDatasetLoader.read_train_data_lstm(test_name, time_step)


    if not os.path.exists('cache/lstm3/timestep={0}'.format(time_step)):
        os.mkdir('cache/lstm3/timestep={0}'.format(time_step))

    if not os.path.exists('cache/lstm3/timestep={0}/{1}'.format(time_step,test_name)):
        os.mkdir('cache/lstm3/timestep={0}/{1}'.format(time_step,test_name))

    cp = ModelCheckpoint('cache/lstm3/timestep=%s/%s/model_weights_%s_{epoch:02d}.h5'%(time_step,test_name, modelStr), monitor='val_loss', save_best_only=False)


    print('start train (test_name : {0})'.format(test_name))


    #バッチの順番をシャッフル
    batch_nums = []
    for i in range(len(X_train_paths[0])//BATCH):
        batch_nums.append(i)
    random.shuffle(batch_nums)

    #validationとtrainにわける 99:1
    train_nums = batch_nums[: len(batch_nums) * 19 // 20]
    validation_nums = batch_nums[len(batch_nums) * 19 // 20:]


    #validation_data
    X_l_val, X_r_val, Y_val = [],[],[]
    for j in range(len(validation_nums)):
        X_val_batch, Y_val_batch = MyDatasetLoader.train_batch_create(X_train_paths, Y_train, validation_nums[j],BATCH)
        for l in X_val_batch[0]:
            X_l_val.append(l)
        for r in X_val_batch[1]:
            X_r_val.append(r)
        for y in Y_val_batch:
            Y_val.append(y)             
    X_val = [np.array(X_l_val,dtype=np.float32), np.array(X_r_val,dtype=np.float32)]
    Y_val = np.array(Y_val,dtype=np.uint8)


    #val_accの記録
    val_acc_hist = []
     
    #学習ループ
    for ep in range(epoch):
        print('Epoch {0}/{1}\r'.format(ep + 1, epoch))
        #trainの順番をシャッフル
        random.shuffle(train_nums)

        #バッチの数だけループ
        for i in range(len(train_nums)): 

            #train
            X_train_batch, Y_train_batch = MyDatasetLoader.train_batch_create(X_train_paths, Y_train, train_nums[i],BATCH)
      
            loss, acc = model.train_on_batch(X_train_batch,
                                                 Y_train_batch)

            #1000batchごとにvalidateしてみる
            if i % 1000 == 0:
                val_loss, val_acc = model.evaluate(X_val,
                                               Y_val, 
                                               batch_size = BATCH,
                                               verbose=0)
                sys.stdout.write('\r{0}/{1}  loss = {2:05f} acc = {3:05f} val_loss = {4:05f} val_acc = {5:05f}'.format(i * BATCH,len(train_nums) * BATCH,loss,acc,val_loss,val_acc))
                sys.stdout.flush()

            else:
                sys.stdout.write('\r{0}/{1}  loss = {2:05f} acc = {3:05f}'.format(i * BATCH,len(train_nums) * BATCH,loss,acc))
                sys.stdout.flush()

            #epochの最後のval_accを記録
            if i == len(train_nums) - 1:
                val_loss, val_acc = model.evaluate(X_val,
                                               Y_val, 
                                               batch_size = BATCH,
                                               verbose=0)
                val_acc_hist.append(val_acc)
                sys.stdout.write('\r{0}/{1}  loss = {2:05f} acc = {3:05f} val_loss = {4:05f} val_acc = {5:05f}'.format(i * BATCH,len(train_nums) * BATCH,loss,acc,val_loss,val_acc))
                sys.stdout.flush()


        print('\r')
        
        #save_weight
        model.save_weights('cache/lstm3/timestep={0}/{1}/model_weights_{2}_{3:02d}.h5'.format(time_step,test_name,modelStr,ep))

    # モデルの構成を保存
    save_model(model, modelStr,'lstm3',test_name)

    #学習の様子をグラフで保存
    plt.plot(range(epoch), val_acc_hist, marker = '.', label='val_acc')
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('acc')
    if not os.path.exists('result/lstm3/timestep={0}/{1}'.format(time_step,test_name)):
        os.makedirs('result/lstm3/timestep={0}/{1}'.format(time_step,test_name))
    plt.savefig('result/lstm3/timestep={0}/{1}/trainplt.png'.format(time_step,test_name))



################## L S T M3  T E S T ##################################################################
def run_test_lstm3(modelStr,epoch,test_name):
    results = []

    X_test_paths, Y_test, img_paths = MyDatasetLoader.read_test_data_lstm(test_name, time_step)

    #端数は切っとく
    batch_num = len(X_test_paths[0])//BATCH
    Y_test = Y_test[:batch_num * BATCH]

    model = read_model(modelStr,epoch,'lstm3',test_name)


    #結果保存用のディレクトリ
    result_path = 'result/lstm3/timestep=%d/%s/ep%d'%(time_step,test_name,epoch)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

   
    print('start test (test_name : {0})'.format(test_name))

    test_result = []
    #バッチの数だけループ
    for i in range(len(X_test_paths[0])//BATCH): 

        #test
        X_test_batch = MyDatasetLoader.test_batch_create(X_test_paths, i, BATCH)
        
    
        result = model.predict_on_batch(X_test_batch)
        for r in result:
           test_result.append(r)


        sys.stdout.write('\r{0}/{1}'.format(i* BATCH,(len(X_test_paths[0])//BATCH)*BATCH))
        sys.stdout.flush()

    test_result_classes = []
    test_result = np.array(test_result,dtype=np.float32)
    test_result_classes = list(map(lambda x:1 if x > float(threshold) else 0,test_result[:,1])) 
    print(test_result_classes)


    #誤識別した画像を保存
    count = 0
    miss_count = 0
    if not os.path.exists(os.path.join(result_path,'img')):
        os.makedirs(os.path.join(result_path,'img'))

    for root, dirs, files in os.walk(os.path.join(result_path,'img/FP'), topdown=False):
         for name in files:
                os.remove(os.path.join(result_path,'img/FP', name))
    for root, dirs, files in os.walk(os.path.join(result_path,'img/FN'), topdown=False):
         for name in files:
                os.remove(os.path.join(result_path,'img/FN', name))

    for res in test_result_classes:
        if res != Y_test[count][1]:
            miss_count += 1
            image = cv2.imread(img_paths[count])
            if int(Y_test[count][1]) == 0:
                dir_path = os.path.join(result_path,'img/FP')
            else:
                dir_path = os.path.join(result_path,'img/FN')
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path)
                    
            head,tail = os.path.split(img_paths[count]) #tailは'0001.png'
            new_path = os.path.join(dir_path, tail)
            cv2.imwrite(new_path, image) 
        count += 1


    #ミスの数を表示
    print('miss_amount:{0}'.format(str(miss_count)))

    #F値などをtextで保存
    tp,tn,fp,fn = 0,0,0,0
    count = 0
    for res in test_result_classes:
        if res == Y_test[count][1]:
            if res == 0:
                tn += 1
            else:
                tp += 1
        else:
            if res == 0:
                fn += 1
            else:
                fp += 1
        count += 1

    if tp + fp != 0:
        accuracy = (tp + tn) / (tp + fn + fp + tn)
        precision = tp / (tp + fp)
        if tp + fn == 0:
            recall = 0
        else:
            recall = tp / (tp + fn)
        if recall + precision == 0:
            f_value = 0
        else:  
            f_value = (2 * recall * precision) / (recall + precision)

    f = open(os.path.join(result_path,'result.txt'.format(test_name)), 'w')
    f.write('\nTHRESHOLD : 0.5')
    f.write('\nTrue Negative  = {0:5d}  | False Negative = {1:5d}'.format(tn, fn)) 
    f.write('\nFalse Positive = {0:5d}  | True Positive  = {1:5d}\n'.format(fp, tp)) 
    f.write('\nAccuracy  = %01.4f' % accuracy)
    f.write('\nPrecision = %01.4f' % precision)
    f.write('\nRecall    = %01.4f' % recall)
    f.write('\nF_value   = %01.4f\n' % f_value)


    #csvファイルに結果を保存
    np.savetxt(os.path.join(result_path,'result.csv'),test_result,delimiter=',')
    np.savetxt(os.path.join(result_path,'seikai.csv'),Y_test,delimiter=',')


    #ROCカーブを計算、画像で保存
    fpr, tpr, thresholds = roc_curve(Y_test[:,1], test_result[:,1])
    roc_auc = auc(fpr, tpr)       

    plt.clf()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(result_path,'ROC.png'))



################## L S T M  T R A I N  + F A C E ##################################################################
def run_train_lstm_face(modelStr,epoch,test_name):
    model = make_model_lstm_face()

    X_train_paths, X_facepos, Y_train = MyDatasetLoader.read_train_data_lstm_face(test_name, time_step)


    if not os.path.exists('cache/lstm_face/timestep={0}'.format(time_step)):
        os.mkdir('cache/lstm_face/timestep={0}'.format(time_step))

    if not os.path.exists('cache/lstm_face/timestep={0}/{1}'.format(time_step,test_name)):
        os.mkdir('cache/lstm_face/timestep={0}/{1}'.format(time_step,test_name))

    cp = ModelCheckpoint('cache/lstm_face/timestep=%s/%s/model_weights_%s_{epoch:02d}.h5'%(time_step,test_name, modelStr), monitor='val_loss', save_best_only=False)



    print('start train (test_name : {0})'.format(test_name))

    #学習ループ
    for ep in range(epoch):
        print('Epoch {0}/{1}\r'.format(ep + 1, epoch))

        #バッチの順番をシャッフル
        batch_nums = []
        for i in range(len(X_train_paths[0])//BATCH):
            batch_nums.append(i)
        random.shuffle(batch_nums)

        #validationとtrainにわける 9:1
        train_nums = batch_nums[: len(batch_nums) * 9 // 10]
        validation_nums = batch_nums[len(batch_nums) * 9 // 10:]


        #validation_data
        X_l_val, X_r_val, X_f_val, Y_val = [],[],[],[]
        for j in range(len(validation_nums)):
            X_val_batch, Y_val_batch = MyDatasetLoader.train_batch_create_face(X_train_paths, X_facepos, Y_train, validation_nums[j],BATCH)
            for l in X_val_batch[0]:
                X_l_val.append(l)
            for r in X_val_batch[1]:
                X_r_val.append(r)
            for f in X_val_batch[2]:
                X_f_val.append(f)
            for y in Y_val_batch:
                Y_val.append(y)             
        X_val = [np.array(X_l_val,dtype=np.float32), np.array(X_r_val,dtype=np.float32), np.array(X_f_val,dtype=np.float32)]
        Y_val = np.array(Y_val,dtype=np.uint8)



        #バッチの数だけループ
        for i in range(len(train_nums)): 

            #train
            X_train_batch, Y_train_batch = MyDatasetLoader.train_batch_create_face(X_train_paths, X_facepos, Y_train, train_nums[i],BATCH)
            
            loss, acc = model.train_on_batch(X_train_batch,
                                                 Y_train_batch)

            #validate
            if i == 0 or i == len(train_nums) - 1:
                val_loss, val_acc = model.evaluate(X_val,
                                               Y_val, 
                                               batch_size = BATCH,
                                               verbose=0)
                sys.stdout.write('\r{0}/{1}  loss = {2:05f} acc = {3:05f} val_loss = {4:05f} val_acc = {5:05f}'.format(i * BATCH,len(train_nums) * BATCH,loss,acc,val_loss,val_acc))
                sys.stdout.flush()
            else:
                sys.stdout.write('\r{0}/{1}  loss = {2:05f} acc = {3:05f}'.format(i * BATCH,len(train_nums) * BATCH,loss,acc))
                sys.stdout.flush()

        print('\r')
        
        #save_weight
        model.save_weights('cache/lstm_face/timestep={0}/{1}/model_weights_{2}_{3:02d}.h5'.format(time_step,test_name,modelStr,ep))

    # モデルの構成を保存
    save_model(model, modelStr,'lstm_face',test_name)


################## L S T M  T E S T + F A C E ##################################################################
def run_test_lstm_face(modelStr,epoch,test_name):
    results = []

    X_test_paths, X_facepos,  Y_test, img_paths = MyDatasetLoader.read_test_data_lstm_face(test_name, time_step)

    #端数は切っとく
    batch_num = len(X_test_paths[0])//BATCH
    Y_test = Y_test[:batch_num * BATCH]

    model = read_model(modelStr,epoch,'lstm_face',test_name)


    #結果保存用のディレクトリ
    result_path = 'result/lstm_face/timestep=%d/%s/ep%d'%(time_step,test_name,epoch)
    if not os.path.exists(result_path):
        os.makedirs(result_path)

   
    print('start test (test_name : {0})'.format(test_name))

    test_result = []
    #バッチの数だけループ
    for i in range(len(X_test_paths[0])//BATCH): 

        #test
        X_test_batch = MyDatasetLoader.test_batch_create_face(X_test_paths, X_facepos, i, BATCH)
        
    
        result = model.predict_on_batch(X_test_batch)
        for r in result:
           test_result.append(r)


        sys.stdout.write('\r{0}/{1}'.format(i* BATCH,(len(X_test_paths[0])//BATCH)*BATCH))
        sys.stdout.flush()

    test_result_classes = []
    test_result = np.array(test_result,dtype=np.float32)
    test_result_classes = list(map(lambda x:1 if x > float(threshold) else 0,test_result[:,1])) 
    print(test_result_classes)


    #誤識別した画像を保存
    count = 0
    miss_count = 0
    if not os.path.exists(os.path.join(result_path,'img')):
        os.makedirs(os.path.join(result_path,'img'))

    for root, dirs, files in os.walk(os.path.join(result_path,'img/FP'), topdown=False):
         for name in files:
                os.remove(os.path.join(result_path,'img/FP', name))
    for root, dirs, files in os.walk(os.path.join(result_path,'img/FN'), topdown=False):
         for name in files:
                os.remove(os.path.join(result_path,'img/FN', name))

    for res in test_result_classes:
        if res != Y_test[count][1]:
            miss_count += 1
            image = cv2.imread(img_paths[count])
            if int(Y_test[count][1]) == 0:
                dir_path = os.path.join(result_path,'img/FP')
            else:
                dir_path = os.path.join(result_path,'img/FN')
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path)
                    
            head,tail = os.path.split(img_paths[count]) #tailは'0001.png'
            new_path = os.path.join(dir_path, tail)
            cv2.imwrite(new_path, image) 
        count += 1


    #ミスの数を表示
    print('miss_amount:{0}'.format(str(miss_count)))

    #F値などをtextで保存
    tp,tn,fp,fn = 0,0,0,0
    count = 0
    for res in test_result_classes:
        if res == Y_test[count][1]:
            if res == 0:
                tn += 1
            else:
                tp += 1
        else:
            if res == 0:
                fn += 1
            else:
                fp += 1
        count += 1

    if tp + fp != 0:
        accuracy = (tp + tn) / (tp + fn + fp + tn)
        precision = tp / (tp + fp)
        if tp + fn == 0:
            recall = 0
        else:
            recall = tp / (tp + fn)
        if recall + precision == 0:
            f_value = 0
        else:  
            f_value = (2 * recall * precision) / (recall + precision)

    f = open(os.path.join(result_path,'result.txt'.format(test_name)), 'w')
    f.write('\nTHRESHOLD : 0.5')
    f.write('\nTrue Negative  = {0:5d}  | False Negative = {1:5d}'.format(tn, fn)) 
    f.write('\nFalse Positive = {0:5d}  | True Positive  = {1:5d}\n'.format(fp, tp)) 
    f.write('\nAccuracy  = %01.4f' % accuracy)
    f.write('\nPrecision = %01.4f' % precision)
    f.write('\nRecall    = %01.4f' % recall)
    f.write('\nF_value   = %01.4f\n' % f_value)


    #csvファイルに結果を保存
    np.savetxt(os.path.join(result_path,'result.csv'),test_result,delimiter=',')
    np.savetxt(os.path.join(result_path,'seikai.csv'),Y_test,delimiter=',')


    #ROCカーブを計算、画像で保存
    fpr, tpr, thresholds = roc_curve(Y_test[:,1], test_result[:,1])
    roc_auc = auc(fpr, tpr)       

    plt.clf()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(result_path,'ROC.png'))


################## CRF  T R A I N ##################################################################
def run_train_crf(modelStr,epoch,test_name):
    cnn_model = read_model(modelStr,20,'cnn',test_name)

    model = make_model_crf()

    X_train,Y_train, frame_nums = MyDatasetLoader.read_train_data_cnn(test_name)

    X_train_feature = cnn_model.predict(X_train,batch_size=128,verbose=1)

    X_train_class = list(map(lambda x: 1 if x > float(0.5) else 0,X_train_feature[:,1])) 


    X_train_class_seq, Y_train_seq = [],[]
    for i in range(len(X_train_class) - time_step - 1):
        X_train_class_seq.append(X_train_class[i : i + time_step])
        Y_train_seq.append(Y_train[i + time_step - 1])
    X_train_class_seq = np.array(X_train_class_seq,dtype=np.float32)      
    Y_train_seq = np.array(Y_train_seq,dtype=np.float32)
    

    if not os.path.exists('cache/crf/{0}'.format(test_name)):
        os.mkdir('cache/crf/{0}'.format(test_name))
    
    cp = ModelCheckpoint('cache/crf/%s/model_weights_%s_{epoch:02d}.h5'%(test_name, modelStr), monitor='val_loss', save_best_only=False)

    print('start train (test_name : {0})'.format(test_name))
    # train実行
    hist = model.fit(X_train_class_seq, Y_train_seq, batch_size=64,
              epochs=epoch,
              verbose=1,
              validation_split = 0.1,
              shuffle=True,
              callbacks=[cp])
    # モデルの構成を保存
    save_model(model, modelStr,'crf',test_name)

################## C R F  T E S T ##################################################################
def run_test_crf(modelStr,epoch,test_name):
    print('test_name : {0}'.format(test_name))
    results = []

    cnn_model = read_model(modelStr,20,'cnn',test_name)

    X_test, Y_test, img_paths = MyDatasetLoader.read_test_data_cnn(test_name)

    X_test_feature = cnn_model.predict(X_test,batch_size=128,verbose=1)

    img_nums = []
    for img_path in img_paths:
        path,ext = os.path.splitext(img_path)
        path,num = os.path.split(path)
        img_nums.append(int(num)) 

    model = read_model(modelStr,epoch,'crf',test_name)
    test_result = model.predict(X_test_feature,batch_size=128,verbose=1)

    test_result_classes = list(map(lambda x:1 if x > float(0.5) else 0,test_result[:,1])) 


    #結果保存用のディレクトリ
    result_path = 'result/crf/%s/ep%d'%(test_name, epoch)
    if not os.path.exists(result_path):
        os.makedirs(result_path)


    #誤識別した画像を保存
    count = 0
    miss_count = 0
    if not os.path.exists(os.path.join(result_path,'img')):
        os.makedirs(os.path.join(result_path,'img'))

    for root, dirs, files in os.walk(os.path.join(result_path,'img/FP'), topdown=False):
         for name in files:
                os.remove(os.path.join(result_path,'img/FP', name))
    for root, dirs, files in os.walk(os.path.join(result_path,'img/FN'), topdown=False):
         for name in files:
                os.remove(os.path.join(result_path,'img/FN', name))

    for res in test_result_classes:
        if res != Y_test[count][1]:
            miss_count += 1
            image = cv2.imread(img_paths[count])
            if int(Y_test[count][1]) == 0:
                dir_path = os.path.join(result_path,'img/FP')
            else:
                dir_path = os.path.join(result_path,'img/FN')
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path)
                    
            head,tail = os.path.split(img_paths[count]) #tailは'0001.png'
            new_path = os.path.join(dir_path, tail)
            cv2.imwrite(new_path, image) 
        count += 1


    #ミスの数を表示
    print('miss_amount:{0}'.format(str(miss_count)))


    #F値などをtextで保存
    tp,tn,fp,fn = 0,0,0,0
    count = 0
    for res in test_result_classes:
        if res == Y_test[count][1]:
            if res == 0:
                tn += 1
            else:
                tp += 1
        else:
            if res == 0:
                fn += 1
            else:
                fp += 1
        count += 1

    if tp + fp != 0:
        accuracy = (tp + tn) / (tp + fn + fp + tn)
        precision = tp / (tp + fp)
        if tp + fn == 0:
            recall = 0
        else:
            recall = tp / (tp + fn)
        if recall + precision == 0:
            f_value = 0
        else:  
            f_value = (2 * recall * precision) / (recall + precision)

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
    for i in range(len(Y_test[:,1])):
        result_pairs.append([Y_test[:,1][i], test_result[:,1][i]])

    result_pairs_save = []
    count = 0
    for i in range(img_nums[len(img_nums) - 1]):
        if i == img_nums[count]:
            result_pairs_save.append(result_pairs[count])
            count+=1
        else:
            result_pairs_save.append([0,0])

    np.savetxt(os.path.join(result_path,'result.csv'),result_pairs_save,delimiter=',')


    #ROCカーブを計算、画像で保存
    fpr, tpr, thresholds = roc_curve(Y_test[:,1], test_result[:,1])
    roc_auc = auc(fpr, tpr)       

    plt.clf()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(result_path,'ROC.png'))






if __name__ == '__main__':
    model = 'DEEPEC-NP'
    if len(sys.argv) < 2:
        print('not enough params')
        exit()

    run_type = sys.argv[1]

    test_names = ['Avec',
                  'Aziz', 
                  'Derek',
                  'Elle',
                  'Emma']
                  #'Hiyane',
                  #'Imaizumi',
                  #'James',
                  #'Kendall',
                  #'Kitazumi',
                  #'Liza',
                  #'Neil',
                  #'Ogawa',
                  #'Selena',
                  #'Shiraishi',
                  #'Taylor']

    #第3引数にtest_nameを指定した場合それに従って実行,指定なしの場合all
    if run_type == 'train_cnn':
        epoch = int(sys.argv[2])
        if len(sys.argv) == 4:   
            test_name = sys.argv[3]
            run_train_cnn(model,epoch,test_name)
        else:
            for test_name in test_names:
                run_train_cnn(model,epoch,test_name)

    elif run_type == 'test_cnn':
        epoch = int(sys.argv[2])
        if len(sys.argv) == 4:   
            test_name = sys.argv[3]
            run_test_cnn(model,epoch,test_name)     
        else:
            for test_name in test_names:
                run_test_cnn(model,epoch,test_name) 

    elif run_type == 'train_lstm':
        epoch = int(sys.argv[2])
        if len(sys.argv) == 4:   
            test_name = sys.argv[3]
            run_train_lstm(model,epoch,test_name)
        else:
            for test_name in test_names:
                run_train_lstm(model,epoch,test_name)

    elif run_type == 'test_lstm':
        epoch = int(sys.argv[2])
        if len(sys.argv) == 4:   
            test_name = sys.argv[3]
            run_test_lstm(model,epoch,test_name)     
        else:
            for test_name in test_names:
                run_test_lstm(model,epoch,test_name) 

    elif run_type == 'train_lstm1':
        epoch = int(sys.argv[2])
        if len(sys.argv) == 4:   
            test_name = sys.argv[3]
            run_train_lstm1(model,epoch,test_name)
        else:
            for test_name in test_names:
                run_train_lstm1(model,epoch,test_name)

    elif run_type == 'test_lstm1':
        epoch = int(sys.argv[2])
        if len(sys.argv) == 4:   
            test_name = sys.argv[3]
            run_test_lstm1(model,epoch,test_name)     
        else:
            for test_name in test_names:
                run_test_lstm1(model,epoch,test_name)

    elif run_type == 'train_lstm_face':
        epoch = int(sys.argv[2])
        if len(sys.argv) == 4:   
            test_name = sys.argv[3]
            run_train_lstm_face(model,epoch,test_name)
        else:
            for test_name in test_names:
                run_train_lstm_face(model,epoch,test_name)

    elif run_type == 'test_lstm_face':
        epoch = int(sys.argv[2])
        if len(sys.argv) == 4:   
            test_name = sys.argv[3]
            run_test_lstm_face(model,epoch,test_name)     
        else:
            for test_name in test_names:
                run_test_lstm_face(model,epoch,test_name) 

    elif run_type == 'train_crf':
        epoch = int(sys.argv[2])
        if len(sys.argv) == 4:   
            test_name = sys.argv[3]
            run_train_crf(model,epoch,test_name)
        else:
            for test_name in test_names:
                run_train_crf(model,epoch,test_name)

    elif run_type == 'test_crf':
        epoch = int(sys.argv[2])
        if len(sys.argv) == 4:   
            test_name = sys.argv[3]
            run_test_crf(model,epoch,test_name)     
        else:
            for test_name in test_names:
                run_test_crf(model,epoch,test_name)

    elif run_type == 'all':
        for test_name in test_names:
            run_train_cnn(model,20,test_name)
            #run_test_cnn(model,10,test_name)
        for test_name in test_names:
            run_train_lstm(model,20,test_name)
            #run_test_lstm(model,1,test_name)    
            #run_test_lstm(model,10,test_name)   


    else:
        print('invalid format')





