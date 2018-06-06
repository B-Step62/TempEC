import os, glob, cv2, sys
import pandas as pd
import numpy as np
import codecs
import re
import csv
import random

from keras.utils import np_utils
from numpy.random import permutation

img_rows, img_cols = 36, 60 #画像の縦横サイズ

def GCN(img):
        img = np.array(img, dtype=np.float32)
        # 正規化(GCN)実行
        img -= np.mean(img)
        if np.std(img) != 0:
           img /= np.std(img)
        return img

def augument(img):
        left = (random.randrange(3)) // 2
        right = (random.randrange(3)) // 2
        up = (random.randrange(3)) // 2
        down = (random.randrange(3)) // 2
        
        img = img[0+up : 36-down, 0+left: 60-right] 
        img = cv2.resize(img, (60, 36))
        return img
        


class MyDatasetLoader:
    @staticmethod
    def read_train_data_cnn(test_name):
        lefts, rights, labels, frame_nums = [], [], [], [] 

        train_dirs = glob.glob('dataset/images/*')

        for train_dir in train_dirs:

            #testデータを省く
            if train_dir.find(test_name) > -1:
                continue

            top, train_name = os.path.split(train_dir)

            #csvの読み込み
            csv_path = 'dataset/labels/{0}.csv'.format(train_name)
            if not os.path.exists(csv_path):
                #print('!Csv file --{0}-- is not found.'.format(csv_path))
                continue             
            f = open(csv_path, 'r')
            reader = csv.reader(f)
            train_dir_labels = []
            for row in reader:
                for i in row:
                    train_dir_labels.append(int(float(i)))


            #画像の読み込み
            #print('Read {0} images as train data... '.format(train_name))
            for i in range(30000):
                if os.path.exists(os.path.join(train_dir,'left/{0}.png'.format(i))):
                    if os.path.exists(os.path.join(train_dir,'right/{0}.png'.format(i))):
                        left_image = cv2.imread(os.path.join(train_dir,'left/{0}.png'.format(i)), 0)
                        if not left_image is None:
                            right_image = cv2.imread(os.path.join(train_dir,'right/{0}.png'.format(i)), 0)

                            lefts.append([GCN(left_image)])
                            rights.append([GCN(right_image)])
                            labels.append(train_dir_labels[i - 1])
                            frame_nums.append(i)

            if len(lefts) < 1:
                print('!Not enough images.\n')
                continue
                
            
        lefts = np.array(lefts, dtype=np.float32)        
        rights = np.array(rights, dtype=np.float32)                
        lefts = lefts.transpose((0,2,3,1))
        rights = rights.transpose((0,2,3,1))
        labels = np.array(labels, dtype=np.uint8)
        labels = np_utils.to_categorical(labels, 2)

        #ランダムに並び変え
        #perm    = permutation(len(labels))
        #lefts   = lefts[perm]
        #rights  = rights[perm]
        #labels = labels[perm]
        return [lefts,rights],labels, frame_nums

            
    @staticmethod
    def read_test_data_cnn(test_name,time_step):
        lefts,rights,labels,left_paths,frame_nums = [], [], [], [], []

        
        #csvの読み込み
        csv_path = 'dataset/labels/{0}.csv'.format(test_name)
        if not os.path.exists(csv_path):
            #print('!Csv file --{0}-- is not found.'.format(csv_path))
            sys.exit() 
        f = open(csv_path, 'r')
        reader = csv.reader(f)
        test_labels = []
        for row in reader:
            for i in row:
                test_labels.append(int(float(i)))


        #画像の読み込み
        #print('Read {0} images as test data... '.format(test_name))
        test_dir = 'dataset/images/{0}'.format(test_name)
        for i in range(30000):
            if os.path.exists(os.path.join(test_dir,'left/{0}.png'.format(i))):
                if os.path.exists(os.path.join(test_dir,'right/{0}.png'.format(i))):
                    left_image = cv2.imread(os.path.join(test_dir,'left/{0}.png'.format(i)), 0)
                    if not left_image is None:
                        right_image = cv2.imread(os.path.join(test_dir,'right/{0}.png'.format(i)), 0)

                        lefts.append([GCN(left_image)])
                        rights.append([GCN(right_image)])
                        labels.append(test_labels[i - 1])
                        left_paths.append(os.path.join(test_dir,'left/{0}.png'.format(i)))
                        frame_nums.append(i)

        if len(lefts) < 1:
            print('!Not enough images.')
            sys.exit()

        dataXlefts, dataXrights, dataY, dataleft_paths = [], [], [], []
        for i in range(len(lefts) - time_step - 1):
            if frame_nums[i + time_step - 1] - frame_nums[i] < time_step :     #途中で途切れていたら使えない
                dataXlefts.append(lefts[i + time_step - 1])
                dataXrights.append(rights[i + time_step - 1])
                dataY.append(labels[i + time_step - 1])
                dataleft_paths.append(left_paths[i + time_step - 1])
                    
                            
        dataXlefts   = np.array(dataXlefts,dtype=np.float32)
        dataXrights  = np.array(dataXrights,dtype=np.float32)
        dataXlefts = dataXlefts.transpose((0,2,3,1))
        dataXrights = dataXrights.transpose((0,2,3,1))
        dataY = np.array(dataY,dtype=np.uint8)
        dataY = np_utils.to_categorical(dataY, 2)            

        return [dataXlefts,dataXrights],dataY,dataleft_paths


    @staticmethod
    def read_train_data_lstm(test_name, time_step):
        dataXlefts, dataXrights, dataY, dataNums = [], [], [], []

        train_dirs = glob.glob('dataset/images/*')


        for train_dir in train_dirs:

            #testデータを省く
            if train_dir.find(test_name) > -1:
                continue


            top, train_name = os.path.split(train_dir)

            #csvの読み込み
            csv_path = 'dataset/labels/{0}.csv'.format(train_name)
            if not os.path.exists(csv_path):
                #print('!Csv file --{0}-- is not found.'.format(csv_path))
                continue            
            f = open(csv_path, 'r')
            reader = csv.reader(f)
            train_dir_labels = []
            for row in reader:
                for i in row:
                    train_dir_labels.append(int(float(i)))

            #while(len(dataXlefts) < 20000):
  
            #画像の読み込み
            #print('Read {0} images as train data... '.format(train_name))
            lefts, rights, labels, frame_nums = [], [], [], []
            for i in range(30000):
                if os.path.exists(os.path.join(train_dir,'left/{0}.png'.format(i))):
                    if os.path.exists(os.path.join(train_dir,'right/{0}.png'.format(i))):
                        #left_image = cv2.imread(os.path.join(train_dir,'left/{0}.png'.format(i)), 0)
                        #if not left_image is None:
                        #right_image = cv2.imread(os.path.join(train_dir,'right/{0}.png'.format(i)), 0)
   
                        lefts.append(os.path.join(train_dir,'left/{0}.png'.format(i)))    #[GCN(left_image)])
                        rights.append(os.path.join(train_dir,'right/{0}.png'.format(i)))   #[GCN(right_image)])
                        labels.append(train_dir_labels[i - 1])
                        frame_nums.append(i)

            if len(lefts) < time_step:
                print('!Not enough images.\n')
                continue
                
            
            #lefts = np.array(lefts, dtype=np.float32)        
            #rights = np.array(rights, dtype=np.float32)                
            #lefts = lefts.transpose((0,2,3,1))
            #rights = rights.transpose((0,2,3,1))
            #labels = np.array(labels, dtype=np.uint8)
            #labels = np_utils.to_categorical(labels, 2)

 
                #time_stepに応じて複数重ねる
            for i in range(len(lefts) - time_step - 1):
                if frame_nums[i + time_step - 1] - frame_nums[i] < time_step + 0:     #途中である程度以上途切れていたら使えない
                    dataXlefts.append(lefts[i : i + time_step])
                    dataXrights.append(rights[i : i + time_step])
                    dataY.append(labels[i + time_step - 1])
                    dataNums.append(frame_nums[i : i + time_step])

       
        #dataXlefts = np.array(dataXlefts,dtype=np.string)
        #dataXrights = np.array(dataXrights,dtype=np.string)      
        #dataY = np.array(dataY,dtype=np.float32)
        #dataNums = np.array(dataNums,dtype=np.float32)

        alsXlefts,alsXrights,alsY,alsNums = [],[],[],[]

        #ランダムに並び替え
        perm = permutation(len(dataY))
        for i in range(len(dataY)):
            alsXlefts.append(dataXlefts[perm[i]])
            alsXrights.append(dataXrights[perm[i]])
            alsY.append(dataY[perm[i]])
            alsNums.append(dataNums[perm[i]])

        #print(alsNums)

        return [alsXlefts, alsXrights],alsY

    @staticmethod
    def read_train_data_lstm_face(test_name, time_step):
        dataXlefts, dataXrights, dataXfaces, dataY, dataNums = [], [], [], [], []

        train_dirs = glob.glob('dataset/images/*')

        for train_dir in train_dirs:

            #testデータを省く
            if train_dir.find(test_name) > -1:
                continue

            top, train_name = os.path.split(train_dir)

            #csvの読み込み
            csv_path = 'dataset/labels/{0}.csv'.format(train_name)
            if not os.path.exists(csv_path):
                #print('!Csv file --{0}-- is not found.'.format(csv_path))
                continue            
            f = open(csv_path, 'r')
            reader = csv.reader(f)
            train_dir_labels = []
            for row in reader:
                for i in row:
                    train_dir_labels.append(int(float(i)))

            face_path = 'dataset/head_pose_vec/{0}.csv'.format(train_name)
            f2 = open(face_path, 'r')
            reader = csv.reader(f2)
            X_faces = []
            for row in reader:
                X_faces.append(row)

            #画像の読み込み
            #print('Read {0} images as train data... '.format(train_name))
            lefts, rights, faces, labels, frame_nums = [], [],[], [], []
            for i in range(30000):
                if os.path.exists(os.path.join(train_dir,'left/{0}.png'.format(i))):
                    if os.path.exists(os.path.join(train_dir,'right/{0}.png'.format(i))):
                        #left_image = cv2.imread(os.path.join(train_dir,'left/{0}.png'.format(i)), 0)
                        #if not left_image is None:
                        #right_image = cv2.imread(os.path.join(train_dir,'right/{0}.png'.format(i)), 0)

                        lefts.append(os.path.join(train_dir,'left/{0}.png'.format(i)))    #[GCN(left_image)])
                        rights.append(os.path.join(train_dir,'right/{0}.png'.format(i)))   #[GCN(right_image)])
                        labels.append(train_dir_labels[i - 1])
            
                        faces.append(X_faces[i - 1])

                        frame_nums.append(i)

            if len(lefts) < time_step:
                print('!Not enough images.\n')
                continue
                
            
            #lefts = np.array(lefts, dtype=np.float32)        
            #rights = np.array(rights, dtype=np.float32)                
            #lefts = lefts.transpose((0,2,3,1))
            #rights = rights.transpose((0,2,3,1))
            #labels = np.array(labels, dtype=np.uint8)
            #labels = np_utils.to_categorical(labels, 2)


            #time_stepに応じて複数重ねる
            for i in range(len(lefts) - time_step - 1):
                if frame_nums[i + time_step - 1] - frame_nums[i] < time_step + 0:     #途中である程度以上途切れていたら使えない
                    dataXlefts.append(lefts[i : i + time_step])
                    dataXrights.append(rights[i : i + time_step])
                    dataXfaces.append(faces[i : i + time_step ])
                    dataY.append(labels[i + time_step - 1])
                    dataNums.append(frame_nums[i : i + time_step])

       
        #dataXlefts = np.array(dataXlefts,dtype=np.string)
        #dataXrights = np.array(dataXrights,dtype=np.string)      
        #dataY = np.array(dataY,dtype=np.float32)
        #dataNums = np.array(dataNums,dtype=np.float32)

        alsXlefts,alsXrights,alsXfaces, alsY,alsNums = [],[],[],[],[]

        #ランダムに並び替え
        #perm = permutation(len(dataY))
        #for i in range(len(dataY)):
        #    alsXlefts.append(dataXlefts[perm[i]])
        #    alsXrights.append(dataXrights[perm[i]])
        #    alsXfaces.append(dataXfaces[perm[i]])
        #    alsY.append(dataY[perm[i]])
        #    alsNums.append(dataNums[perm[i]])

        #print(alsNums)

        return [dataXlefts, dataXrights], dataXfaces, dataY  #[alsXlefts, alsXrights],alsY

    @staticmethod
    def data_generator(X_train_paths,Y_train,time_step):
        X_left_batch, X_right_batch, Y_batch = [],[],[]

        #batch_num = lstm_data_num // 64


        for i in range(len(Y_train)): #batch_num * 64):
            X_left_paths = X_train_paths[0][i]
            X_right_paths = X_train_paths[1][i]
            X_lefts, X_rights = [],[]

            for left_path in X_left_paths:
                X_lefts.append(GCN(cv2.imread(left_path)))
            X_lefts = np.array(X_lefts,dtype=np.float32)
            X_lefts = X_lefts.transpose((0,2,3,1))

            for right_path in X_right_paths:
                X_rights.append(GCN(cv2.imread(right_path)))
            X_rights = np.array(X_rights,dtype=np.float32)
            X_rights = X_rights.transpose((0,2,3,1))
             
            X_left_batch.append(X_lefts)
            X_right_batch.append(X_rights)
            Y_batch.append(Y_train[i]) 

            if len(Y_batch) == 64:
                X_left_batch = np.array(X_left_batch,dtype=np.float32)
                X_right_batch = np.array(X_right_batch,dtype=np.float32)
                Y_batch = np.array(Y_batch,dtype=np.uint8)

                Y_batch= np_utils.to_categorical(Y_batch, 2)
   
                #ランダムに並び替え
                perm = permutation(64)
                X_left_batch = X_left_batch[perm]
                X_right_batch = X_right_batch[perm]
                Y_batch = Y_batch[perm]

                X_train = [X_left_batch, X_right_batch]
                
              
                yield X_train, Y_batch
                X_left_batch, X_right_batch, Y_batch = [],[],[]
            
            if i == len(X_train_paths) - 1:
                yield [X_left_batch, X_right_batch], Y_batch

    @staticmethod
    def train_batch_create(X_train_paths,Y_train,batch_num,BATCH):
        X_left_batch, X_right_batch, Y_batch = [],[],[]

        for i in range(BATCH):

            X_left_paths = X_train_paths[0][i + batch_num * BATCH]
            X_right_paths = X_train_paths[1][i + batch_num * BATCH]
            X_lefts, X_rights = [],[]
        
            for left_path in X_left_paths:
                X_lefts.append([GCN(cv2.imread(left_path,0))])
            X_lefts = np.array(X_lefts,dtype=np.float32)

            for right_path in X_right_paths:
                X_rights.append([GCN(cv2.imread(right_path,0))])
            X_rights = np.array(X_rights,dtype=np.float32)

             
            X_left_batch.append(X_lefts)
            X_right_batch.append(X_rights)
            Y_batch.append(Y_train[i + batch_num * BATCH]) 

            #if batch_num == 100:
            #    img = cv2.hconcat([cv2.imread(X_left_paths[0],0),cv2.imread(X_right_paths[0],0)])
            #    for num in range(9):
            #        img = cv2.vconcat([img, cv2.hconcat([cv2.imread(X_left_paths[num+1],0), cv2.imread(X_right_paths[num+1],0)])])

            #    pattern = r'([0-9]{4,5})'
            #    frame_num = int(re.findall(pattern,X_left_paths[9])[0])
            #    cv2.imwrite('test/{0}_{1}.png'.format(frame_num,Y_train[i + batch_num * 64]),img)        

        X_left_batch = np.array(X_left_batch,dtype=np.float32)
        X_left_batch = X_left_batch.transpose((0,1,3,4,2))
        X_right_batch = np.array(X_right_batch,dtype=np.float32)
        X_right_batch = X_right_batch.transpose((0,1,3,4,2))
        Y_batch = np.array(Y_batch,dtype=np.uint8)

        Y_batch= np_utils.to_categorical(Y_batch, 2)
   
        #ランダムに並び替え
        perm = permutation(BATCH)
        X_left_batch = X_left_batch[perm]
        X_right_batch = X_right_batch[perm]
        Y_batch = Y_batch[perm]
            
                
              
        return [X_left_batch, X_right_batch], Y_batch

    @staticmethod
    def train_batch_create_face(X_train_paths,X_facepos,Y_train,batch_num,BATCH):
        X_left_batch, X_right_batch, X_facepos_batch, Y_batch = [],[],[],[]

        for i in range(BATCH):

            X_left_paths = X_train_paths[0][i + batch_num * BATCH]
            X_right_paths = X_train_paths[1][i + batch_num * BATCH]
            X_lefts, X_rights = [],[]
        
            for left_path in X_left_paths:
                X_lefts.append([GCN(cv2.imread(left_path,0))])
            X_lefts = np.array(X_lefts,dtype=np.float32)

            for right_path in X_right_paths:
                X_rights.append([GCN(cv2.imread(right_path,0))])
            X_rights = np.array(X_rights,dtype=np.float32)

             
            X_left_batch.append(X_lefts)
            X_right_batch.append(X_rights)
            X_facepos_batch.append(X_facepos[i + batch_num * BATCH])
            Y_batch.append(Y_train[i + batch_num * BATCH]) 

            #if batch_num == 100:
            #    img = cv2.hconcat([cv2.imread(X_left_paths[0],0),cv2.imread(X_right_paths[0],0)])
            #    for num in range(9):
            #        img = cv2.vconcat([img, cv2.hconcat([cv2.imread(X_left_paths[num+1],0), cv2.imread(X_right_paths[num+1],0)])])

            #    pattern = r'([0-9]{4,5})'
            #    frame_num = int(re.findall(pattern,X_left_paths[9])[0])
            #    cv2.imwrite('test/{0}_{1}.png'.format(frame_num,Y_train[i + batch_num * 64]),img)        

        X_left_batch = np.array(X_left_batch,dtype=np.float32)
        X_left_batch = X_left_batch.transpose((0,1,3,4,2))
        X_right_batch = np.array(X_right_batch,dtype=np.float32)
        X_right_batch = X_right_batch.transpose((0,1,3,4,2))
        X_facepos_batch = np.array(X_facepos_batch,dtype=np.float32)
        Y_batch = np.array(Y_batch,dtype=np.uint8)

        Y_batch= np_utils.to_categorical(Y_batch, 2)
   
        #ランダムに並び替え
        perm = permutation(BATCH)
        X_left_batch = X_left_batch[perm]
        X_right_batch = X_right_batch[perm]
        X_facepos_batch = X_facepos_batch[perm]
        Y_batch = Y_batch[perm]
            
                
              
        return [X_left_batch, X_right_batch, X_facepos_batch], Y_batch



    @staticmethod
    def read_test_data_lstm(test_name, time_step):
        dataXlefts, dataXrights, dataY, dataNums, data_paths = [], [], [], [], []

        test_dir = 'dataset/images/{0}'.format(test_name)

        #csvの読み込み
        csv_path = 'dataset/labels/{0}.csv'.format(test_name)
        if not os.path.exists(csv_path):
            #print('!Csv file --{0}-- is not found.'.format(csv_path))
            sys.exit()           
        f = open(csv_path, 'r')
        reader = csv.reader(f)
        test_dir_labels = []
        for row in reader:
            for i in row:
                test_dir_labels.append(int(float(i)))


        #画像の読み込み
        #print('Read {0} images as test data... '.format(test_name))
        lefts, rights, labels, frame_nums, paths = [], [], [], [], []
        for i in range(30000):
            if os.path.exists(os.path.join(test_dir,'left/{0}.png'.format(i))):
                if os.path.exists(os.path.join(test_dir,'right/{0}.png'.format(i))):

                    lefts.append(os.path.join(test_dir,'left/{0}.png'.format(i)))    #[GCN(left_image)])
                    rights.append(os.path.join(test_dir,'right/{0}.png'.format(i)))   #[GCN(right_image)])
                    labels.append(test_dir_labels[i - 1])
                    frame_nums.append(i)
                    paths.append(os.path.join(test_dir,'left/{0}.png'.format(i)))

        if len(lefts) < time_step:
             print('!Not enough images.\n')
             sys.exit()
                
        labels = np.array(labels, dtype=np.uint8)
        labels = np_utils.to_categorical(labels, 2)


        #time_stepに応じて複数重ねる
        for i in range(len(lefts) - time_step - 1):
            if frame_nums[i + time_step - 1] - frame_nums[i] < time_step + 0:     #途中である程度以上途切れていたら使えない
                dataXlefts.append(lefts[i : i + time_step])
                dataXrights.append(rights[i : i + time_step])
                dataY.append(labels[i + time_step - 1])
                data_paths.append(paths[i + time_step - 1])

    
        dataY = np.array(dataY,dtype=np.float32)

        return [dataXlefts, dataXrights], dataY, data_paths


    @staticmethod
    def test_batch_create(X_test_paths,batch_num,BATCH):
        X_left_batch, X_right_batch,= [],[]

        for i in range(BATCH):

            X_left_paths = X_test_paths[0][i + batch_num * BATCH]
            X_right_paths = X_test_paths[1][i + batch_num * BATCH]
            X_lefts, X_rights = [],[]
        
            for left_path in X_left_paths:
                X_lefts.append([GCN(cv2.imread(left_path,0))])
            X_lefts = np.array(X_lefts,dtype=np.float32)

            for right_path in X_right_paths:
                X_rights.append([GCN(cv2.imread(right_path,0))])
            X_rights = np.array(X_rights,dtype=np.float32)

             
            X_left_batch.append(X_lefts)
            X_right_batch.append(X_rights)

            #if batch_num == 100:
            #    img = cv2.hconcat([cv2.imread(X_left_paths[0],0),cv2.imread(X_right_paths[0],0)])
            #    for num in range(9):
            #        img = cv2.vconcat([img, cv2.hconcat([cv2.imread(X_left_paths[num+1],0), cv2.imread(X_right_paths[num+1],0)])])

            #    pattern = r'([0-9]{4,5})'
            #    frame_num = int(re.findall(pattern,X_left_paths[9])[0])
            #    cv2.imwrite('test/{0}_{1}.png'.format(frame_num,Y_test[i + batch_num * BATCH]),img)        

        X_left_batch = np.array(X_left_batch,dtype=np.float32)
        X_left_batch = X_left_batch.transpose((0,1,3,4,2))
        X_right_batch = np.array(X_right_batch,dtype=np.float32)
        X_right_batch = X_right_batch.transpose((0,1,3,4,2))


                
              
        return [X_left_batch, X_right_batch]


    @staticmethod
    def read_test_data_lstm_face(test_name, time_step):
        dataXlefts, dataXrights, dataXfaces, dataY, dataNums, data_paths = [], [], [], [], [], []

        test_dir = 'dataset/images/{0}'.format(test_name)

        #csvの読み込み
        csv_path = 'dataset/labels/{0}.csv'.format(test_name)
        if not os.path.exists(csv_path):
            #print('!Csv file --{0}-- is not found.'.format(csv_path))
            sys.exit()           
        f = open(csv_path, 'r')
        reader = csv.reader(f)
        test_dir_labels = []
        for row in reader:
            for i in row:
                test_dir_labels.append(int(float(i)))

        face_path = 'dataset/head_pose_vec/{0}.csv'.format(test_name)
        f2 = open(face_path, 'r')
        reader = csv.reader(f2)
        X_faces = []
        for row in reader:
            X_faces.append(row)


        #画像の読み込み
        #print('Read {0} images as test data... '.format(test_name))
        lefts, rights, faces, labels, frame_nums, paths = [], [], [], [], [], []
        for i in range(30000):
            if os.path.exists(os.path.join(test_dir,'left/{0}.png'.format(i))):
                if os.path.exists(os.path.join(test_dir,'right/{0}.png'.format(i))):

                    lefts.append(os.path.join(test_dir,'left/{0}.png'.format(i)))    #[GCN(left_image)])
                    rights.append(os.path.join(test_dir,'right/{0}.png'.format(i)))   #[GCN(right_image)])
                    faces.append(X_faces[i-1])
                    labels.append(test_dir_labels[i - 1])
                    frame_nums.append(i)
                    paths.append(os.path.join(test_dir,'left/{0}.png'.format(i)))

        if len(lefts) < time_step:
             print('!Not enough images.\n')
             sys.exit()
                
        labels = np.array(labels, dtype=np.uint8)
        labels = np_utils.to_categorical(labels, 2)


        #time_stepに応じて複数重ねる
        for i in range(len(lefts) - time_step - 1):
            if frame_nums[i + time_step - 1] - frame_nums[i] < time_step + 0:     #途中である程度以上途切れていたら使えない
                dataXlefts.append(lefts[i : i + time_step])
                dataXrights.append(rights[i : i + time_step])
                dataXfaces.append(faces[i : i + time_step])
                dataY.append(labels[i + time_step - 1])
                data_paths.append(paths[i + time_step - 1])

    
        dataY = np.array(dataY,dtype=np.float32)

        return [dataXlefts, dataXrights], dataXfaces, dataY, data_paths


    @staticmethod
    def test_batch_create_face(X_test_paths,X_facepos,batch_num,BATCH):
        X_left_batch, X_right_batch, X_faces_batch= [],[],[]

        for i in range(BATCH):

            X_left_paths = X_test_paths[0][i + batch_num * BATCH]
            X_right_paths = X_test_paths[1][i + batch_num * BATCH]
            X_lefts, X_rights = [],[]
        
            for left_path in X_left_paths:
                X_lefts.append([GCN(cv2.imread(left_path,0))])
            X_lefts = np.array(X_lefts,dtype=np.float32)

            for right_path in X_right_paths:
                X_rights.append([GCN(cv2.imread(right_path,0))])
            X_rights = np.array(X_rights,dtype=np.float32)

             
            X_left_batch.append(X_lefts)
            X_right_batch.append(X_rights)
            X_faces_batch.append(X_facepos[i + batch_num * BATCH])

            #if batch_num == 100:
            #    img = cv2.hconcat([cv2.imread(X_left_paths[0],0),cv2.imread(X_right_paths[0],0)])
            #    for num in range(9):
            #        img = cv2.vconcat([img, cv2.hconcat([cv2.imread(X_left_paths[num+1],0), cv2.imread(X_right_paths[num+1],0)])])

            #    pattern = r'([0-9]{4,5})'
            #    frame_num = int(re.findall(pattern,X_left_paths[9])[0])
            #    cv2.imwrite('test/{0}_{1}.png'.format(frame_num,Y_test[i + batch_num * BATCH]),img)        

        X_left_batch = np.array(X_left_batch,dtype=np.float32)
        X_left_batch = X_left_batch.transpose((0,1,3,4,2))
        X_right_batch = np.array(X_right_batch,dtype=np.float32)
        X_right_batch = X_right_batch.transpose((0,1,3,4,2))
        X_faces_batch = np.array(X_faces_batch,dtype=np.float32)

                
              
        return [X_left_batch, X_right_batch, X_faces_batch]




