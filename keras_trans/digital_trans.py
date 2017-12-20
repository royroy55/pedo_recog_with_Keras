# coding: utf-8
# mnistで学習済みのmodelを使ってデジタル数字を転移学習
# 学習済みモデルはmnistのデータに対して

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

import sys
sys.path.append('/usr/local/lib/python3.6/site-packages')
import cv2
import os
import glob
import seaborn
import pandas
from keras.datasets import mnist
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from sklearn.model_selection import train_test_split
from PIL import Image
from matplotlib import pyplot as plt


# 分類クラス
batch_size = 128
nb_classes = 10
nb_epoch = 100

img_rows = 28
img_cols = 28

image_list = []
label_list = []

for dir in os.listdir("test_image"):
    if dir == ".DS_Store":
        continue

    dir1 = "test_image/" + dir
    # フォルダ"0"のラベルは"0"、フォルダ"1"のラベルは"1", ... , フォルダ"9"のラベルは"9"
    label = dir

    for file in os.listdir(dir1):
        if file != ".DS_Store":
        # Macだと、if file != ".DS_Store": 　なのかもしれない。。。
            # 配列label_listに正解ラベルを追加
            label_list.append(label)
            filepath = dir1 + "/" + file
            # 画像を読み込み、グレースケールに変換し、28x28pixelに変換し、numpy配列へ変換する。
            # 画像の1ピクセルは、それぞれが0-255の数値。
            image = np.array(Image.open(filepath).convert("L").resize((28, 28)))
            # print(filepath)
            # さらにフラットな1次元配列に変換。
            image = image.reshape(1, 784).astype("float32")[0]
            # 出来上がった配列をimage_listに追加。
            image_list.append(image / 255.)

image_list = np.array(image_list)
label_list = np.array(label_list)

X_train, X_test, y_train, y_test = train_test_split(image_list, label_list, test_size=0.3, random_state=111)

X_train = np.reshape(X_train, (len(X_train),  img_rows, img_cols, 1))
X_test = np.reshape(X_test, (len(X_test), img_rows, img_cols, 1))

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

'''
files = [r.split('/')[-1] for r in glob.glob('/Users/royroy55/deeplearning/keras_trans/test_image/*.png')]

for file in files:
    test_label.append(int(file[0]))

print(test_label)
quit()
'''

model_filename = 'model/cnn_model.json'
weights_filename = 'model/mnist_model.hdf5'

json_string = open(model_filename).read()
model = model_from_json(json_string)
print('model\'s summary')
model.summary()
model.load_weights(weights_filename)

'''
# MNISTのデータでテストするとき
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = np.reshape(X_train, (len(X_train),  img_rows, img_cols, 1))
X_test = np.reshape(X_test, (len(X_test), img_rows, img_cols, 1))
X_train = X_train.astype('float32')
X_test  = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
'''


model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])


plt.plot(history.history['acc'], label='train_acc')
plt.plot(history.history['val_acc'], label='test_acc')

plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig('acc_graph.png')
plt.clf()

plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='test_loss')

plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig('loss_graph.png')
