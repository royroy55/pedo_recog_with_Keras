# coding: utf-8

import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import os
import csv
import cv2

image_dir = '/Users/royroy55/deeplearning/keras_trans/test_image/'

datas = []

files = os.listdir('/Users/royroy55/Annotationtool/datas/')
for i, file in enumerate(files):
    with open('/Users/royroy55/Annotationtool/datas/' + file, 'r') as f:
        reader = csv.reader(x.replace('\0', '') for x in f)
        for row in reader:
            datas.append(row)

# data[0] pedo~.csvのデータ
# data[0][0] image path

number = 1
for data in datas:
    # 対象画像を読み込んでリサイズ
    if os.path.exists(data[0]) == False: continue
    im = cv2.imread(data[0])
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (544, 408))

    temp = []
    data_shaped = []

    data[1] = data[1].split('[')
    for i in data[1]:
        if ']' in i:
            i = i.replace(']', '')
            temp.append(i)

    for i in temp:
        data_shaped.append(i.split(','))

    #print data_shaped
    # いざ切り抜き
    for i in data_shaped:
        #dst = im[int(float(i[1])):int(float(i[3])), int(float(i[2])):int(float(i[4]))]
        dst = im[int(float(i[2])):int(float(i[4])), int(float(i[1])):int(float(i[3]))]
        dst = cv2.resize(dst, (28, 28))
        cv2.imwrite(image_dir + i[0] + '/' + str(number) + '.png', dst)
        number += 1
