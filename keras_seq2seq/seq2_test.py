# coding: utf-8
# 全部の画像をW:H = 1:1にしてしまって、その中の0.00~みたいな座標データをぶち込む
# シンボルと座標を分けるために、数値は10次元のベクトル(その数値のところだけ1あと全部0)で表現
# シンボル10次元に対して座標2次元だから重みがどうなるか怖い
# これで学習することで、画像の縮尺成分をseq2seqは吸収してくれるのではないかと仮定
# Annotationするときに一旦W*H = 544 * 408に整形しているため、
# 現在の座標データはその画像サイズ上の座標
# 座標はstart_X, start_y, end_x, end_yの順で格納してある

from seq2seq.models import SimpleSeq2Seq
import numpy as np
import matplotlib.pylab as plt
import json
import os
import csv

# とりあえずデータ読み込むの作る
def read_datas():
    data_path = [] # csvファイルのpathを格納
    day_ans = []   # 年月日の答えを格納
    step_ans = []  # 歩数の答えを格納
    ans = {}
    num_cood = {}  #数値と座標を格納


    data_json = open('answer.json', 'r')
    data_json = json.load(data_json)

    for key, value in data_json.items():
        data_path.append(value['datas_path'])
        day_ans.append(value['day'])
        step_ans.append(value['steps'])

    count = 1
    for i, j in zip(day_ans, step_ans):
        ans['pedo' + str(count)] = [i, j]
        count += 1

    count = 1
    for i in data_path:
        with open(i, 'r') as f:
            reader = csv.reader(f)

            for row in reader:
                prepa = []
                temp = row[1]
                temp = temp.replace('[', '')
                temp = temp.replace(']', '')
                temp = temp.split(',')
                temp = list(map(float, temp))
                for j in range(0, len(temp), 5):
                    # まず数値を10次元のベクトルに変換
                    ax = np.zeros((10,), dtype=np.int)
                    ax[int(temp[j])] = 1

                    # 画像サイズが1:1になるように座標もその範囲内に変換
                    # 0 < width < 544なのでwidth/544する
                    start_x, end_x = temp[j+1]/544, temp[j+3]/544
                    # 0 < height < 408なのでheight/408する
                    start_y, end_y = temp[j+2]/408, temp[j+4]/408
                    prepa.append([ax, [start_x, end_x, start_y, end_y]])
                num_cood['pedo' + str(count)] = prepa
        count += 1

    return ans, num_cood


ans, num_cood = read_datas()
#print('answer')
#print(ans)
#print('number & coodinate')
#print(num_cood)
#quit()

# いざseq2seqにデータを入れていこう
# シンプルな Seq2Seq モデルを構築
model = SimpleSeq2Seq(input_dim=1, hidden_dim=10, output_length=8, output_dim=1)

# 学習の設定
model.compile(loss='mse', optimizer='rmsprop')

# データ作成
# 入力：数値とx座標、y座標
# 出力：各画像の年月日
a = np.random.random(1000)
x = np.array([np.sin([[p] for p in np.arange(0, 0.8, 0.1)] + aa) for aa in a])
y = -x

# 学習
model.fit(x, y, nb_epoch=5, batch_size=32)

# 未学習のデータでテスト
x_test = np.array([np.sin([[p] for p in np.arange(0, 0.8, 0.1)] + aa) for aa in np.arange(0, 1.0, 0.1)])
y_test = -x_test
print(model.evaluate(x_test, y_test, batch_size=32))

# 未学習のデータで生成
predicted = model.predict(x_test, batch_size=32)

plt.plot(np.arange(0, 0.8, 0.1), [xx[0] for xx in x_test[9]])
plt.plot(np.arange(0, 0.8, 0.1), [xx[0] for xx in predicted[9]])
plt.show()
