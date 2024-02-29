
import time
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential  # noqa   该注释用来忽略该警告
from tensorflow.keras.layers import LSTM, Dense  # noqa
import tensorflow as tf


class DataCenter:
    name = ''
    train_nums = 4

    def __init__(self, name='default', train_nums = 4):
        self.name = name
        self.train_nums = train_nums


    def dict2record(self, filename: str):
        f = open(filename, 'r', encoding="utf-8")
        record = {}  # {id:[id,time,x,y,length,width,type,v,a]}
        for lines in f.readlines():
            r = lines.split(',')
            vid = r[0]
            x_y = [float(r[2]), float(r[3])]
            v_a = [float(r[7]), float(r[8])]
            if vid in record.keys():
                tem = [x_y + v_a]
                record[vid] += tem
            else:
                record[vid] = []
                tem = [x_y + v_a]
                record[vid] += tem
        f.close()

        id_list = []
        for vid in record.keys():
            if len(record[vid]) <= self.train_nums:
                id_list.append(vid)
        for vid in id_list:
            record.pop(vid)
        return record

    def dealRecord(self, full_record: dict):
        dealed_re = {key: full_record[key] for key in full_record.keys()}
        for vid in full_record.keys():
            dealed_re[vid] = []
            for i in range(1, len(full_record[vid])):
                delta_x = full_record[vid][i][0] - full_record[vid][i-1][0]
                delta_y = full_record[vid][i][1] - full_record[vid][i-1][1]
                delta_v = full_record[vid][i][2] - full_record[vid][i-1][2]
                delta_a = full_record[vid][i][3] - full_record[vid][i-1][3]
                tem = [delta_x, delta_y, delta_v, delta_a]
                dealed_re[vid] += [tem]
        data_list = list(dealed_re.values())
        data = np.concatenate(data_list, axis=0)
        max_values = np.max(data, axis=0)
        min_values = np.min(data, axis=0)
        for vid in dealed_re.keys():
            for each in dealed_re[vid]:
                each = [(each[i] - min_values[i]) / (max_values[i] - min_values[i]) for i in range(3)]
        f = open('./max_min.csv', 'a')
        f.writelines("min values:" + str(min_values) + "\n")
        f.writelines("max values:" + str(max_values))
        f.close()
        return dealed_re, min_values, max_values


    def z_score(self, record: dict, filepath: str):
        f = open(filepath, 'a')
        # del_id = []
        # z_scored_re = []
        x = []
        y = []
        v = []
        a = []
        for v_id in record.keys():
            for bsms in record[v_id]:
                x.append(bsms[0])
                y.append(bsms[1])
                v.append(bsms[2])
                a.append(bsms[3])
            data = pd.DataFrame({'x': x,
                                 'y': y,
                                 'v': v,
                                 'a': a, })
        u1, u2, u3, u4 = data['x'].mean(), data['y'].mean(), data['v'].mean(), data['a'].mean()
        std1, std2, std3, std4 = data['x'].std(), data['y'].std(), data['v'].std(), data['a'].std()
        z_scored_re = [u1, std1, u2, std2, u3, std3, u4, std4]
        f.write("\nz-score:\n")
        for each in z_scored_re:
            f.write(str(each) + '\t')
        f.write('\n')
        # for vid in del_id:
        #     record.pop(vid)
        f.close()

    def train_model(self, record: dict):

        data_list = list(record.values())
        sequences = {}
        for i, vid in enumerate(record.keys()):
            sequences[vid] = data_list[i]
        X_seq = []
        y_seq = []

        for v_id in sequences.keys():
            sequence = sequences[v_id]
            for i in range(len(sequence) - self.train_nums):
                X_seq.append(sequence[i: i + self.train_nums])
                y_seq.append(sequence[i + self.train_nums])
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        indices = np.arange(len(X_seq))
        np.random.shuffle(indices)
        X_seq = X_seq[indices]
        y_seq = y_seq[indices]
        l = len(X_seq)
        test_X = X_seq[ : int(0.7*l)]
        test_y = y_seq[ : int(0.7*l)]
        val_x =  X_seq[ int(0.7*l): ]
        val_y =  y_seq[ int(0.7*l): ]

        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(units=128, return_sequences=True, input_shape=(None, 4)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(units=64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(units=32, return_sequences=False),
            tf.keras.layers.Dense(units=8),
            tf.keras.layers.Dense(units=4)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')

        history = model.fit(test_X, test_y, validation_data=(val_x, val_y), epochs=250, batch_size=32)
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        model.save(f"./result/model0704.h5")
        f = open('./result/loss0704.csv', 'a')
        f.write(str(loss))
        f.close()
        f2 = open('./result/val_loss0704.csv', 'a')
        f2.write(str(val_loss))
        f2.close()
        return model


if __name__ == "__main__":
    dc = DataCenter()
    record = dc.dict2record('./实验/detection/data/data_train1.csv')
    d_record, minlist, maxlist = dc.dealRecord(record)
    model = dc.train_model(d_record)
    # dc.z_score(record,"max_min.csv")
