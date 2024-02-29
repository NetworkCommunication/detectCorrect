import csv
import random
from datetime import datetime
from datetime import timedelta

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns


def deal_dataset():
    filepath = "D:\\Desktop\\NGSIM-dataset.csv"
    f = open(filepath,'r',encoding="utf-8")
    nf = open('D:\\Desktop\\NGSIM-lankershim.csv','w',encoding="utf-8")
    writer = csv.writer(nf)

    record_num = 0
    for line in f.readlines():
        if 'lankershim' in line:
            record_num += 1
            s = line.split('"')
            for i in range(len(s)):
                if i % 2 == 1:
                    tem = s[i].split(',')
                    final_tem = ''
                    for j in range(len(tem)):
                        final_tem += tem[j]
                    s[i] = final_tem
            n_str = ''
            for ele in s:
                n_str += ele
            tem = n_str.split(',')
            record = []
            for num in [0, 3, 4, 5, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19]:
                record.append(tem[num])
            ori_time = datetime.utcfromtimestamp(float(record[1])/1000)
            record[1] = (ori_time+timedelta(hours=-7)).strftime("%Y-%m-%d %H:%M:%S.%f")
            writer.writerow(record)
    f.close()
    nf.close()

def split_data():
    ori_file = open(".\data-lankershim.csv",'r',encoding="utf-8")
    f1 = open(".\data_train1.csv",'w')
    f2 = open(".\data_test2.csv",'w')
    f3 = open(".\data_test3.csv",'w')

    i1 = 0
    i2 = 0
    i3 = 0

    for lines in ori_file.readlines():
        record_time = lines.split(',')[1]
        if "2005-06-16 08:28:00.000000" <= record_time <= "2005-06-16 08:38:00.000000":
            f1.write(lines)
            i1 += 1
        elif "2005-06-16 08:38:00.000000" < record_time <= "2005-06-16 08:48:00.000000":
            f2.write(lines)
            i2 += 1
        elif "2005-06-16 08:48:00.000000" < record_time <= "2005-06-16 08:58:00.000000":
            f3.write(lines)
            i3 += 1
        else:
            pass
    ori_file.close()
    f1.close()
    f2.close()
    f3.close()
    print(i1,i2,i3)



def make_err():
    f = open('bsm.csv', 'r')
    # f = open('.\data_test2.csv', 'r')
    f2 = open('bsm1.csv', 'a')

    record = {}
    for lines in f.readlines():
        r = lines.split(',')
        id = r[0]
        # print(id)
        if id in record.keys():
            record[id] += [lines]
        else:
            record[id] = []
            record[id] += [lines]

    for id in record.keys():
        if id == '1':
            start_record = random.randint(0, len(record[id]))
            random_err = random.randint(25, 30)
            # print(id ,start_record)
            for i in range(start_record):
                s = str(record[id][i].split('\n')[0]) + ",T\n"
                f2.writelines(s)
            for i in range(start_record, len(record[id]) - 1):
                tem = record[id][i].split(',')
                tem[2] = float(tem[2]) + random_err
                tem[-1] = str(tem[-1].split('\n')[0]) + ",F\n"
                csv_data = ','.join(map(str, tem))
                # csv_data = csv_data + ""
                f2.writelines(csv_data)
        # elif id == '97':
        #     start_record = random.randint(0, len(record[id]))
        #     random_err = random.randint(15, 25)
        #     # print(id ,start_record)
        #     for i in range(start_record):
        #         s = str(record[id][i].split('\n')[0]) + ",T\n"
        #         f2.writelines(s)
        #     for i in range(start_record, len(record[id]) - 1):
        #         tem = record[id][i].split(',')
        #         tem[3] = float(tem[3]) + random_err  # err_y
        #         tem[-1] = str(tem[-1].split('\n')[0]) + ",F\n"
        #         csv_data = ','.join(map(str, tem))
        #         f2.writelines(csv_data)
        # elif id == '905':
        #     start_record = 271
        #     random_err_k = random.randint(5,10)
        #     # print(id ,start_record)
        #     for i in range(start_record):
        #         s = str(record[id][i].split('\n')[0]) + ",T\n"
        #         f2.writelines(s)
        #     for i in range(start_record, len(record[id]) - 1):
        #         tem = record[id][i].split(',')
        #         tem[2] = float(tem[2]) * random_err_k  # err_x
        #         tem[-1] = str(tem[-1].split('\n')[0]) + ",F\n"
        #         csv_data = ','.join(map(str, tem))
        #         f2.writelines(csv_data)
        # y
        # elif id == '172':
        #     start_record = random.randint(0, len(record[id]))
        #     random_err_k = random.randint(15, 25)
        #     # print(id ,start_record)
        #     for i in range(start_record):
        #         s = str(record[id][i].split('\n')[0]) + ",T\n"
        #         f2.writelines(s)
        #     for i in range(start_record, len(record[id]) - 1):
        #         tem = record[id][i].split(',')
        #         tem[3] = float(tem[3]) * random_err_k  # err_y
        #         tem[-1] = str(tem[-1].split('\n')[0]) + ",F\n"
        #         csv_data = ','.join(map(str, tem))
        #         f2.writelines(csv_data)
        # elif id == '998':
        #     start_record = random.randint(0, len(record[id]))
        #     random_err_k = random.randint(8, 10)
        #     # print(id ,start_record)
        #     for i in range(start_record):
        #         s = str(record[id][i].split('\n')[0]) + ",T\n"
        #         f2.writelines(s)
        #     for i in range(start_record, len(record[id]) - 1):
        #         tem = record[id][i].split(',')
        #         tem[2] = float(tem[2]) * random_err_k ** 2  # err_x
        #         tem[-1] = str(tem[-1].split('\n')[0]) + ",F\n"
        #         csv_data = ','.join(map(str, tem))
        #         f2.writelines(csv_data)
        # elif id == '1028':
        #     start_record = random.randint(0, len(record[id]))
        #     random_err = random.randint(5,8)
        #     # print(id ,start_record)
        #     for i in range(start_record):
        #         s = str(record[id][i].split('\n')[0]) + ",T\n"
        #         f2.writelines(s)
        #     for i in range(start_record, len(record[id]) - 1):
        #         tem = record[id][i].split(',')
        #         tem[3] = float(tem[3]) * random_err_k ** 2  # err_y
        #         tem[-1] = str(tem[-1].split('\n')[0]) + ",F\n"
        #         csv_data = ','.join(map(str, tem))
        #         f2.writelines(csv_data)
        # elif id == '1034':
        #     start_record = random.randint(0, len(record[id]))
        #     random_err = random.randint(5, 10)
        #     # print(id ,start_record)
        #     for i in range(start_record):
        #         s = str(record[id][i].split('\n')[0]) + ",T\n"
        #         f2.writelines(s)
        #     for i in range(start_record, len(record[id]) - 1):
        #         tem = record[id][i].split(',')
        #         tem[3] = float(tem[3]) * random_err_k   # err_y
        #         tem[-1] = str(tem[-1].split('\n')[0]) + ",F\n"
        #         csv_data = ','.join(map(str, tem))
        #         f2.writelines(csv_data)
        # # v
        # elif id == '1025':
        #     start_record = random.randint(0, len(record[id]))
        #     random_err = random.randint(10, 20)
        #     # print(id ,start_record)
        #     for i in range(start_record):
        #         s = str(record[id][i].split('\n')[0]) + ",T\n"
        #         f2.writelines(s)
        #     for i in range(start_record, len(record[id]) - 1):
        #         tem = record[id][i].split(',')
        #         tem[7] = float(tem[7]) + random_err  # err_v
        #         tem[-1] = str(tem[-1].split('\n')[0]) + ",F\n"
        #         csv_data = ','.join(map(str, tem))
        #         f2.writelines(csv_data)
        # elif id == '1097':
        #     start_record = 150
        #     random_err = random.randint(5,10)
        #     # print(id ,start_record)
        #     for i in range(start_record):
        #         s = str(record[id][i].split('\n')[0]) + ",T\n"
        #         f2.writelines(s)
        #     for i in range(start_record, len(record[id]) - 1):
        #         tem = record[id][i].split(',')
        #         tem[7] = float(tem[7]) * random_err  # err_v
        #         tem[-1] = str(tem[-1].split('\n')[0]) + ",F\n"
        #         csv_data = ','.join(map(str, tem))
        #         f2.writelines(csv_data)

        # a
        elif id == '937':
            start_record = random.randint(0, len(record[id]))
            random_err = random.randint(15,20)
            # print(id ,start_record)
            for i in range(start_record):
                s = str(record[id][i].split('\n')[0]) + "\n"
                f2.writelines(s)
            for i in range(start_record, len(record[id]) - 1):
                tem = record[id][i].split(',')
                tem[8] = float(tem[8]) * random_err  # err_a
                # tem[-1] = str(tem[-1].split('\n')[0]) + ",F\n"
                tem[-1] = "F\n"
                csv_data = ','.join(map(str, tem))
                f2.writelines(csv_data)
        # elif id == '1061':
        #     start_record = random.randint(0, len(record[id]))
        #     random_err = random.randint(15,20)
        #     # print(id ,start_record)
        #     for i in range(start_record):
        #         s = str(record[id][i].split('\n')[0]) + ",T\n"
        #         f2.writelines(s)
        #     for i in range(start_record, len(record[id]) - 1):
        #         tem = record[id][i].split(',')
        #         tem[8] = float(tem[8]) * random_err  # err_a
        #         tem[-1] = str(tem[-1].split('\n')[0]) + ",F\n"
        #         csv_data = ','.join(map(str, tem))
        #         f2.writelines(csv_data)

        else:
            for i in range(len(record[id])):
                s = str(record[id][i].split('\n')[0]) + "\n"
                f2.writelines(s)
    f2.close()
    f2 = open('bsm1.csv', 'r')

def judje_norm():
    filename = './sorted1.csv'
    # filename = './bsm.csv'
    f = open(filename, 'r',encoding="utf-8")
    record = {}
    location = {}
    state = {}
    for lines in f.readlines():
        r = lines.split(',')
        id = r[0]
        x_y = [float(r[2]),float(r[3])]
        v_a = [float(r[7]),float(r[8])]
        if r[0] in record.keys():
            # dis = math.sqrt((x_y[0] - location[r[0]][0]) ** 2 + (x_y[1] - location[r[0]][1]) ** 2)
            delta_x = x_y[0] - location[id][0]
            delta_y = x_y[1] - location[id][1]
            # delta_v = v_a[0] - state[id][0]
            # delta_a = v_a[1] - state[id][1]
            # record[r[0]] = record[r[0]] + [[dis]+v_a]
            record[id] = record[id] + [[delta_x,delta_y]+v_a]

            location[id] = x_y
            # state[id] = v_a
            # record[line[0]] = record[line[0]] + [[x] + r]
        else:
            location[id] = x_y
            # state[id] = v_a
            bsm = [[0.0,0.0]+v_a]
            record[id] = bsm
    f.close()

    i = 0
    for id in record.keys():
        x = []
        y = []
        v = []
        a = []
        for bsms in record[id]:
            x.append(bsms[0])
            y.append(bsms[1])
            v.append(bsms[2])
            a.append(bsms[3])
            # print(type(x),len(x))
        data = pd.DataFrame({'x': x,
                            'y': y,
                            'v': v,
                            'a': a,})
        u1, u2, u3, u4 = data['x'].mean(), data['y'].mean(), data['v'].mean(), data['a'].mean()
        std1, std2, std3, std4 = data['x'].std(), data['y'].std(), data['v'].std(), data['a'].std()
        if stats.kstest(data['v'], 'norm', (u3, std3))[1] < 0.05:
            i += 1

    print(len(record), ",",i)

def Sym():
    filename = './sorted1.csv'
    # filename = './bsm.csv'
    f = open(filename, 'r', encoding="utf-8")
    record = {}
    location = {}
    state = {}
    for lines in f.readlines():
        r = lines.split(',')
        id = r[0]
        x_y = [float(r[2]), float(r[3])]
        v_a = [float(r[7]), float(r[8])]
        if r[0] in record.keys():
            delta_x = x_y[0] - location[id][0]
            delta_y = x_y[1] - location[id][1]
            delta_v = v_a[0] - state[id][0]
            delta_a = v_a[1] - state[id][1]
            record[id] = record[id] + [[delta_x, delta_y] + [delta_v, delta_a]]

            location[id] = x_y
            state[id] = v_a
        else:
            location[id] = x_y
            state[id] = v_a
            bsm = [[0.0, 0.0] + [0.0, 0.0]]
            record[id] = bsm
    f.close()
    x_data = []
    y_data = []
    v_data = []
    a_data = []

    for id in record.keys():
        x_data.extend([bsms[0] for bsms in record[id]])
        y_data.extend([bsms[1] for bsms in record[id]])
        v_data.extend([bsms[2] for bsms in record[id]])
        a_data.extend([bsms[3] for bsms in record[id]])

    sns.kdeplot(a_data, color='skyblue')
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.title('Density Plot of x_data')
    plt.show()

def norm():
    filename = './sorted1.csv'
    # filename = './bsm.csv'
    f = open(filename, 'r', encoding="utf-8")
    record = {}
    for lines in f.readlines():
        r = lines.split(',')
        id = r[0]
        x_y = [float(r[2]), float(r[3])]
        v_a = [float(r[7]), float(r[8])]
        if id not in record:
            record[id] = []

        record[id].append([x_y[0], x_y[1],v_a[0],v_a[1]])
    f.close()
    x_data = []
    y_data = []
    v_data = []
    a_data = []

    for id in record.keys():
        x_data.extend([bsms[0] for bsms in record[id]])
        y_data.extend([bsms[1] for bsms in record[id]])
        v_data.extend([bsms[2] for bsms in record[id]])
        a_data.extend([bsms[3] for bsms in record[id]])

    sns.kdeplot(a_data, color='skyblue')
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.title('Density Plot of x_data')
    plt.show()


if __name__ == "__main__":
    make_err()

    # print('start!')
    # judje_norm()
    # Sym()
    # norm()