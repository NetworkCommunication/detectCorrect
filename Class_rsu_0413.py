import datetime
import math
import time
from interval import Interval
import numpy as np
from tensorflow.keras.models import load_model  # noqa


class RSU:
    his_record = {}
    nei_range_long = 75.0 / 2
    nei_range_width = 21.0 / 2
    model = load_model('result/model070422.h5')
    # model = load_model('./myTest/model0622.h5')
    max_values = []
    min_values = []
    f = open('./max_min.csv', 'r')
    line = f.readlines()[-1]
    mean_std = line.split('\t')
    f.close()
    x_err_value = 2
    y_err_value = 4
    v_err_value = 4
    a_err_value = 0.8

    threshold = [3,3,3,3]

    # result_path = 'correct-result_2222_24404.csv'
    result_path = 'result.csv'


    def __init__(self):
        lines = open('max_min.csv', 'r').readlines()
        tem = lines[0].split(" ")
        self.min_values = tem[2:5] + [tem[6]]
        # print(self.min_values)
        for i in range(len(self.min_values)):
            self.min_values[i] = float(self.min_values[i])
        tem = lines[1].split(" ")
        self.max_values = tem[2:5] + [tem[6]]
        for i in range(len(self.max_values)):
            self.max_values[i] = float(self.max_values[i])

    def deal_record(self, record: dict):
        for ve_id in record.keys():
            if ve_id not in self.his_record.keys():
                self.his_record[ve_id] = []
            self.his_record[ve_id] += [record[ve_id]]  # time,x,y,v,a

            if len(self.his_record[ve_id]) > 1:
                now_time = datetime.datetime.strptime(str(self.his_record[ve_id][-1][0]), '%Y-%m-%d %H:%M:%S.%f')
                last_time = datetime.datetime.strptime(str(self.his_record[ve_id][-2][0]), '%Y-%m-%d %H:%M:%S.%f')
                if (now_time - last_time).total_seconds() > 0.1:
                    self.his_record[ve_id] = [self.his_record[ve_id][-1]]

    def process_record(self, ve_id, now_time):
        f = open(self.result_path, 'a')

        flag, err = self.judge_BSM(self.his_record[ve_id][-1])
        signal = 'T'
        if flag == 1:
            nei = self.find_nei(ve_id, now_time)
            signal = 'F'
            if len(nei) == 0:
                single_corr1 = self.single_prediction(ve_id)
                single_corr2 = self.single_correction(ve_id,err)
                multi = [single_corr2]
                multi.append(single_corr1)
                corr_project, final = self.corr_project(multi, self.his_record[ve_id][-1], ve_id)

            else:
                multi_corr = self.multi_prediction(ve_id, nei, err)
                single_corr = self.single_prediction(ve_id)
                multi_corr.append(single_corr)
                corr_project, final = self.corr_project(multi_corr, self.his_record[ve_id][-1], ve_id)

            if all(x == 0 for x in corr_project):
                signal = 'T'

            self.his_record[ve_id][-1][1:] = final
        else:
            corr_project = [0.0, 0.0, 0.0, 0.0]
        f.writelines(ve_id + "," + str(self.his_record[ve_id][-1][0]) + "," + str(self.his_record[ve_id][-1][1:]) +  "," + signal + '\n')
        return corr_project


    def judge_BSM(self, v_record: list):
        flag = 0
        err_arr = []
        threshold = self.threshold
        z_score = []
        for i in range(4):
            z_score.append((v_record[i+1] - float(self.mean_std[2*(i+1)-2]))/float(self.mean_std[2*(i+1)-1]))
        # f.writelines(str(z_score)+"\n")
        if abs(z_score[0]) > threshold[0] or abs(z_score[1]) > threshold[1]:
            flag = 1
            err_arr.append(1)
        if abs(z_score[2]) > threshold[2]:
            flag = 1
            err_arr.append(2)
        if abs(z_score[3]) > threshold[3]:
            flag = 1
            err_arr.append(3)
        if all(abs(z_score[i]) <= threshold[i] for i in range(len(z_score))):
            flag, err_arr = 0, 0
        return flag, err_arr

    def single_prediction(self, v_id, data_length=5):
        pred_data = self.his_record[v_id][-1][1:]
        if len(self.his_record[v_id]) - 1 >= data_length:
            input_data = self.his_record[v_id][-1-data_length: -1]
            for i in range(len(input_data)):
                input_data[i] = input_data[i][1:]
            in_data = []
            for i in range(1, len(input_data)):
                delta_x = input_data[i][0] - input_data[i - 1][0]
                delta_y = input_data[i][1] - input_data[i - 1][1]
                delta_v = input_data[i][2] - input_data[i - 1][2]
                delta_a = input_data[i][3] - input_data[i - 1][3]
                tem = [delta_x, delta_y, delta_v, delta_a]
                in_data.append(tem)
            for i in range(len(in_data)):
                for j in range(len(in_data[i])):
                    in_data[i][j] = (in_data[i][j] - self.min_values[j]) / (self.max_values[j] - self.min_values[j])
            pred_data = self.model.predict(np.array([in_data]), verbose=0)[0].tolist()
            # print(type(pred_data))
            for i in range(len(pred_data)):
                pred_data[i] = pred_data[i] * (self.max_values[i] - self.min_values[i]) + self.min_values[i]
            pred_data = [pred_data[i] + self.his_record[v_id][-2][i + 1] for i in range(len(pred_data))]
            return pred_data
        else:  # when len(history_data)<2,it cannot correct BSM by itself
            return pred_data

    def single_correction(self, v_id, err_arr):
        if len(self.his_record[v_id]) <= 2:
            return self.his_record[v_id][-1][1:]
        mess_t = 0.1
        pred_data = self.his_record[v_id][-1][1:]
        mess_1, mess_2 = self.his_record[v_id][-2], self.his_record[v_id][-3]

        if 1 in err_arr:  # position error
            x1, y1 = mess_1[1], mess_1[2]
            x2, y2 = mess_2[1], mess_2[2]
            angle = math.atan((y1-y2)/(x1-x2)) if x1 != x2 else math.pi/2
            if angle != 90:
                # x = x1 + x1 * mess_t + 0.5 * (mess_1[4] * math.cos(angle)) * mess_t ** 2
                # y = y1 + y1 * mess_t + 0.5 * (mess_1[4] * math.sin(angle)) * mess_t ** 2
                x = x1 + mess_1[3] * mess_t + 0.5 * (mess_1[4] * math.cos(angle)) * mess_t ** 2
                y = y1 + mess_1[3] * mess_t + 0.5 * (mess_1[4] * math.sin(angle)) * mess_t ** 2
            else:  # angle = 90,only y has movement
                x = x1
                y = y1 + mess_1[3] * mess_t + 0.5 * (mess_1[4] * math.sin(angle)) * mess_t ** 2
            pred_data[0], pred_data[1] = x, y
        if 2 in err_arr:  # v error
            v1, a1 = mess_1[3], mess_1[4]
            v = v1 + 0.5 * (a1 + self.his_record[v_id][-1][4]) * mess_t  # v = v0 +0.5*(a0+a1)*t
            pred_data[2] = v
        if 3 in err_arr:  # a error
            a = (self.his_record[v_id][-1][3] - mess_1[3]) / mess_t # (v1-v0)/t
            pred_data[3] = a
        return pred_data

    def find_nei(self, v_id: str, now_time: str):
        nei = []
        last_time = datetime.datetime.strptime(now_time, '%Y-%m-%d %H:%M:%S.%f') - datetime.timedelta(seconds=0.1)
        last_time = str(last_time)
        if len(self.his_record[v_id]) >= 2:
            v_x = float(self.his_record[v_id][-2][1])
            v_y = float(self.his_record[v_id][-2][2])
            for nei_id in self.his_record.keys():
                if len(self.his_record[nei_id]) >= 2 and self.his_record[nei_id][-2][0] == last_time:
                    x = float(self.his_record[nei_id][-2][1])
                    y = float(self.his_record[nei_id][-2][2])
                    x_range = Interval(v_x - self.nei_range_width, v_x + self.nei_range_width)
                    y_range = Interval(v_y - self.nei_range_long, v_y + self.nei_range_long)
                    if x in x_range and y in y_range and x != v_x and y != v_y:
                        nei.append(nei_id)
        else:
            pass
        return nei

    def multi_prediction(self, v_id: str, nei_id: list, err_arr: int, record_len=3):
        if len(nei_id) == 0 or len(self.his_record[v_id]) == 1:
            return []
        ve_corrected = self.his_record[v_id][-1][1:]
        v_his_len = len(self.his_record[v_id]) - 1

        if 1 in err_arr:
            v_x = []
            v_y = []
            corr = []
            for each_id in nei_id:
                tem = ve_corrected
                nei_his_len = len(self.his_record[each_id])
                ref_len = min(min(record_len, v_his_len), nei_his_len)
                nei_his = self.his_record[each_id][-1 - ref_len:-1]
                v_his = self.his_record[v_id][-1 - ref_len:-1]

                delta_x = [nei_his[i][1] - v_his[i][1] for i in range(len(nei_his))]
                delta_y = [nei_his[i][2] - v_his[i][2] for i in range(len(nei_his))]
                if all(delta_x[i] == delta_x[0] for i in range(len(delta_x))):
                    v_x = nei_his[-1][1] - delta_x[-1]
                elif all(delta_x[i] - delta_x[i + 1] == delta_x[i + 1] - delta_x[i + 2] for i in
                         range(len(delta_x) - 2)):
                    v_x = nei_his[-1][1] - delta_x[-1] + delta_x[-2] - delta_x[-1]
                else:
                    ave_delta_v = sum(delta_x) / len(delta_x)
                    v_x = nei_his[-1][1] - ave_delta_v

                if all(delta_y[i] == delta_y[0] for i in range(len(delta_y))):
                    v_y = nei_his[-1][2] - delta_y[-1]
                elif all(delta_y[i] - delta_y[i + 1] == delta_y[i + 1] - delta_y[i + 2] for i in
                         range(len(delta_y) - 2)):
                    v_y = nei_his[-1][2] - delta_y[-1] - delta_y[-1] + delta_y[-2]
                else:
                    v_y = nei_his[-1][2] - sum(delta_y) / len(delta_y)
                tem[0] = v_x
                tem[1] = v_y
                corr.append(tem)
            return corr
        if 2 in err_arr:
            v_v = []
            corr = []
            for each_id in nei_id:
                tem = ve_corrected
                v_his_len = len(self.his_record[v_id]) - 1
                nei_his_len = len(self.his_record[each_id])
                ref_len = min(min(record_len, v_his_len), nei_his_len)

                nei_his = self.his_record[each_id][-1 - ref_len:-1]
                v_his = self.his_record[v_id][-1 - ref_len:-1]

                delta_v = [nei_his[i][3] - v_his[i][3] for i in range(len(nei_his))]
                if all(delta_v[i] == delta_v[0] for i in range(len(delta_v))):
                    v_v = nei_his[-1][3] - delta_v[-1]
                elif all(delta_v[i] - delta_v[i + 1] == delta_v[i + 1] - delta_v[i + 2] for i in
                         range(len(delta_v) - 2)):
                    v_v = nei_his[-1][3] - delta_v[-1] - delta_v[-1] + delta_v[-2]
                else:
                    v_v = nei_his[-1][3] - sum(delta_v) / len(delta_v)
                tem[2] = v_v
                corr.append(tem)
            return corr

        if 3 in err_arr:
            v_a = []
            corr = []
            for each_id in nei_id:
                tem = ve_corrected
                v_his_len = len(self.his_record[v_id]) - 1
                nei_his_len = len(self.his_record[each_id])
                ref_len = min(min(record_len, v_his_len), nei_his_len)

                nei_his = self.his_record[each_id][-1 - ref_len:-1]
                v_his = self.his_record[v_id][-1 - ref_len:-1]

                delta_a = [nei_his[i][4] - v_his[i][4] for i in range(len(nei_his))]
                if all(delta_a[i] == delta_a[0] for i in range(len(delta_a))):
                    v_a = nei_his[-1][4] - delta_a[-1]
                elif all(delta_a[i] - delta_a[i + 1] == delta_a[i + 1] - delta_a[i + 2] for i in
                         range(len(delta_a) - 2)):
                    v_a = nei_his[-1][4] - delta_a[-1] - delta_a[-1] + delta_a[-2]
                else:
                    v_a = nei_his[-1][4] - sum(delta_a) / len(delta_a)
                tem[3] = v_a
            corr.append(tem)
            return corr

    def corr_project(self, multi: list, now_data: list, v_id: str):
        for i in range(len(multi) - 1, -1, -1):
            if all(abs((multi[i][j] - float(self.mean_std[2 * j]))/float(self.mean_std[2 * j + 1])) < self.threshold[j] for j in range(4)):
                pass
            else:
                multi.pop(i)
        if len(multi) == 0:
            return [0.0, 0.0, 0.0, 0.0], now_data[1:]
        final_data = [0.0, 0.0, 0.0, 0.0]
        for i in range(len(multi)):
            final_data[0] += (multi[i][0] / len(multi))
            final_data[1] += (multi[i][1] / len(multi))
            final_data[2] += (multi[i][2] / len(multi))
            final_data[3] += (multi[i][3] / len(multi))
        if abs(final_data[0] - self.his_record[v_id][-1][1]) <= self.x_err_value:
            final_data[0] = self.his_record[v_id][-1][1]
        if abs(final_data[1] - self.his_record[v_id][-1][2]) <= self.y_err_value:
            final_data[1] = self.his_record[v_id][-1][2]
        if abs(final_data[2] - self.his_record[v_id][-1][3]) <= self.v_err_value:
            final_data[2] = self.his_record[v_id][-1][3]
        if abs(final_data[3] - self.his_record[v_id][-1][4]) <= self.a_err_value:
            final_data[3] = self.his_record[v_id][-1][4]

        project = [final_data[i] - now_data[i + 1] for i in range(len(final_data))]
        return project, final_data
