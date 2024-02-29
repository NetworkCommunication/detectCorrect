import numpy as np
import matplotlib.pyplot as plt
import matplotlib;matplotlib.use('TkAgg')
from pylab import *
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def normx(x,mu,sigma):
    pdf = np.exp(-(x-mu)**2/(2*sigma**2))/(sigma * np.sqrt(2 * np.pi))
    return pdf

def plot_zt():
    mu,sigma = 2.0,3.0
    x = np.arange(mu - 4 * sigma,mu + 4*sigma, 0.01)
    y = normx(x,mu,sigma)

    x1 = np.arange(mu - 3 * sigma,mu- 2*sigma, 0.01)
    x2 = np.arange(mu - 2 * sigma,mu- 1*sigma, 0.01)
    x3 = np.arange(mu - 1 * sigma,mu+ 1*sigma, 0.01)
    x4 = np.arange(mu + 1 * sigma,mu+ 2*sigma, 0.01)
    x5 = np.arange(mu + 2 * sigma,mu+ 3*sigma, 0.01)

    plt.fill_between(x1,normx(x1,mu,sigma),color = '#C493FF',alpha = 0.7)
    plt.fill_between(x2,normx(x2,mu,sigma),color = '#D5CABD',alpha = 0.7)
    plt.fill_between(x3,normx(x3,mu,sigma),color = '#FEFEDF',alpha = 0.7)
    plt.fill_between(x4,normx(x4,mu,sigma),color = '#D5CABD',alpha = 0.7)
    plt.fill_between(x5,normx(x5,mu,sigma),color = '#C493FF',alpha = 0.7)

    plt.plot(x, y, 'k-',linewidth = 2)
    plt.vlines(mu, 0, normx(mu,mu, sigma),color = 'y',ls = 'dotted')
    plt.vlines(mu + sigma, 0, normx(mu + sigma, mu, sigma), color='y', ls='dotted')
    plt.vlines(mu - sigma, 0, normx(mu - sigma, mu, sigma), color='y', ls='dotted')
    plt.vlines(mu + 2 *sigma, 0, normx(mu + 2 * sigma, mu, sigma), color='y', ls='dotted')
    plt.vlines(mu - 2 *sigma, 0, normx(mu - 2 * sigma, mu, sigma), color='y', ls='dotted')
    plt.vlines(mu + 3 *sigma, 0, normx(mu + 3 * sigma, mu, sigma), color='y', ls='dotted')
    plt.vlines(mu - 3 *sigma, 0, normx(mu - 3 * sigma, mu, sigma), color='y', ls='dotted')

    plt.xticks([mu-3*sigma,mu-2*sigma, mu-sigma, mu, mu+sigma,mu+2*sigma,mu+3*sigma],[r"$\mu-3\sigma$",r"$\mu-2\sigma$",r"$\mu-\sigma$", r"$\mu$",r"$\mu+\sigma$",r"$\mu+2\sigma$",r"$\mu+3\sigma$"])
    plt.grid()
    plt.show()
# plot_zt()
def plot_loss():
    f = open('./result/val_loss0704.csv', 'r')
    line = f.readline().split(",")[1: -1]
    f.close()
    x = [i for i in range(len(line))]
    y = [float(line[i]) for i in range(len(line))]
    print([y[i+1] - y[i] for i in range(285,len(y)-1)])
    # plt.plot(x,y,lw=2,c='black',label='loss')
    plt.rcParams['font.sans-serif'] = ['SimHei']

    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    ax.plot(x, y, color='b', linewidth=2)

    plt.xlim(0, 300)
    plt.ylim(1.3,3.5)
    # plt.axhline(y=1.381, ls="-", c="r", label="y = 1.381", linewidth=1.5)
    plt.tick_params(labelsize=25)

    plt.xlabel("training epoch",size = 30)
    plt.ylabel("loss",size= 30)
    ax.legend(labels=["Losses in each training epoch"], ncol=2,prop={'size':30,})
    plt.show()
# plot_loss()

def detect_effect():
    # test_dataset = '.\\data_test_errordata.csv'
    test_dataset = './bsm.csv'
    f = open(test_dataset,'r')
    judge_result_add = './result.csv'
    f2 = open(judge_result_add,'r')

    judge_re = f2.readlines()
    real_results = f.readlines()

    err_data = []
    right_data = []
    ju_err = []
    ju_right =[]

    for each in real_results:
        result = each.split(",")
        tem = result[0] + "," + result[1]
        if 'F' in result[-1]:
            err_data.append(tem)
        else:
            right_data.append(tem)

    for ju_each in judge_re:
        # id_time = each.split("[")[0].strip(",")
        id_time = ju_each.split(",")[0:2]   #.strip(",")
        tem = id_time[0]+","+id_time[1]
        # print(tem)
        if 'F' in ju_each.split(",")[-1]:
            ju_err.append(tem)
        else:
            ju_right.append(tem)

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for i in err_data:
        if i in ju_err:
            tp += 1
        else:
            fn += 1
    for i in right_data:
        if i in ju_right:
            tn += 1
        else:
            fp += 1

    # print("tp, fp, tn, fn = {},{},{},{}".format(tp, fp, tn, fn))
    print("recall is {} and accuracy is {}".format(tp / (tp + fn), (tp + tn) / (tp + tn + fp + fn)))

detect_effect()

def cal_error_real():
    # f_real = open('data_test.csv', 'r')
    # f_err = open('.\\data_test_errordata.csv', 'r')
    f_real = open('realbsm.csv', 'r')
    f_err = open('bsm.csv', 'r')

    real_data = {}
    err_data = {}
    right_data = {}

    for each in f_real.readlines():
        r = each.split(",")
        idtime = r[0] + "," + r[1]
        rec = r[2: 4]
        rec += r[7: 9]
        real_data[idtime] = rec
    for ele in f_err.readlines():
        r = ele.split(",")
        idtime = r[0] + "," + r[1]
        rec = r[2: 4]
        rec += r[7: 9]
        if "F" in r[-1]:
            err_data[idtime] = rec
        else:
            right_data[idtime] = rec

    dis_e_r = [0.0, 0.0, 0.0, 0.0]

    for err_idtime in err_data.keys():
        tem_e_r = [abs(float(err_data[err_idtime][i]) - float(real_data[err_idtime][i]))
                   for i in range(len(real_data[err_idtime]))]
        dis_e_r = [tem_e_r[i] + dis_e_r[i] for i in range(len(tem_e_r))]

    print("error - real = {}".format(dis_e_r))

cal_error_real()

def cal_corr_real():
    f_real = open('realbsm.csv', 'r')
    f_err = open('bsm.csv', 'r')
    # f_corr = open('result2.csv', 'r')
    f_corr = open('result.csv', 'r')

    # f_real = open('data_test.csv', 'r')
    # f_err = open('data_test_errordata.csv','r')
    # f_corr = open('.\\jiang\\correct-result_2222_12404.csv', 'r')

    real_data = {}
    err_data = {}
    correction_data = {}
    for each in f_real.readlines():
        r = each.split(",")
        idtime = r[0] + "," + r[1]
        rec = r[2: 4]
        rec += r[7: 9]  # x,y,v,a,
        real_data[idtime] = rec

    for ele in f_err.readlines():
        r = ele.split(",")
        idtime = r[0] + "," + r[1]
        rec = r[2: 4]
        rec += r[7: 9]
        if "F" in r[-1]:
            err_data[idtime] = rec
        else:
            del real_data[idtime]

    dis_c_r = [0.0, 0.0, 0.0, 0.0]

    for each in f_corr.readlines():
        r = each.replace("[","").replace("]","").split(",")
        idtime = r[0] + "," + r[1]
        correction_data[idtime] = r[2:6]

    for idt in real_data.keys():
        if idt in correction_data.keys():
            dis_c_r = [dis_c_r[i] + abs(float(real_data[idt][i]) - float(correction_data[idt][i])) for i in range(len(dis_c_r))]
        else:
            dis_c_r = [dis_c_r[i] + 0.0 for i in range(len(dis_c_r))]
    print("corre - real = {}".format(dis_c_r))
cal_corr_real()


