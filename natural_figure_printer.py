import matplotlib.pyplot as plt
from sklearn import linear_model
import pickle
from criterion import criterion_dic
from criterion import nse
from matplotlib.collections import LineCollection
import pandas
import numpy as np
import matplotlib.ticker as ticker
from matplotlib import rcParams

config = {
    "font.family": 'serif',
    "font.size": 15,
    "mathtext.fontset": 'stix',
    "font.serif": ['Times New Roman'],
    'font.weight': 'bold',
    "axes.labelweight": 'bold'
}
rcParams.update(config)


if __name__ == '__main__':

    '''
    --------------------------------------------------------------------------------------------------------------------
    Calibration of T_long and T_short
    --------------------------------------------------------------------------------------------------------------------
    '''
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    result_dic = {'training': [], 'testing':[]}
    experiment_list_long = [[0, 12], [1, 12], [2, 12], [3, 12], [4, 12], [5, 12]]
    for e in experiment_list_long:
        f_read = open('./result/long%02d_short%02d_training_true.pkl' % (e[0], e[1]), 'rb')
        true = pickle.load(f_read)
        f_read.close()
        f_read = open('./result/long%02d_short%02d_training_pred.pkl' % (e[0], e[1]), 'rb')
        pred = pickle.load(f_read)
        f_read.close()
        result_dic['training'].append(criterion_dic['nse'](np.squeeze(true, axis=-1), np.squeeze(pred, axis=-1)))

        f_read = open('./result/long%02d_short%02d_testing_true.pkl' % (e[0], e[1]), 'rb')
        true = pickle.load(f_read)
        f_read.close()
        f_read = open('./result/long%02d_short%02d_testing_pred.pkl' % (e[0], e[1]), 'rb')
        pred = pickle.load(f_read)
        f_read.close()
        result_dic['testing'].append(criterion_dic['nse'](np.squeeze(true, axis=-1), np.squeeze(pred, axis=-1)))

    fig = plt.figure(figsize=(16, 4))
    ax1_l = fig.add_subplot(1, 2, 1)

    x = np.arange(0, len(experiment_list_long))
    # x_ticks = ['$T_{long}=12$', '$T_{long}=24$', '$T_{long}=36$', '$T_{long}=48$', '$T_{long}=60$',]
    x_ticks = ['0', '12', '24', '36', '48', '60']
    color_dic = {'training': '#1f70a9', 'testing': '#449945'}
    marker_dic = {'training': '^', 'testing': '^'}
    ax1_l.set_ylabel('NSE')
    ax1_l.set_ylim(0, 1.0)
    ax1_l.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax1_l.set_xlabel("$T_{l}$(Month)", fontsize=16)
    ax1_l.set_xticks(x, x_ticks)
    ax1_l.text(-0.9, 1.0, '(a)', fontsize=20)
    ax1_l.vlines(3, ymin=0, ymax=1.5, linestyles='--', colors='black', linewidth=1)
    ax1_l.grid(axis='y')

    ax1_l.plot(x, result_dic['training'], label='Training', color=color_dic['training'], marker=marker_dic['training'])
    ax1_l.plot(x, result_dic['testing'], label='Testing', color=color_dic['testing'], marker=marker_dic['testing'])
    for i, x_i in enumerate(x):
        ax1_l.text(x_i-0.2, result_dic['training'][i]-0.09, '%03.2f' % result_dic['training'][i], color=color_dic['training'])
        if i == 0:
            ax1_l.text(x_i-0.2, result_dic['testing'][i] + 0.05, '%03.2f' % result_dic['testing'][i], color=color_dic['testing'])
        else:
            ax1_l.text(x_i - 0.2, result_dic['testing'][i] - 0.09, '%03.2f' % result_dic['testing'][i], color=color_dic['testing'])
    ax1_l.legend(loc='lower right', fontsize=14)


    result_dic = {'training': [], 'testing': []}
    experiment_list_short = [[3, 0], [3, 4], [3, 8], [3, 12], [3, 16], [3, 20]]
    for e in experiment_list_short:
        f_read = open('./result/long%02d_short%02d_training_true.pkl' % (e[0], e[1]), 'rb')
        true = pickle.load(f_read)
        f_read.close()
        f_read = open('./result/long%02d_short%02d_training_pred.pkl' % (e[0], e[1]), 'rb')
        pred = pickle.load(f_read)
        f_read.close()
        result_dic['training'].append(criterion_dic['nse'](np.squeeze(true, axis=-1), np.squeeze(pred, axis=-1)))

        f_read = open('./result/long%02d_short%02d_testing_true.pkl' % (e[0], e[1]), 'rb')
        true = pickle.load(f_read)
        f_read.close()
        f_read = open('./result/long%02d_short%02d_testing_pred.pkl' % (e[0], e[1]), 'rb')
        pred = pickle.load(f_read)
        f_read.close()
        result_dic['testing'].append(criterion_dic['nse'](np.squeeze(true, axis=-1), np.squeeze(pred, axis=-1)))

    ax2_l = fig.add_subplot(1, 2, 2)
    x = np.arange(0, len(experiment_list_short))
    x_ticks = ['0', '4', '8', '12', '16', '20', ]
    ax2_l.set_ylabel('NSE')
    ax2_l.set_ylim(0, 1.0)
    ax2_l.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax2_l.set_xlabel("$T_{s}$(Month)", fontsize=16)
    ax2_l.text(-0.9, 1.0, '(b)', fontsize=20)
    ax2_l.set_xticks(x, x_ticks)
    ax2_l.vlines(3, ymin=0, ymax=1.0, linestyles='--', colors='black', linewidth=1)
    ax2_l.grid(axis='y')
    ax2_l.plot(x, result_dic['training'], label='Training', color=color_dic['training'], marker=marker_dic['training'])
    ax2_l.plot(x, result_dic['testing'], label='Testing', color=color_dic['testing'], marker=marker_dic['testing'])
    for i, x_i in enumerate(x):
        ax2_l.text(x_i-0.2, result_dic['training'][i]-0.09, '%03.2f' % result_dic['training'][i], color=color_dic['training'])
        ax2_l.text(x_i-0.2, result_dic['testing'][i] - 0.09, '%03.2f' % result_dic['testing'][i], color=color_dic['testing'])
    ax2_l.legend(loc='lower right', fontsize=14)
    plt.subplots_adjust(wspace=0.2, )
    plt.savefig('./figure/Calibration of T_long and T_short.png', bbox_inches='tight')
    plt.show()

    '''
    --------------------------------------------------------------------------------------------------------------------
    Comparison between observed and simulated spring discharge (natural period)
    --------------------------------------------------------------------------------------------------------------------
    '''
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    f_read = open('./result/long03_short12_training_true.pkl', 'rb')
    true_training = pickle.load(f_read)
    f_read.close()

    f_read = open('./result/long03_short12_training_pred.pkl', 'rb')
    pred_training = pickle.load(f_read)
    f_read.close()

    f_read = open('./result/long03_short12_testing_true.pkl', 'rb')
    true_testing = pickle.load(f_read)
    f_read.close()

    f_read = open('./result/long03_short12_testing_pred.pkl', 'rb')
    pred_testing = pickle.load(f_read)
    f_read.close()

    true = np.concatenate((true_training.reshape(true_training.shape[0]), true_testing.reshape(true_testing.shape[0])))
    pred_training = pred_training.reshape(pred_training.shape[0])
    pred_testing = pred_testing.reshape(pred_testing.shape[0])
    x_train = np.arange(0, pred_training.shape[0])
    x_test = np.arange(pred_training.shape[0], true.shape[0])
    pred = np.concatenate((pred_training, pred_testing))

    x = np.arange(0, true.shape[0])

    fig = plt.figure(figsize=(16, 5))
    gs = fig.add_gridspec(1, 3, hspace=0, wspace=0, width_ratios=[6, 5, 5])
    (ax1, ax2, ax3) = gs.subplots(sharey='row')
    fs = 14
    ax1.set_xlim(0, true.shape[0])
    ax1.set_ylim(10.5, 19.5)
    ax1.set_yticks(np.arange(11, 20), labels=np.arange(11, 20), fontsize=fs)
    ax1.set_xlabel('Time(Year)', fontsize=fs)
    ax1.set_ylabel('Spring discharge ($\\rm m^3/s$)', fontsize=fs)
    ax1.xaxis.set_major_locator(ticker.FixedLocator(np.linspace(0, true.shape[0], 12)))
    ax1.xaxis.set_major_formatter(ticker.NullFormatter())
    ax1.xaxis.set_minor_locator(ticker.FixedLocator(np.arange(6, true.shape[0], 12)))
    ax1.xaxis.set_minor_formatter(ticker.FixedFormatter(np.arange(1960, 1971, 1)))
    ax1.tick_params(axis='x', which='minor', size=0, labelsize=fs)  # size=0 means bu xian shi xiao ke du
    ax1.tick_params(axis='y', which='both', right=True)  # you mian ye xian shi ke du
    ax1.tick_params(axis='both', which='both', direction='in')
    ax1.plot(x, true, color='#c22f2f', label='Observed spring discharges', linewidth=1)
    ax1.plot(x_train, pred_training, color='#1f70a9', label='Simulated values (training)', linewidth=1)
    ax1.plot(x_test, pred_testing, color='#449945', label='Simulated values (testing)', linewidth=1)

    NSE = nse(true_training.squeeze(), pred_training.squeeze())
    ax1.text(15, 15, 'NSE=%5.3f' % NSE, color='#1f70a9')
    NSE = nse(true_testing.squeeze(), pred_testing.squeeze())
    ax1.text(9 * 12-5, 15, 'NSE=%5.3f' % NSE, color='#449945')
    ax1.text(6, 18.5, '(a)', fontsize=20)
    ax1.legend(fontsize=13)  # 自由调整图例位置

    LR = linear_model.LinearRegression(fit_intercept=True)
    LR.fit(true_training, pred_training)
    R2 = LR.score(true_training, pred_training)
    k = LR.coef_
    b = LR.intercept_
    ax2.scatter(pred_training, true_training, color='royalblue', s=12)
    ax2.set_xlim(10.5, 19.5)
    x_tick = np.arange(11, 20)
    ax2.set_xticks(x_tick, labels=x_tick, fontsize=fs)
    ax2.set_xlabel('Simulated value in training ($\\rm m^3/s$)', fontsize=fs)
    ax2.tick_params(axis='both', which='both', direction='in')
    ax2.plot(LR.predict(true_training), true_training, color='red')
    if b > 0:
        ax2.text(15, 12, '$y=%3.2fx+%3.2f$' % (k, b))
    else:
        ax2.text(15, 12, '$y=%3.2fx%3.2f$' % (k, b))
    ax2.text(15, 11, '$\\rm R^2=%5.3f$' % R2)
    ax2.text(11, 18.5, '(b)', fontsize=20)
    ax2.axline((9.5, 9.5), (19.5, 19.5), linestyle='--', color='black')

    LR = linear_model.LinearRegression(fit_intercept=True)
    LR.fit(true_testing, pred_testing)
    R2 = LR.score(true_testing, pred_testing)
    k = LR.coef_
    b = LR.intercept_
    ax3.scatter(pred_testing, true_testing, color='royalblue', s=12)
    ax3.set_xlim(10.5, 19.5)
    x_tick = np.arange(11, 20)
    ax3.set_xticks(x_tick, labels=x_tick, fontsize=fs)
    ax3.set_xlabel('Simulated value in testing ($\\rm m^3/s$)', fontsize=fs)
    ax3.tick_params(axis='both', which='both', direction='in')
    ax3.plot(LR.predict(true_testing), true_testing, color='red')
    if b > 0:
        ax3.text(15, 12, '$y=%3.2fx+%3.2f$' % (k, b))
    else:
        ax3.text(15, 12, '$y=%3.2fx%3.2f$' % (k, b))
    ax3.text(15, 11, '$\\rm R^2=%5.3f$' % R2)
    ax3.text(11, 18.5, '(c)', fontsize=20)
    ax3.axline((9.5, 9.5), (19.5, 19.5), linestyle='--', color='black')

    plt.savefig('./figure/Comparison between observed and simulated spring discharge (natural period).png', bbox_inches='tight')
    plt.show()

    '''
    --------------------------------------------------------------------------------------------------------------------
    The average precipitation for each source area and corresponding received average attention Natural period
    --------------------------------------------------------------------------------------------------------------------
    '''
    f_read = open('./result/long03_short12_w_s.pkl', 'rb')
    w_s = pickle.load(f_read)
    f_read.close()

    data_frame = pandas.read_excel('./data/rainfall_and_spring_flow.xlsx', sheet_name=0, keep_default_na=False)
    data_frame = data_frame.iloc[:, 2:9]
    data_np = np.array(data_frame)
    start_year = 1959
    end_year = 1970
    start_index = (start_year - 1959) * 12
    end_index = (end_year - 1959 + 1) * 12
    data_np = data_np[start_index:end_index]
    precipitation_mean = np.mean(data_np, axis=0)

    w_s_training = w_s['training']
    w_s_testing = w_s['testing']
    w_s_tr_te = np.concatenate([w_s_training, w_s_testing], axis=0)
    w_s_avg = np.mean(w_s_tr_te, axis=0)
    # w_s_percentage = (w_s_sum / np.sum(w_s_sum)) * 100

    labels = ['Yangquan', 'Pingding', 'Yuxian', 'Shouyang', 'Xiyang', 'Heshun', 'Zuoquan']
    x = np.arange(len(labels))  # the label locations
    width = 0.3  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 5))
    color_1 = 'royalblue'
    color_2 = 'darkorange'
    rects1 = ax.bar(x - width / 2, precipitation_mean, width, label='Average precipitation',
                    color=color_1)
    ax_tw = ax.twinx()
    rects2 = ax_tw.bar(x + width / 2, w_s_avg, width, label='Average attention', color=color_2)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Average precipitation (mm)', color=color_1, fontsize=18)
    ax.tick_params(axis='y', labelcolor=color_1)
    ax.set_xlabel('Source areas', fontsize=18)
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.text(x[0] - width*1.5, 90, '(a)', fontsize=25)
    ax_tw.set_ylim(0, 0.3)
    ax_tw.set_ylabel('Average attention', color=color_2, fontsize=18)
    ax_tw.tick_params(axis='y', labelcolor=color_2)

    def autolabel(rects, ax, color):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('%4.2f' % height,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=90, fontsize=13, color=color)

    autolabel(rects1, ax, color_1)
    autolabel(rects2, ax_tw, color_2)

    # fig.tight_layout()
    plt.savefig('./figure/The average precipitation for each source area and corresponding received average attention Natural period.png', bbox_inches='tight')
    plt.show()

    '''
    --------------------------------------------------------------------------------------------------------------------
    The average precipitation for each month and corresponding average attention received during natural period
    --------------------------------------------------------------------------------------------------------------------
    '''
    f_read = open('./result/long03_short12_w_t.pkl', 'rb')
    w_t = pickle.load(f_read)
    f_read.close()

    data_frame = pandas.read_excel('./data/rainfall_and_spring_flow.xlsx', sheet_name=0, keep_default_na=False)
    data_frame = data_frame.iloc[:, 2:9]
    data_np = np.array(data_frame)
    start_year = 1959
    end_year = 1970
    total_month = (end_year - start_year + 1) * 12
    start_index = (start_year - 1959) * 12
    end_index = (end_year - 1959 + 1) * 12
    data_np = data_np[start_index:end_index]
    data_np = data_np.reshape(total_month // 12, 12, 7)

    precipitation_mean = np.mean(data_np, axis=-1)
    precipitation_mean = np.mean(precipitation_mean, axis=0)

    w_t_training = np.stack(w_t['training'], axis=0)
    w_t_testing = np.stack(w_t['testing'], axis=0)
    w_t_tr_te = np.concatenate([w_t_training, w_t_testing], axis=0)
    for i in range(w_t_tr_te.shape[0] // 12):
        temp = w_t_tr_te[i * 12:(i + 1) * 12]
        for j in range(12):
            part_1 = temp[j, 0:12 - j]
            if j == 0:
                whole = part_1
            else:
                part_2 = temp[j, 12 - j:]
                whole = np.concatenate([part_2, part_1], axis=0)
            temp[j] = whole
        w_t_tr_te[i * 12:(i + 1) * 12] = temp
    w_t_avg = np.mean(w_t_tr_te, axis=-1)
    w_t_avg = np.mean(w_t_avg, axis=0)
    # w_t_percentage = (w_t_sum / np.sum(w_t_sum)) * 100

    labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 5))
    color_1 = 'royalblue'
    color_2 = 'darkorange'
    rects1 = ax.bar(x - width / 2, precipitation_mean, width, label='Average precipitation',
                    color=color_1)
    ax_tw = ax.twinx()
    rects2 = ax_tw.bar(x + width / 2, w_t_avg, width, label='Average attention', color=color_2)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Average precipitation (mm)', color=color_1, fontsize=18)
    ax.tick_params(axis='y', labelcolor=color_1)
    ax.set_xlabel('Months', fontsize=18)
    ax.set_ylim(0, 200)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.text(x[0] - width*1.5, 175, '(a)', fontsize=25)
    ax_tw.set_ylim(0, 0.6)
    ax_tw.set_ylabel('Average attention', color=color_2, fontsize=18)
    ax_tw.tick_params(axis='y', labelcolor=color_2)

    def autolabel(rects, ax, color):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('%4.2f' % height,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', rotation=90, fontsize=13, color=color)

    autolabel(rects1, ax, color_1)
    autolabel(rects2, ax_tw, color_2)

    # fig.tight_layout()
    plt.savefig('./figure/The average precipitation for each month and corresponding average attention received during natural period.png',
                bbox_inches='tight')
    plt.show()

    '''
    --------------------------------------------------------------------------------------------------------------------
    The average precipitation distribution over the preceding 12 months for each predicted month in natural period
    --------------------------------------------------------------------------------------------------------------------
    '''
    config = {
        "font.size": 20,
    }
    rcParams.update(config)

    data_frame = pandas.read_excel('./data/rainfall_and_spring_flow.xlsx', sheet_name=0, keep_default_na=False)
    data_frame = data_frame.iloc[:, 2:9]
    data_np = np.array(data_frame)
    start_year = 1959
    end_year = 1970
    total_month = (end_year - start_year + 1) * 12
    start_index = (start_year - 1959) * 12
    end_index = (end_year - 1959 + 1) * 12
    data_np = data_np[start_index:end_index]
    data_list = []
    for i in range(0, total_month - 12):
        data_list.append(data_np[i:i + 12])
    data_np = np.concatenate(data_list, axis=0).reshape(-1, 12, 12, 7)
    data_avg = np.average(data_np, axis=0)
    precipitation_avg = np.average(data_avg, axis=-1)


    def polygon_under_graph(xlist, ylist):
        """
        Construct the vertex list which defines the polygon filling the space under
        the (xlist, ylist) line graph.  Assumes the xs are in ascending order.
        """
        return [(xlist[0], 0.), *zip(xlist, ylist), (xlist[-1], 0.)]

    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(projection='3d')

    # Make verts a list such that verts[i] is a list of (x, y) pairs defining
    # polygon i.
    verts = []

    # Set up the x sequence
    xs = np.linspace(0, 11, 12)

    # The ith polygon will appear on the plane y = zs[i]
    zs = range(precipitation_avg.shape[0])

    for i in zs:
        ys = precipitation_avg[i]
        verts.append(polygon_under_graph(xs, ys))

    color_list = ['#FA4F4F', '#F89B5C', '#FDBF08', '#FACD5D', '#C3F448', '#6AFBBC', '#62FAEE', '#34BCF9', '#4468FA',
                  '#874DF8', '#B649F8', '#7857A5']
    poly = LineCollection(verts, facecolors=color_list, alpha=0.5, colors=color_list)
    ax.add_collection3d(poly, zs=zs, zdir='y')

    # x_tick = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    x_tick = [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    ax.set_xlabel('\nPreceding 12 months', fontsize=25)
    ax.set_xticks(xs, x_tick)

    y_tick = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
    ax.set_ylabel('\nPredicted months', fontsize=25)
    ax.set_yticks(np.arange(0, 12), y_tick)

    ax.set_zlabel('\nAverage precipitation (mm)', fontsize=25)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 11)
    ax.set_zlim(0, 200)
    ax.view_init(elev=25, azim=-20)
    plt.savefig('./figure/The average precipitation distribution over the preceding 12 months for each predicted month in natural period.png', bbox_inches='tight')
    plt.show()

    '''
    --------------------------------------------------------------------------------------------------------------------
    average attention distribution over the preceding 12 months for each predicted month in natural period
    --------------------------------------------------------------------------------------------------------------------
    '''
    start_year = 1959
    end_year = 1970
    f_read = open('./result/long03_short12_w_t.pkl', 'rb')
    w_t = pickle.load(f_read)
    f_read.close()
    num_past_month = 12
    num_region = 7
    wt_training = np.stack(w_t['training'])
    wt_testing = np.stack(w_t['testing'])
    wt_tr_tes = np.concatenate([wt_training, wt_testing], axis=0)
    wt_tr_tes = np.concatenate(wt_tr_tes, axis=0).reshape(-1, 12, 12, 7)

    wt_tr_tes_avg = np.average(wt_tr_tes, axis=0)
    wt_tr_tes_avg = np.average(wt_tr_tes_avg, axis=-1)


    # normal_w = (wt_tr_tes_avg-wt_tr_tes_avg.min())/(wt_tr_tes_avg.max()-wt_tr_tes_avg.min())

    def polygon_under_graph(xlist, ylist):
        """
        Construct the vertex list which defines the polygon filling the space under
        the (xlist, ylist) line graph.  Assumes the xs are in ascending order.
        """
        return [(xlist[0], 0.), *zip(xlist, ylist), (xlist[-1], 0.)]
        # return [ *zip(xlist, ylist)]


    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(projection='3d')

    # Make verts a list such that verts[i] is a list of (x, y) pairs defining
    # polygon i.
    verts = []

    # Set up the x sequence
    xs = np.linspace(0, 11, 12)

    # The ith polygon will appear on the plane y = zs[i]
    zs = range(wt_tr_tes_avg.shape[0])

    for i in zs:
        ys = wt_tr_tes_avg[i]
        verts.append(polygon_under_graph(xs, ys))

    color_list = ['#FA4F4F', '#F89B5C', '#FDBF08', '#FACD5D', '#C3F448', '#6AFBBC', '#62FAEE', '#34BCF9', '#4468FA',
                  '#874DF8', '#B649F8', '#7857A5']
    poly = LineCollection(verts, facecolors=color_list, alpha=0.5, colors=color_list)
    ax.add_collection3d(poly, zs=zs, zdir='y')

    # x_tick = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    x_tick = [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    ax.set_xlabel('\nPreceding 12 months', fontsize=25)
    ax.set_xticks(xs, x_tick)

    y_tick = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sept', 'Oct', 'Nov', 'Dec']
    ax.set_ylabel('\nPredicted months', fontsize=25)
    ax.set_yticks(np.arange(0, 12), y_tick)

    ax.set_zlabel('\nAverage attention', fontsize=25)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 11)
    ax.set_zlim(0, 0.6)
    ax.view_init(elev=25, azim=-20)
    plt.savefig('./figure/average attention distribution over the preceding 12 months for each predicted month in natural period.png', bbox_inches='tight')
    plt.show()