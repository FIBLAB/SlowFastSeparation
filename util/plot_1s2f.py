import numpy as np
import pandas as pd
import scienceplots
import matplotlib.pyplot as plt;plt.style.use(['science'])


def plot_epoch_test_log(tau, max_epoch):

    class MSE():
        def __init__(self, tau):
            self.tau = tau
            self.mse_x = [[] for _ in range(max_epoch)]
            self.mse_y = [[] for _ in range(max_epoch)]
            self.mse_z = [[] for _ in range(max_epoch)]
            self.LB_id = [[] for _ in range(max_epoch)]

    fp = open(f'logs/1S2F/TimeSelection/tau_{tau}/test_log.txt', 'r')
    items = []
    for line in fp.readlines():
        tau = float(line[:-1].split(',')[0])
        seed = int(line[:-1].split(',')[1])
        mse_x = float(line[:-1].split(',')[2])
        mse_y = float(line[:-1].split(',')[3])
        mse_z = float(line[:-1].split(',')[4])
        epoch = int(line[:-1].split(',')[5])
        LB_id = float(line[:-1].split(',')[6])

        find = False
        for M in items:
            if M.tau == tau:
                M.mse_x[epoch].append(mse_x)
                M.mse_y[epoch].append(mse_y)
                M.mse_z[epoch].append(mse_z)
                M.LB_id[epoch].append(LB_id)
                find = True
                    
        if not find:
            M = MSE(tau)
            M.mse_x[epoch].append(mse_x)
            M.mse_y[epoch].append(mse_y)
            M.mse_z[epoch].append(mse_z)
            M.LB_id[epoch].append(LB_id)
            items.append(M)
    fp.close()

    for M in items:
        mse_x_list = []
        mse_y_list = []
        mse_z_list = []
        LB_id_list = []
        MiND_id_list = []
        MADA_id_list = []
        PCA_id_list = []
        for epoch in range(max_epoch):
            mse_x_list.append(np.mean(M.mse_x[epoch]))
            mse_y_list.append(np.mean(M.mse_y[epoch]))
            mse_z_list.append(np.mean(M.mse_z[epoch]))
            LB_id_list.append(np.mean(M.LB_id[epoch]))

    plt.figure(figsize=(12,9))
    plt.title(f'tau = {M.tau}')
    ax1 = plt.subplot(2,1,1)
    plt.xlabel('epoch')
    plt.ylabel('ID')
    plt.plot(range(max_epoch), LB_id_list)
    ax2 = plt.subplot(2,1,2)
    plt.xlabel('epoch')
    plt.ylabel('MSE')
    plt.plot(range(max_epoch), mse_x_list, label='x')
    plt.plot(range(max_epoch), mse_y_list, label='y')
    plt.plot(range(max_epoch), mse_z_list, label='z')
    # plt.ylim((0., 1.05*max(np.max(mse_x_list), np.max(mse_y_list), np.max(mse_z_list))))
    plt.legend()
    plt.savefig(f'logs/1S2F/TimeSelection/tau_{tau}/ID_per_epoch.pdf', dpi=300)
    plt.close()


def plot_id_per_tau(tau_list, id_epoch):

    id_per_tau = [[] for _ in tau_list]
    for i, tau in enumerate(tau_list):
        fp = open(f'logs/1S2F/TimeSelection/tau_{round(tau,2)}/test_log.txt', 'r')
        for line in fp.readlines():
            seed = int(line[:-1].split(',')[1])
            epoch = int(line[:-1].split(',')[5])
            LB_id = float(line[:-1].split(',')[6])

            if epoch in id_epoch:
                id_per_tau[i].append([LB_id])
    
    for i in range(len(tau_list)):
        id_per_tau[i] = np.mean(id_per_tau[i], axis=0)
    id_per_tau = np.array(id_per_tau)

    round_id_per_tau = []
    for id in id_per_tau:
        round_id_per_tau.append([round(id[0])])
    round_id_per_tau = np.array(round_id_per_tau)

    plt.figure(figsize=(6,6))
    plt.rcParams.update({'font.size':16})
    for i, item in enumerate(['MLE']):
        plt.plot(tau_list, id_per_tau[:,i], marker="o", markersize=6, label="ID")
        plt.plot(tau_list, round_id_per_tau[:,i], marker="^", markersize=6, label="ID-rounding")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()
    plt.xlabel(r'$\tau / s$', fontsize=18)
    plt.ylabel('Intrinsic dimensionality', fontsize=18)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig('logs/1S2F/TimeSelection/id_per_tau.pdf', dpi=300)


def plot_1s2f_autocorr():

    data = np.load('Data/1S2F/origin/1/data.npz')
    X = np.array(data['X'])[:, np.newaxis]
    Y = np.array(data['Y'])[:, np.newaxis]
    Z = np.array(data['Z'])[:, np.newaxis]

    data = pd.DataFrame(np.concatenate((X,Y,Z), axis=-1), columns=['X', 'Y', 'Z'])
    
    corrX, corrY, corrZ = [], [], []
    lag_list = np.arange(0, 7*100, 30)
    from tqdm import tqdm
    for lag in tqdm(lag_list):
        corrX.append(data['X'].autocorr(lag=lag))
        corrY.append(data['Y'].autocorr(lag=lag))
        corrZ.append(data['Z'].autocorr(lag=lag))
    import scienceplots
    plt.style.use(['science'])
    plt.figure(figsize=(6,6))
    plt.rcParams.update({'font.size':16})
    plt.plot(lag_list*1e-2, np.array(corrX), marker="o", markersize=6, label=r'$X$')
    plt.plot(lag_list*1e-2, np.array(corrY), marker="^", markersize=6, label=r'$Y$')
    plt.plot(lag_list*1e-2, np.array(corrZ), marker="D", markersize=6, label=r'$Z$')
    plt.xlabel(r'$t/s$', fontsize=18)
    plt.ylabel('Autocorrelation coefficient', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()
    # plt.subplots_adjust(bottom=0.15, left=0.2)
    plt.savefig('corr.pdf', dpi=300)


def plot_evolve(length):
    
    our = open(f'Results/1S2F/ours_evolve_test_{length}.txt', 'r')
    lstm = open(f'Results/1S2F/lstm_evolve_test_{length}.txt', 'r')
    tcn = open(f'Results/1S2F/tcn_evolve_test_{length}.txt', 'r')
    ode = open(f'Results/1S2F/neural_ode_evolve_test_{length}.txt', 'r')
    
    our_data = [[] for seed in range(10)]
    lstm_data = [[] for seed in range(10)]
    tcn_data = [[] for seed in range(10)]
    ode_data = [[] for seed in range(10)]
    for i, data in enumerate([our, lstm, tcn, ode]):
        for line in data.readlines():
            tau = float(line.split(',')[0])
            seed = int(line.split(',')[1])
            mse = float(line.split(',')[2])
            rmse = float(line.split(',')[3])
            mae = float(line.split(',')[4])
            mape = float(line.split(',')[5])
            
            if i==0:
                our_data[seed-1].append([tau,mse,rmse,mae,mape])
            elif i==1:
                lstm_data[seed-1].append([tau,mse,rmse,mae,mape])
            elif i==2:
                tcn_data[seed-1].append([tau,mse,rmse,mae,mape])
            elif i==3:
                ode_data[seed-1].append([tau,mse,rmse,mae,mape])
    
    our_data = np.mean(np.array(our_data), axis=0)
    lstm_data = np.mean(np.array(lstm_data), axis=0)
    tcn_data = np.mean(np.array(tcn_data), axis=0)
    ode_data = np.mean(np.array(ode_data), axis=0)
    
    plt.figure(figsize=(16,16))
    for i, item in enumerate(['mse', 'rmse', 'mae', 'mape']):
        ax = plt.subplot(2,2,i+1)
        ax.plot(our_data[:,0], our_data[:,i+1], label='our')
        ax.plot(lstm_data[:,0], lstm_data[:,i+1], label='lstm')
        ax.plot(tcn_data[:,0], tcn_data[:,i+1], label='tcn')
        ax.plot(ode_data[:,0], ode_data[:,i+1], label='ode')
        ax.set_title(item)
        ax.set_xlabel('t / s')
        ax.legend()
    plt.savefig(f'Results/1S2F/evolve_test_{length}.pdf', dpi=300)
    
    item = ['our','lstm','tcn', 'ode']
    for i, data in enumerate([our_data, lstm_data, tcn_data, ode_data]):
        print(f'{item[i]} | tau[{data[0,0]:.1f}] RMSE={data[0,2]:.4f}, MAE={data[0,3]:.4f}, MAPE={100*data[0,4]:.2f}% | \
              tau[{data[9,0]:.1f}] RMSE={data[9,2]:.4f}, MAE={data[9,3]:.4f}, MAPE={100*data[9,4]:.2f}% | \
              tau[{data[49,0]:.1f}] RMSE={data[49,2]:.4f}, MAE={data[49,3]:.4f}, MAPE={100*data[49,4]:.2f}%')


if __name__ == '__main__':
    
    # plot_1s2f_autocorr()
    plot_evolve(3.0)
