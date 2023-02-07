import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Process
import warnings;warnings.simplefilter('ignore')

from .gillespie import generate_origin


def findNearestPoint(data_t, start=0, object_t=10.0):
    """Find the nearest time point to object time"""

    index = start

    if index >= len(data_t):
        return index

    while not (data_t[index] <= object_t and data_t[index+1] > object_t):
        if index < len(data_t)-2:
            index += 1
        elif index == len(data_t)-2: # last one
            index += 1
            break
    
    return index


def time_discretization(seed, total_t, dt=None, is_print=False):
    """Time-forward NearestNeighbor interpolate to discretizate the time"""

    data = np.load(f'Data/origin/{seed}/origin.npz')
    data_t = data['t']
    data_X = data['X']
    data_Y = data['Y']
    data_Z = data['Z']

    dt = 5e-6 if dt is None else dt # 5e-6是手动1s2f这个实验仿真得出的时间间隔大概平均值
    current_t = 0.0
    index = 0
    t, X, Y, Z = [], [], [], []
    while current_t < total_t:
        index = findNearestPoint(data_t, start=index, object_t=current_t)
        t.append(current_t)
        X.append(data_X[index])
        Y.append(data_Y[index])
        Z.append(data_Z[index])

        current_t += dt

        if is_print == 1: print(f'\rSeed[{seed}] interpolating {current_t:.6f}/{total_t}', end='')

    import scienceplots
    plt.style.use(['science'])
    plt.figure(figsize=(16,5))
    plt.rcParams.update({'font.size':16})
    # plt.title(f'dt = {dt}')
    plt.subplot(1,3,1)
    plt.plot(t, X, label=r'$X$')
    plt.xlabel(r'$t / s$', fontsize=18)
    plt.ylabel(r'$X$', fontsize=18)
    plt.subplot(1,3,2)
    plt.plot(t, Y, label=r'$Y$')
    plt.xlabel(r'$t / s$', fontsize=18)
    plt.ylabel(r'$Y$', fontsize=18)
    plt.subplot(1,3,3)
    plt.plot(t, Z, label=r'$Z$')
    plt.xlabel(r'$t / s$', fontsize=18)
    plt.ylabel(r'$Z$', fontsize=18)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.subplots_adjust(
        left=0.05,
        right=0.95,
        top=0.9,
        bottom=0.15,
        wspace=0.2
    )
    plt.savefig(f'Data/origin/{seed}/data.pdf', dpi=300)

    np.savez(f'Data/origin/{seed}/data.npz', dt=dt, t=t, X=X, Y=Y, Z=Z)
# time_discretization(1, total_t=15.1, dt=0.01)

def generate_original_data(trace_num, total_t):

    os.makedirs('Data/origin', exist_ok=True)

    # generate original data by gillespie algorithm
    subprocess = []
    for seed in range(1, trace_num+1):
        if not os.path.exists(f'Data/origin/{seed}/origin.npz'):
            IC = [np.random.randint(5,200), np.random.randint(5,100), np.random.randint(0,5000)]
            subprocess.append(Process(target=generate_origin, args=(total_t, seed, IC), daemon=True))
            subprocess[-1].start()
            # print(f'\rStart process[seed={seed}] for origin data' + ' '*30)
        else:
            pass
    while any([subp.exitcode == None for subp in subprocess]):
        pass
    print()
    
    # time discretization by time-forward NearestNeighbor interpolate
    subprocess = []
    for seed in range(1, trace_num+1):
        if not os.path.exists(f'Data/origin/{seed}/data.npz'):
            dt = 1e-2
            is_print = len(subprocess)==0
            subprocess.append(Process(target=time_discretization, args=(seed, total_t, dt, is_print), daemon=True))
            subprocess[-1].start()
            # print(f'\rStart process[seed={seed}] for time-discrete data' + ' '*30)
    while any([subp.exitcode == None for subp in subprocess]):
        pass

    print(f'save origin data form seed 1 to {trace_num} at Data/origin/')
    
    
def generate_dataset(trace_num, tau, sample_num=None, is_print=False, sequence_length=None, neural_ode=False):

    if not neural_ode and (sequence_length is not None) and os.path.exists(f"Data/data/tau_{tau}/train_{sequence_length}.npz") and os.path.exists(f"Data/data/tau_{tau}/val_{sequence_length}.npz") and os.path.exists(f"Data/data/tau_{tau}/test_{sequence_length}.npz"):
        return
    elif not neural_ode and (sequence_length is None) and os.path.exists(f"Data/data/tau_{tau}/train.npz") and os.path.exists(f"Data/data/tau_{tau}/val.npz") and os.path.exists(f"Data/data/tau_{tau}/test.npz"):
        return
    elif neural_ode and (sequence_length is None) and os.path.exists(f"Data/data/tau_{tau}/neural_ode_train.npz") and os.path.exists(f"Data/data/tau_{tau}/neural_ode_val.npz") and os.path.exists(f"Data/data/tau_{tau}/neural_ode_test.npz"):
        return

    # load original data
    if is_print: print('loading original trace data:')
    data = []
    iter = tqdm(range(1, trace_num+1)) if is_print else range(1, trace_num+1)
    for trace_id in iter:
        tmp = np.load(f"Data/origin/{trace_id}/data.npz")
        dt = tmp['dt']
        X = np.array(tmp['X'])[:, np.newaxis, np.newaxis] # (sample_num, channel, feature_num)
        Y = np.array(tmp['Y'])[:, np.newaxis, np.newaxis]
        Z = np.array(tmp['Z'])[:, np.newaxis, np.newaxis]

        trace = np.concatenate((X, Y, Z), axis=-1)
        data.append(trace[np.newaxis])
    data = np.concatenate(data, axis=0)

    if is_print: print(f'tau[{tau}]', 'data shape', data.shape, '# (trace_num, time_length, channel, feature_num)')

    # save statistic information
    data_dir = f"Data/data/tau_{tau}"
    os.makedirs(data_dir, exist_ok=True)
    np.savetxt(data_dir + "/data_mean.txt", np.mean(data, axis=(0,1)))
    np.savetxt(data_dir + "/data_std.txt", np.std(data, axis=(0,1)))
    np.savetxt(data_dir + "/data_max.txt", np.max(data, axis=(0,1)))
    np.savetxt(data_dir + "/data_min.txt", np.min(data, axis=(0,1)))
    np.savetxt(data_dir + "/tau.txt", [tau]) # Save the timestep

    # single-sample time steps
    if sequence_length is None:
        sequence_length = 2 if tau != 0. else 1
        seq_none = True
    else:
        seq_none = False
    
    ##################################
    # Create [train,val,test] dataset
    ##################################
    train_num = int(0.7*trace_num)
    val_num = int(0.1*trace_num)
    test_num = int(0.2*trace_num)
    trace_list = {'train':range(train_num), 'val':range(train_num,train_num+val_num), 'test':range(train_num+val_num,train_num+val_num+test_num)}
    for item in ['train','val','test']:
                
        # select trace num
        N_TRACE = len(trace_list[item])
        data_item = data[trace_list[item]]

        # subsampling
        step_length = int(tau/dt) if tau!=0. else 1

        # select sliding window index from N trace
        idxs_timestep = []
        idxs_ic = []
        for ic in range(N_TRACE):
            seq_data = data_item[ic]
            idxs = np.arange(0, np.shape(seq_data)[0]-step_length*(sequence_length-1), 1)
            for idx_ in idxs:
                idxs_ic.append(ic)
                idxs_timestep.append(idx_)

        # generator item dataset
        sequences = []
        parallel_sequences = [[] for _ in range(N_TRACE)]
        for bn in range(len(idxs_timestep)):
            idx_ic = idxs_ic[bn]
            idx_timestep = idxs_timestep[bn]
            tmp = data_item[idx_ic, idx_timestep : idx_timestep+step_length*(sequence_length-1)+1 : step_length]
            sequences.append(tmp)
            parallel_sequences[idx_ic].append(tmp)
            if is_print: print(f'\rtau[{tau}] sliding window for {item} data [{bn+1}/{len(idxs_timestep)}]', end='')
        if is_print: print()

        sequences = np.array(sequences) 
        if is_print: print(f'tau[{tau}]', f"{item} dataset (sequence_length={sequence_length})", np.shape(sequences))

        # keep sequences_length equal to sample_num
        if sample_num is not None:
            repeat_num = int(np.floor(N_TRACE*sample_num/len(sequences)))
            idx = np.random.choice(range(len(sequences)), N_TRACE*sample_num-len(sequences)*repeat_num, replace=False)
            idx = np.sort(idx)
            tmp1 = sequences[idx]
            tmp2 = None
            for i in range(repeat_num):
                if i == 0:
                    tmp2 = sequences
                else:
                    tmp2 = np.concatenate((tmp2, sequences), axis=0)
            sequences = tmp1 if tmp2 is None else np.concatenate((tmp1, tmp2), axis=0)
        if is_print: print(f'tau[{tau}]', f"after process", np.shape(sequences))

        # save
        if neural_ode:
            for i in range(len(parallel_sequences)):
                parallel_sequences[i] = np.array(parallel_sequences[i])
            parallel_sequences = np.array(parallel_sequences)
            np.savez(data_dir+f'/neural_ode_{item}.npz', data=parallel_sequences[:,:,0])
        else:
            if not seq_none:
                np.savez(data_dir+f'/{item}_{sequence_length}.npz', data=sequences)
            else:
                np.savez(data_dir+f'/{item}.npz', data=sequences)

            # plot
            if seq_none:
                plt.figure(figsize=(16,10))
                plt.title(f'{item.capitalize()} Data' + f' | sample_num[{len(sequences) if sample_num is None else sample_num}]')
                for i in range(3):
                    ax = plt.subplot(3,1,i+1)
                    ax.set_title(['X','Y','Z'][i])
                    plt.plot(sequences[:, 0, 0, i])
                plt.subplots_adjust(left=0.05, bottom=0.05,  right=0.95,  top=0.95,  hspace=0.35)
                plt.savefig(data_dir+f'/{item}_input.pdf', dpi=300)

                plt.figure(figsize=(16,10))
                plt.title(f'{item.capitalize()} Data' + f' | sample_num[{len(sequences) if sample_num is None else sample_num}]')
                for i in range(3):
                    ax = plt.subplot(3,1,i+1)
                    ax.set_title(['X','Y','Z'][i])
                    plt.plot(sequences[:, sequence_length-1, 0, i])
                plt.subplots_adjust(left=0.05, bottom=0.05,  right=0.95,  top=0.95,  hspace=0.35)
                plt.savefig(data_dir+f'/{item}_target.pdf', dpi=300)
            
        
def generate_informer_dataset(trace_num, sample_num=None):
    
    # load original data
    simdata = []
    for trace_id in tqdm(range(1, trace_num+1)):
        tmp = np.load(f"Data/origin/{trace_id}/data.npz")
        X = np.array(tmp['X'])[:, np.newaxis, np.newaxis] # (sample_num, channel, feature_num)
        Y = np.array(tmp['Y'])[:, np.newaxis, np.newaxis]
        Z = np.array(tmp['Z'])[:, np.newaxis, np.newaxis]

        trace = np.concatenate((X, Y, Z), axis=-1)
        simdata.append(trace[np.newaxis])
    simdata = np.concatenate(simdata, axis=0)

    for tau in [0.3, 3.0, 15.0]:
        # subsampling
        dt = tmp['dt']
        subsampling = int(tau/dt) if tau!=0. else 1
        data = simdata[:, ::subsampling]
        print(f'tau[{tau}]', 'data shape', data.shape, '# (trace_num, time_length, channel, feature_num)')
        
        import pandas as pd
        data = np.concatenate(data, axis=0)[:,0]
        df = pd.DataFrame(data, columns=['X','Y','Z'])
        
        dt = pd.date_range('2016-07-01 00:00:00', periods=len(df), freq='h')
        df['date'] = dt
        df = df[['date','X','Y','Z']]
        
        df.to_csv(f'tau_{tau}.csv', index=False)
# generate_informer_dataset(trace_num=100, sample_num=None)