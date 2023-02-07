import os
import numpy as np
import matplotlib.pyplot as plt
from pytorch_lightning import seed_everything
import warnings;warnings.simplefilter('ignore')


def generate_dataset(trace_num, tau, sample_num=None, is_print=False, sequence_length=None):

    if sequence_length is not None and os.path.exists(f"Data/data/tau_{tau}/train_{sequence_length}.npz") and os.path.exists(f"Data/data/tau_{tau}/val_{sequence_length}.npz") and os.path.exists(f"Data/data/tau_{tau}/test_{sequence_length}.npz"):
        return
    elif sequence_length is None and os.path.exists(f"Data/data/tau_{tau}/train.npz") and os.path.exists(f"Data/data/tau_{tau}/val.npz") and os.path.exists(f"Data/data/tau_{tau}/test.npz"):
        return
    
    # load original data
    if is_print: print('loading original trace data:')
    tmp = np.load(f"Data/origin/origin.npz")
    dt = tmp['dt']
    data = np.array(tmp['trace'])[:trace_num,:,np.newaxis] # (trace_num, time_length, channel, feature_num)
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
    train_num = int(0.5*trace_num)
    val_num = int(0.1*trace_num)
    test_num = int(0.4*trace_num)
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
            idxs = np.arange(0, np.shape(seq_data)[0]-step_length*sequence_length, 1)
            for idx_ in idxs:
                idxs_ic.append(ic)
                idxs_timestep.append(idx_)

        # generator item dataset
        sequences = []
        for bn in range(len(idxs_timestep)):
            idx_ic = idxs_ic[bn]
            idx_timestep = idxs_timestep[bn]
            tmp = data_item[idx_ic, idx_timestep : idx_timestep+step_length*sequence_length : step_length]
            sequences.append(tmp)
            if is_print: print(f'\rtau[{tau}] sliding window for {item} data [{bn+1}/{len(idxs_timestep)}]', end='')
        if is_print: print()

        sequences = np.array(sequences) 
        if is_print: print(f'tau[{tau}]', f"{item} dataset (sequence_length{sequence_length})", np.shape(sequences))

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

        # save item dataset
        if not seq_none:
            np.savez(data_dir+f'/{item}_{sequence_length}.npz', data=sequences)
        else:
            np.savez(data_dir+f'/{item}.npz', data=sequences)

        # plot
        if seq_none:
            plt.figure(figsize=(16,10))
            plt.title(f'{item.capitalize()} Data' + f' | sample_num[{len(sequences) if sample_num is None else sample_num}]')
            plt.plot(sequences[:,0,0,0], label='c1')
            plt.plot(sequences[:,0,0,1], label='c2')
            plt.plot(sequences[:,0,0,2], label='c3')
            plt.plot(sequences[:,0,0,3], label='c4')
            plt.legend()
            plt.savefig(data_dir+f'/{item}.pdf', dpi=300)
generate_dataset(10, 1.0, sample_num=None, is_print=True, sequence_length=10)

            
def generate_informer_dataset(trace_num, sample_num=None):
    
    # load original data
    print('loading original trace data:')
    tmp = np.load(f"Data/origin/origin.npz")
    simdata = np.array(tmp['trace'])[:trace_num,:,np.newaxis] # (trace_num, time_length, channel, feature_num)

    for tau in [0.1, 1.0, 5.0]:
        # subsampling
        dt = tmp['dt']
        subsampling = int(tau/dt) if tau!=0. else 1
        data = simdata[:, ::subsampling]
        print(f'tau[{tau}]', 'data shape', data.shape, '# (trace_num, time_length, channel, feature_num)')
        
        import pandas as pd
        data = np.concatenate(data, axis=0)[:,0]
        df = pd.DataFrame(data, columns=['c1','c2','c3','c4'])
        
        dt = pd.date_range('2016-07-01 00:00:00', periods=len(df), freq='h')
        df['date'] = dt
        df = df[['date','c1','c2','c3','c4']]
        
        df.to_csv(f'tau_{tau}.csv', index=False)

# generate_informer_dataset(trace_num=1000, sample_num=None)