# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from torch import nn
from torch.nn.utils import weight_norm
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Process
from pytorch_lightning import seed_everything

from util import set_cpu_num
from Data.dataset import _1S2FDataset
from Data.generator import generate_dataset


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.tanh = nn.Tanh()
        
        # scale inside the model
        self.register_buffer('min', torch.zeros(1, input_size, dtype=torch.float32))
        self.register_buffer('max', torch.ones(1, input_size, dtype=torch.float32))

    def forward(self, x):
        # (batchsize,sequence_length,1,3)-->(batchsize,sequence_length,3)
        x = nn.Flatten(start_dim=-2)(x)
        
        # (batchsize,sequence_length,3)-->(batchsize,3,sequence_length)
        x = x.transpose(2,1).contiguous()
        
        # (batchsize,sequence_length,3)-->(batchsize,sequence_length,1,3)
        y = self.tcn(x).transpose(2,1).contiguous()
        y = self.tanh(self.linear(y))
        y = y.unsqueeze(-2)
        
        return y
        
    def scale(self, x):
        return (x-self.min) / (self.max-self.min+1e-6)
    
    def descale(self, x):
        return x * (self.max-self.min+1e-6) + self.min


def train(tau, delta_t, is_print=False, random_seed=729):
        
    # prepare
    device = torch.device('cpu')
    data_filepath = 'Data/data/tau_' + str(delta_t)
    log_dir = f'logs/tcn/tau_{tau}/seed{random_seed}'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_dir+"/checkpoints/", exist_ok=True)

    # init model
    model = TCN(input_size=3, output_size=3, num_channels=[32,16,8], kernel_size=3, dropout=0.1)
    model.min = torch.from_numpy(np.loadtxt(data_filepath+"/data_min.txt").astype(np.float32)).unsqueeze(0)
    model.max = torch.from_numpy(np.loadtxt(data_filepath+"/data_max.txt").astype(np.float32)).unsqueeze(0)
    model = model.to(device)
    
    # training params
    lr = 0.001
    batch_size = 32
    max_epoch = 50
    weight_decay = 0.001
    MSE_loss = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # dataset
    train_dataset = _1S2FDataset(data_filepath, 'train', length=n)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dataset = _1S2FDataset(data_filepath, 'val')
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    # training pipeline
    train_loss = []
    best_loss = 1.
    for epoch in range(1, max_epoch+1):
        
        losses = []
        
        # train
        model.train()
        # for input, target in train_loader:
        for input, _, internl_units in train_loader:
            
            input = model.scale(input.to(device)) # (batchsize,1,1,3)
            # target = model.scale(target.to(device))

            loss = 0
            for i in range(1, len(internl_units)):
                
                unit = model.scale(internl_units[i].to(device)) # t+i
                output = model(input)
                for _ in range(1, n):
                    input = torch.concat([input[:,1:], output[:,0].unsqueeze(1)], dim=1)
                    output = model(input)
                
                loss += MSE_loss(output, unit)
            
            # output = model(input)
            # loss = MSE_loss(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # record loss
            losses.append(loss.detach().item())
            
        train_loss.append(np.mean(losses[0]))
        
        # validate
        with torch.no_grad():
            targets = []
            outputs = []
            
            model.eval()
            for input, target in val_loader:
                
                input = model.scale(input.to(device)) # (batchsize,1,1,3)
                target = model.scale(target.to(device))
                
                output = model(input)

                # record results
                outputs.append(output.cpu())
                targets.append(target.cpu())
            
            # trans to tensor
            outputs = torch.concat(outputs, axis=0)
            targets = torch.concat(targets, axis=0)
            
            # cal loss
            loss = MSE_loss(outputs, targets)
            if is_print: print(f'\rTau[{tau}] | epoch[{epoch}/{max_epoch}] | val-mse={loss:.5f}', end='')
                        
            # plot per 5 epoch
            if epoch % 5 == 0:
                
                os.makedirs(log_dir+f"/val/epoch-{epoch}/", exist_ok=True)
                
                # plot total infomation one-step prediction curve
                plt.figure(figsize=(16,5))
                for j, item in enumerate(['X','Y','Z']):
                    ax = plt.subplot(1,3,j+1)
                    ax.set_title(item)
                    plt.plot(targets[:,0,0,j], label='true')
                    plt.plot(outputs[:,0,0,j], label='predict')
                plt.subplots_adjust(wspace=0.2)
                plt.savefig(log_dir+f"/val/epoch-{epoch}/predict.pdf", dpi=300)
                plt.close()
        
            # record best model
            if loss < best_loss:
                best_loss = loss
                best_model = model.state_dict()

    # save model
    torch.save(best_model, log_dir+f"/checkpoints/epoch-{epoch}.ckpt")
    if is_print: print(f'\nsave best model at {log_dir}/checkpoints/epoch-{epoch}.ckpt (val best_loss={best_loss})')
    
    # plot loss curve
    train_loss = np.array(train_loss)
    plt.figure()
    plt.plot(train_loss)
    plt.xlabel('epoch')
    plt.title('Training Loss Curve')
    plt.savefig(log_dir+'/train_loss_curve.pdf', dpi=300)
    

def test_evolve(tau, ckpt_epoch, delta_t, n, is_print=False, random_seed=729):
        
    # prepare
    device = torch.device('cpu')
    data_filepath = 'Data/data/tau_' + str(delta_t)
    log_dir = f'logs/tcn/tau_{tau}/seed{random_seed}'
    os.makedirs(log_dir+f"/test/", exist_ok=True)

    # load model
    batch_size = 32
    model = TCN(input_size=3, output_size=3, num_channels=[32,16,8], kernel_size=3, dropout=0.1)
    ckpt_path = log_dir+f'/checkpoints/epoch-{ckpt_epoch}.ckpt'
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt)
    model = model.to(device)
    
    # dataset
    test_dataset = _1S2FDataset(data_filepath, 'test')
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    # testing pipeline        
    with torch.no_grad():
        targets = []
        outputs = []
        
        model.eval()
        for input, target in test_loader:
            input = model.scale(input.to(device))
            target = model.scale(target.to(device))

            output = model(input)
            for _ in range(1, n):
                input = torch.concat([input[:,1:], output[:,0].unsqueeze(1)], dim=1)
                output = model(input)

            targets.append(target)
            outputs.append(output)
        
        targets = torch.concat(targets, axis=0)
        outputs = torch.concat(outputs, axis=0)
        
    # metrics
    pred = outputs.detach().cpu().numpy()
    true = targets.detach().cpu().numpy()
    MAPE = np.mean(np.abs((pred - true) / true))
    targets = model.descale(targets)
    outputs = model.descale(outputs)
    pred = outputs.detach().cpu().numpy()
    true = targets.detach().cpu().numpy()
    MSE = np.mean((pred - true) ** 2)
    RMSE = np.sqrt(MSE)
    MAE = np.mean(np.abs(pred - true))
    
    # plot total infomation prediction curve
    plt.figure(figsize=(16,5))
    for j, item in enumerate(['X','Y','Z']):
        ax = plt.subplot(1,3,j+1)
        ax.set_title(item)
        plt.plot(true[:,0,0,j], label='true')
        plt.plot(pred[:,0,0,j], label='predict')
    plt.subplots_adjust(wspace=0.2)
    plt.savefig(log_dir+f"/test/predict_{delta_t}.pdf", dpi=300)
    plt.close()
    
    return MSE, RMSE, MAE, MAPE


def main(trace_num, tau, n, is_print=False, long_test=False, random_seed=729):
    
    seed_everything(random_seed)
    set_cpu_num(1)
    
    sample_num = None
    
    if not long_test:
        # train
        train(tau, round(tau/n,3), is_print=is_print, random_seed=random_seed)
    else:
        # test evolve
        ckpt_epoch = 50
        for i in tqdm(range(1, 25*n+1)):
            delta_t = round(tau/n*i, 3)
            generate_dataset(trace_num, delta_t, sample_num, False)
            MSE, RMSE, MAE, MAPE = test_evolve(tau, ckpt_epoch, delta_t, i, is_print, random_seed)
            with open(f'results/tcn_evolve_test_{tau}.txt','a') as f:
                f.writelines(f'{round(tau/n*i, 3)}, {random_seed}, {MSE}, {RMSE}, {MAE}, {MAPE}\n')


if __name__ == '__main__':
    
    trace_num = 100
    
    workers = []
    
    tau = 3.0
    n = 10
    
    # train
    sample_num = None
    generate_dataset(trace_num, round(tau/n, 3), sample_num, False, n)
    for seed in range(1,10+1):
        is_print = True if len(workers)==0 else False
        workers.append(Process(target=main, args=(trace_num, tau, n, is_print, False, seed), daemon=True))
        workers[-1].start()
    while any([sub.exitcode==None for sub in workers]):
        pass
    workers = []
    
    # test
    for seed in range(1,10+1):
        main(trace_num, tau, n, True, True, seed)
    #     is_print = True if len(workers)==0 else False
    #     workers.append(Process(target=main, args=(trace_num, tau, n, is_print, True, seed), daemon=True))
    #     workers[-1].start()
    # while any([sub.exitcode==None for sub in workers]):
    #     pass
    # workers = []
    
    torch.cuda.empty_cache()