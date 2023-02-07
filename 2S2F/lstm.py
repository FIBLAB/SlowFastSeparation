# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Process
from pytorch_lightning import seed_everything

from util import set_cpu_num
from Data.dataset import _2S2FDataset
from Data.generator import generate_dataset


class LSTM(nn.Module):
    
    def __init__(self, in_channels, input_1d_width, hidden_dim=64, layer_num=2):
        super(LSTM, self).__init__()
        
        # (batchsize,1,1,3)-->(batchsize,1,3)
        self.flatten = nn.Flatten(start_dim=2)
        
        # (batchsize,1,3)-->(batchsize, hidden_dim)
        self.layer_num = layer_num
        self.hidden_dim = hidden_dim
        self.cell = nn.LSTM(
            input_size=in_channels*input_1d_width, 
            hidden_size=hidden_dim, 
            num_layers=layer_num, 
            dropout=0.01, 
            batch_first=True # input: (batch_size, squences, features)
            )
        
        # (batchsize, hidden_dim)-->(batchsize, 3)
        self.fc = nn.Linear(hidden_dim, in_channels*input_1d_width)
        
        # (batchsize, 3)-->(batchsize,1,1,3)
        self.unflatten = nn.Unflatten(-1, (1, in_channels, input_1d_width))

        # scale inside the model
        self.register_buffer('min', torch.zeros(in_channels, input_1d_width, dtype=torch.float32))
        self.register_buffer('max', torch.ones(in_channels, input_1d_width, dtype=torch.float32))
    
    def forward(self, x, device=torch.device('cuda:1')):
        
        h0 = torch.zeros(self.layer_num * 1, len(x), self.hidden_dim, dtype=torch.float32, device=device)
        c0 = torch.zeros(self.layer_num * 1, len(x), self.hidden_dim, dtype=torch.float32, device=device)
        
        x = self.flatten(x)
        _, (h, c)  = self.cell(x, (h0, c0))
        y = self.fc(h[-1])
        y = self.unflatten(y)
        
        return y
    
    def scale(self, x):
        return (x-self.min) / (self.max-self.min+1e-6)
    
    def descale(self, x):
        return x * (self.max-self.min+1e-6) + self.min


def train(tau, delta_t, is_print=False, random_seed=729):
        
    # prepare
    device = torch.device('cuda:1')
    data_filepath = 'Data/data/tau_' + str(delta_t)
    log_dir = f'logs/lstm/tau_{tau}/seed{random_seed}'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_dir+"/checkpoints/", exist_ok=True)

    # init model
    model = LSTM(in_channels=1, input_1d_width=4)
    model.min = torch.from_numpy(np.loadtxt(data_filepath+"/data_min.txt").astype(np.float32)).unsqueeze(0)
    model.max = torch.from_numpy(np.loadtxt(data_filepath+"/data_max.txt").astype(np.float32)).unsqueeze(0)
    model.to(device)
    
    # training params
    lr = 0.001
    batch_size = 128
    max_epoch = 50
    weight_decay = 0.001
    MSE_loss = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # dataset
    train_dataset = _2S2FDataset(data_filepath, 'train', length=n)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dataset = _2S2FDataset(data_filepath, 'val')
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
                output = model(input, device)
                for _ in range(1,i):
                    output = model(output, device)
                
                loss += MSE_loss(output, unit)
            
            # output = model(input, device)
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
                
                input = model.scale(input.to(device)) # (batchsize,1,1,4)
                target = model.scale(target.to(device))
                
                output = model(input, device)

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
                plt.figure(figsize=(16,4))
                for j, item in enumerate(['c1','c2','c3', 'c4']):
                    ax = plt.subplot(1,4,j+1)
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
    device = torch.device('cuda:1')
    data_filepath = 'Data/data/tau_' + str(delta_t)
    log_dir = f'logs/lstm/tau_{tau}/seed{random_seed}'
    os.makedirs(log_dir+f"/test/", exist_ok=True)

    # load model
    batch_size = 128
    model = LSTM(in_channels=1, input_1d_width=4)
    ckpt_path = log_dir+f'/checkpoints/epoch-{ckpt_epoch}.ckpt'
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt)
    model = model.to(device)
    
    # dataset
    test_dataset = _2S2FDataset(data_filepath, 'test')
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    # testing pipeline        
    with torch.no_grad():
        targets = []
        outputs = []
        
        model.eval()
        for input, target in test_loader:
            input = model.scale(input.to(device))
            target = model.scale(target.to(device))

            output = model(input, device)
            for _ in range(1, n):
                output = model(output, device)

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
    plt.figure(figsize=(16,4))
    for j, item in enumerate(['c1','c2','c3', 'c4']):
        ax = plt.subplot(1,4,j+1)
        ax.set_title(item)
        plt.plot(true[:,0,0,j], label='true')
        plt.plot(pred[:,0,0,j], label='predict')
    plt.subplots_adjust(wspace=0.2)
    plt.savefig(log_dir+f"/test/predict_{delta_t}.pdf", dpi=300)
    plt.close()

    c1_evolve_mae = np.mean(np.abs(pred[:,0,0,0] - true[:,0,0,0]))
    c2_evolve_mae = np.mean(np.abs(pred[:,0,0,1] - true[:,0,0,1]))
    
    return MSE, RMSE, MAE, MAPE, c1_evolve_mae, c2_evolve_mae


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
        for i in tqdm(range(1, 6*n+1+2)):
            delta_t = round(tau/n*i, 3)
            generate_dataset(trace_num, delta_t, sample_num, False)
            MSE, RMSE, MAE, MAPE, c1_mae, c2_mae = test_evolve(tau, ckpt_epoch, delta_t, i, is_print, random_seed)
            with open(f'results/lstm_evolve_test_{tau}.txt','a') as f:
                f.writelines(f'{delta_t}, {random_seed}, {MSE}, {RMSE}, {MAE}, {MAPE}, {c1_mae}, {c2_mae}\n')


if __name__ == '__main__':
    
    trace_num = 200
    
    workers = []
    
    tau = 0.8
    n = 8
    
    # train
    sample_num = None
    generate_dataset(trace_num, round(tau/n, 3), sample_num, False)
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