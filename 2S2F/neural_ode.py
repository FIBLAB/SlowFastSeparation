# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Process
from pytorch_lightning import seed_everything
if True:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

from util import set_cpu_num
from Data.dataset import _2S2FDataset
from Data.generator import generate_dataset


class NeuralODEfunc(nn.Module):

    def __init__(self, obs_dim, nhidden=64):
        super(NeuralODEfunc, self).__init__()
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(obs_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, obs_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.tanh(out)
        out = self.fc3(out)
        return out


class Neural_ODE(nn.Module):

    def __init__(self, in_channels, input_1d_width, nhidden=64):
        super(Neural_ODE, self).__init__()

        self.ode = NeuralODEfunc(in_channels*input_1d_width, nhidden)

        self.flatten = nn.Flatten(start_dim=-2)
        self.unflatten = nn.Unflatten(-1, (in_channels, input_1d_width))

        # scale inside the model
        self.register_buffer('min', torch.zeros(in_channels, input_1d_width, dtype=torch.float32))
        self.register_buffer('max', torch.ones(in_channels, input_1d_width, dtype=torch.float32))
    
    def evolve(self, x0, t):
        x0 = self.flatten(x0)[:,0]
        out = odeint(self.ode, x0, t).permute(1, 0, 2)
        out = self.unflatten(out)
        return out

    def scale(self, x):
        return (x-self.min) / (self.max-self.min+1e-6)
    
    def descale(self, x):
        return x * (self.max-self.min+1e-6) + self.min


def train(tau, delta_t, is_print=False, random_seed=729):
        
    # prepare
    device = torch.device('cuda:1')
    data_filepath = 'Data/data/tau_' + str(delta_t)
    log_dir = f'logs/neural_ode/tau_{tau}/seed{random_seed}'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_dir+"/checkpoints/", exist_ok=True)

    # init model
    model = Neural_ODE(in_channels=1, input_1d_width=4)
    model.min = torch.from_numpy(np.loadtxt(data_filepath+"/data_min.txt").astype(np.float32)).unsqueeze(0)
    model.max = torch.from_numpy(np.loadtxt(data_filepath+"/data_max.txt").astype(np.float32)).unsqueeze(0)
    model.to(device)
    
    # training params
    lr = 0.001
    batch_size = 128
    max_epoch = 5
    weight_decay = 0.001
    MSE_loss = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # dataset
    train_dataset = _2S2FDataset(data_filepath, 'train', length=n)
    # train_dataset = _2S2FDataset(data_filepath, 'train')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dataset = _2S2FDataset(data_filepath, 'val', length=n)
    # val_dataset = _2S2FDataset(data_filepath, 'val')
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    # training pipeline
    train_loss = []
    for epoch in range(1, max_epoch+1):
        
        losses = []
        
        # train
        model.train()
        for input, _, internl_units in train_loader:
        # for input, target in train_loader:
            
            input = model.scale(input.to(device)) # (batchsize,1,1,3)
            # target = model.scale(target.to(device)) # (batchsize,1,1,3)

            loss = 0
            t = torch.tensor([delta_t], device=device) # delta_t
            for i in range(1, len(internl_units)):
                unit = model.scale(internl_units[i].to(device)) # t+i           
                output = model.evolve(input, t)
                for _ in range(1, i):
                    output = model.evolve(output, t)
                loss += MSE_loss(output, unit)
            # t = torch.tensor([delta_t], device=device) # delta_t
            # output = model.evolve(input, t)
            # loss = MSE_loss(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # record loss
            losses.append((loss/n).detach().item())
            
        train_loss.append(np.mean(losses[0]))
        
        # validate
        with torch.no_grad():
            targets = []
            outputs = []
            
            model.eval()
            for input, target, _ in val_loader:
            # for input, target in val_loader:
                
                input = model.scale(input.to(device)) # (batchsize,1,1,4)
                target = model.scale(target.to(device))
                
                t = torch.tensor([tau-delta_t], dtype=torch.float32, device=device)
                # t = torch.tensor([delta_t], dtype=torch.float32, device=device)
                output = model.evolve(input, t)

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
        
            # save model
            torch.save(model.state_dict(), log_dir+f"/checkpoints/epoch-{epoch}.ckpt")
    
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
    log_dir = f'logs/neural_ode/tau_{tau}/seed{random_seed}'
    os.makedirs(log_dir+f"/test/", exist_ok=True)

    # load model
    batch_size = 128
    model = Neural_ODE(in_channels=1, input_1d_width=4)
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

            t = torch.tensor([delta_t/n], dtype=torch.float32, device=device)
            output = model.evolve(input, t)
            for _ in range(1, n):
                output = model.evolve(output, t)
            
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

    if not long_test:
        # train
        train(tau, round(tau/n,3), is_print=is_print, random_seed=random_seed)
    else:
        # test evolve
        ckpt_epoch = 5
        for i in tqdm(range(1, 6*n+1+2)):
            delta_t = round(tau/n*i, 3)
            MSE, RMSE, MAE, MAPE, c1_mae, c2_mae = test_evolve(tau, ckpt_epoch, delta_t, i, is_print, random_seed)
            with open(f'results/neuralODE_evolve_test_{tau}.txt','a') as f:
                f.writelines(f'{delta_t}, {random_seed}, {MSE}, {RMSE}, {MAE}, {MAPE}, {c1_mae}, {c2_mae}\n')


if __name__ == '__main__':
    
    trace_num = 200
    
    workers = []
    
    tau = 0.8
    n = 8
    
    # # train
    # sample_num = None
    # generate_dataset(trace_num, round(tau/n,3), sample_num, True)
    # generate_dataset(trace_num, round(tau/n,3), sample_num, True, n)
    # for seed in range(1,10+1):
    #     is_print = True if len(workers)==0 else False
    #     workers.append(Process(target=main, args=(trace_num, tau, n, is_print, False, seed), daemon=True))
    #     workers[-1].start()
    # while any([sub.exitcode==None for sub in workers]):
    #     pass
    # workers = []
    
    # test
    sample_num = None
    print('data:')
    for i in tqdm(range(1, 6*n+1+2)):
        delta_t = round(tau/n*i, 3)
        generate_dataset(trace_num, delta_t, sample_num, False)
    print('test:')
    for seed in range(1,10+1):
        # main(trace_num, tau, n, True, True, seed)
        is_print = True if len(workers)==0 else False
        workers.append(Process(target=main, args=(trace_num, tau, n, is_print, True, seed), daemon=True))
        workers[-1].start()
    while any([sub.exitcode==None for sub in workers]):
        pass
    workers = []