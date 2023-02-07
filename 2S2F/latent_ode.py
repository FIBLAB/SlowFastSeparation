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


class RecognitionRNN(nn.Module):

    def __init__(self, latent_dim=4, obs_dim=2, nhidden=20):
        super(RecognitionRNN, self).__init__()
        self.nhidden = nhidden
        self.flatten = nn.Flatten(start_dim=1)
        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, latent_dim * 2)

    def forward(self, x, h):
        x = self.flatten(x)
        combined = torch.cat((x, h), dim=1)
        h = torch.tanh(self.i2h(combined))
        out = self.h2o(h)
        return out, h

    def initHidden(self, nbatch):
        return torch.zeros(nbatch, self.nhidden)


class Decoder(nn.Module):

    def __init__(self, in_channels, input_1d_width, latent_dim=4, nhidden=32):
        super(Decoder, self).__init__()
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.Tanh()
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, in_channels*input_1d_width)
        self.unflatten = nn.Unflatten(-1, (in_channels, input_1d_width))

    def forward(self, z):
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.unflatten(out)
        return out


class LatentODEfunc(nn.Module):

    def __init__(self, latent_dim=4, nhidden=20):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out


class Latent_ODE(nn.Module):

    def __init__(self, in_channels, input_1d_width, latent_dim=4, nhidden=20, rnn_nhidden=20):
        super(Latent_ODE, self).__init__()

        self.latent_dim = latent_dim

        self.ode = LatentODEfunc(latent_dim, nhidden)
        self.rnn_encoder = RecognitionRNN(latent_dim, in_channels*input_1d_width, rnn_nhidden)
        self.decoder = Decoder(in_channels, input_1d_width, latent_dim, nhidden)

        # scale inside the model
        self.register_buffer('min', torch.zeros(in_channels, input_1d_width, dtype=torch.float32))
        self.register_buffer('max', torch.ones(in_channels, input_1d_width, dtype=torch.float32))

    def obs2latent(self, obs, h):
        return self.rnn_encoder(obs, h)

    def latent2obs(self, latent):
        return self.decoder(latent)

    def initHidden(self, nbatch):
        return self.rnn_encoder.initHidden(nbatch)

    def scale(self, x):
        return (x-self.min) / (self.max-self.min+1e-6)
    
    def descale(self, x):
        return x * (self.max-self.min+1e-6) + self.min


def log_normal_pdf(x, mean, logvar):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl


def train(tau, delta_t, is_print=False, random_seed=729):
        
    # prepare
    device = torch.device('cuda:0')
    data_filepath = 'Data/data/tau_' + str(delta_t)
    log_dir = f'logs/latent_ode/tau_{tau}/seed{random_seed}'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_dir+"/checkpoints/", exist_ok=True)

    # init model
    model = Latent_ODE(in_channels=1, input_1d_width=4, latent_dim=4)
    model.min = torch.from_numpy(np.loadtxt(data_filepath+"/data_min.txt").astype(np.float32)).unsqueeze(0)
    model.max = torch.from_numpy(np.loadtxt(data_filepath+"/data_max.txt").astype(np.float32)).unsqueeze(0)
    model.to(device)
    
    # training params
    lr = 1e-4
    batch_size = 32
    max_epoch = 2000
    weight_decay = 0.001
    MSE_loss = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # dataset
    train_dataset = _2S2FDataset(data_filepath, 'train', neural_ode=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dataset = _2S2FDataset(data_filepath, 'val', neural_ode=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    # training pipeline
    train_loss = []
    for epoch in range(1, max_epoch+1):
        
        losses = []
        
        # train
        model.train()
        dt = 0.01 # same with it in dataset
        sunsample = int(delta_t/dt)
        for t_step, (input, target) in enumerate(train_loader):
            input = model.scale(input[:, ::sunsample].to(device)) # (batchsize,time_length,1,3)
            target = model.scale(target[:,::sunsample].to(device))

            # backward in time to infer q(z_0)
            h = model.initHidden(input.size(0)).to(device)
            # for t in reversed(range(input.size(1))):
            for t in range(input.size(1)):
                obs = input[:, t]
                out, h = model.obs2latent(obs, h)
            qz0_mean, qz0_logvar = out[:, :model.latent_dim], out[:, model.latent_dim:]
            epsilon = torch.randn(qz0_mean.size()).to(device)
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

            # forward in time and solve ode for reconstructions
            t = torch.tensor([t_step+i for i in range(input.size(1))], device=device) * delta_t
            latent_next = odeint(model.ode, z0, t).permute(1, 0, 2)
            output = model.latent2obs(latent_next)

            # compute loss
            noise_std = 3.
            noise_std_ = torch.zeros(output[:,:,0].size()).to(device) + noise_std
            noise_logvar = 2. * torch.log(noise_std_).to(device)
            logpx = log_normal_pdf(input[:,:,0], output[:,:,0], noise_logvar).sum(-1).sum(-1)
            pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(device)
            analytic_kl = normal_kl(qz0_mean, qz0_logvar, pz0_mean, pz0_logvar).sum(-1)
            loss = torch.mean(-logpx + analytic_kl, dim=0)
            # loss = torch.mean(-logpx, dim=0)

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
                input = model.scale(input[:, ::sunsample].to(device)) # (batchsize,1,1,3)
                target = model.scale(target[:, ::sunsample].to(device))
                
                # backward in time to infer q(z_0)
                h = model.initHidden(input.size(0)).to(device)
                for t in range(input.size(1)):
                    obs = input[:, t]
                    out, h = model.obs2latent(obs, h)
                qz0_mean, qz0_logvar = out[:, :model.latent_dim], out[:, model.latent_dim:]
                epsilon = torch.randn(qz0_mean.size()).to(device)
                z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

                # forward in time and solve ode for reconstructions
                t = torch.tensor([t_step+i for i in range(input.size(1))], device=device) * delta_t
                latent_next = odeint(model.ode, z0, t).permute(1, 0, 2)
                output = model.latent2obs(latent_next)

                # compute loss
                noise_std = 3.
                noise_std_ = torch.zeros(output[:,:,0].size()).to(device) + noise_std
                noise_logvar = 2. * torch.log(noise_std_).to(device)
                logpx = log_normal_pdf(input[:,:,0], output[:,:,0], noise_logvar).sum(-1).sum(-1)
                pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(device)
                analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                                        pz0_mean, pz0_logvar).sum(-1)
                loss = torch.mean(-logpx + analytic_kl, dim=0)

                # loss = MSE_loss(output, target)

                # record results
                outputs.append(output[0,:,0].cpu())
                targets.append(target[0,:,0].cpu())
            
            # trans to tensor
            outputs = torch.concat(outputs, axis=0)
            targets = torch.concat(targets, axis=0)
            
            # cal loss
            if is_print: print(f'\rTau[{tau}] | epoch[{epoch}/{max_epoch}] | val-loss={loss.detach().item():.5f}', end='')
                        
            # plot per 5 epoch
            if epoch % 5 == 0:
                
                os.makedirs(log_dir+f"/val/epoch-{epoch}/", exist_ok=True)
                
                # plot total infomation one-step prediction curve
                plt.figure(figsize=(16,4))
                for j, item in enumerate(['c1','c2','c3', 'c4']):
                    ax = plt.subplot(1,4,j+1)
                    ax.set_title(item)
                    plt.plot(targets[:,j], label='true')
                    plt.plot(outputs[:,j], label='predict')
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
    device = torch.device('cuda:0')
    data_filepath = 'Data/data/tau_' + str(delta_t)
    log_dir = f'logs/latent_ode/tau_{tau}/seed{random_seed}'
    os.makedirs(log_dir+f"/test/", exist_ok=True)

    # load model
    batch_size = 32
    model = Latent_ODE(in_channels=1, input_1d_width=4, latent_dim=4)
    ckpt_path = log_dir+f'/checkpoints/epoch-{ckpt_epoch}.ckpt'
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt)
    model = model.to(device)
    
    # dataset
    test_dataset = _2S2FDataset(data_filepath, 'test', neural_ode=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    # testing pipeline        
    with torch.no_grad():
        targets = []
        outputs = []
        
        model.eval()
        dt = 0.01 # same with it in dataset
        sunsample = int(delta_t/dt)
        for t_step, (input, target) in enumerate(test_loader):
            input = model.scale(input[:, ::sunsample].to(device))
            target = model.scale(target[:, ::sunsample].to(device))

            # backward in time to infer q(z_0)
            h = model.initHidden(input.size(0)).to(device)
            for t in reversed(range(input.size(1))):
                obs = input[:, t]
                out, h = model.obs2latent(obs, h)
            qz0_mean, qz0_logvar = out[:, :model.latent_dim], out[:, model.latent_dim:]
            epsilon = torch.randn(qz0_mean.size()).to(device)
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

            # forward in time and solve ode for reconstructions
            t = torch.tensor([t_step+i for i in range(input.size(1))], device=device) * delta_t
            latent_next = odeint(model.ode, z0, t).permute(1, 0, 2)
            output = model.latent2obs(latent_next)

            targets.append(target[0,0])
            outputs.append(output[0,0])
        
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
        plt.plot(true[:,j], label='true')
        plt.plot(pred[:,j], label='predict')
    plt.subplots_adjust(wspace=0.2)
    plt.savefig(log_dir+f"/test/predict_{delta_t}.pdf", dpi=300)
    plt.close()

    c1_evolve_mae = np.mean(np.abs(pred[:,0] - true[:,0]))
    c2_evolve_mae = np.mean(np.abs(pred[:,1] - true[:,1]))
    
    return MSE, RMSE, MAE, MAPE, c1_evolve_mae, c2_evolve_mae


def main(trace_num, tau, n, is_print=False, long_test=False, random_seed=729):
    
    seed_everything(729)
    set_cpu_num(1)
    
    if not long_test:
        # train
        train(tau, round(tau/n,3), is_print=is_print, random_seed=random_seed)
    else:
        # test evolve
        ckpt_epoch = 2000
        for i in tqdm(range(1, 6*n+1+2)):
            delta_t = round(tau/n*i, 3)
            generate_dataset(trace_num, delta_t, None, True, neural_ode=True)
            MSE, RMSE, MAE, MAPE, c1_mae, c2_mae = test_evolve(tau, ckpt_epoch, delta_t, i, is_print, random_seed)
            with open(f'neuralODE_evolve_test_{tau}.txt','a') as f:
                f.writelines(f'{delta_t}, {random_seed}, {MSE}, {RMSE}, {MAE}, {MAPE}, {c1_mae}, {c2_mae}\n')


if __name__ == '__main__':
    
    trace_num = 200
    
    workers = []
    
    tau = 0.8
    n = 8
    
    # train
    sample_num = None
    generate_dataset(trace_num, round(tau/n, 3), sample_num, True, neural_ode=True)
    for seed in range(1,10+1):
        is_print = True if len(workers)==0 else False
        workers.append(Process(target=main, args=(trace_num, tau, n, is_print, False, seed), daemon=True))
        workers[-1].start()
    while any([sub.exitcode==None for sub in workers]):
        pass
    workers = []
    
    # test
    # for seed in range(1,10+1):
    #     main(trace_num, tau, n, True, True, seed)
    #     is_print = True if len(workers)==0 else False
    #     workers.append(Process(target=main, args=(trace_num, tau, n, is_print, True, seed), daemon=True))
    #     workers[-1].start()
    # while any([sub.exitcode==None for sub in workers]):
    #     pass
    # workers = []