# -*- coding: utf-8 -*-
import torch
from torch import nn
from scipy.stats import special_ortho_group


class Koopman_OPT(nn.Module):

    def __init__(self, koopman_dim):
        super(Koopman_OPT, self).__init__()

        self.koopman_dim = koopman_dim
        
        # # (tau,)-->(koopman_dim, koopman_dim)
        # self.V = nn.Sequential(
        #     nn.Linear(1, 64),
        #     nn.Tanh(),
        #     nn.Linear(64, self.koopman_dim**2),
        #     nn.Unflatten(-1, (self.koopman_dim, self.koopman_dim))
        # )
        
        # # (tau,)-->(koopman_dim,)
        # self.Lambda = nn.Sequential(
        #     nn.Linear(1, 64),
        #     nn.Tanh(),
        #     nn.Linear(64, self.koopman_dim)
        # )

        self.register_parameter('V', nn.Parameter(torch.Tensor(koopman_dim, koopman_dim)))
        self.register_parameter('Lambda', nn.Parameter(torch.Tensor(koopman_dim)))

        # init
        torch.nn.init.normal_(self.V, mean=0, std=0.01)
        torch.nn.init.normal_(self.Lambda, mean=0, std=0.01)

    def forward(self, tau):

        # K: (koopman_dim, koopman_dim), K = V * Lambda * V^-1
        # V, Lambda = self.V(tau), self.Lambda(tau)
        # K = torch.mm(torch.mm(V, torch.diag(Lambda)), torch.inverse(V))
        K = torch.mm(torch.mm(torch.inverse(self.V), torch.diag(torch.exp(tau*self.Lambda))), self.V)

        return K
    

class LSTM_OPT(nn.Module):
    
    def __init__(self, in_channels, input_1d_width, hidden_dim, layer_num, tau_s, device):
        super(LSTM_OPT, self).__init__()
        
        self.device = device
        self.tau_s = tau_s
        
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

        # mechanism new
        self.a_head = nn.Linear(hidden_dim, 1)
        self.b_head = nn.Linear(hidden_dim, in_channels*input_1d_width)
        
        # (batchsize, 3)-->(batchsize,1,1,3)
        self.unflatten = nn.Unflatten(-1, (in_channels, input_1d_width))
    
    def forward(self, x):
        
        h0 = torch.zeros(self.layer_num * 1, len(x), self.hidden_dim, dtype=torch.float32, device=self.device)
        c0 = torch.zeros(self.layer_num * 1, len(x), self.hidden_dim, dtype=torch.float32, device=self.device)
        
        x = self.flatten(x)
        _, (h, c)  = self.cell(x, (h0, c0))
        
        # y = self.fc(h[-1])

        a = self.a_head(h[-1]).unsqueeze(-2) # (batch_size, 1, 1)
        b = self.b_head(h[-1]).unsqueeze(-2) # (batch_size, 1, in_channels*input_1d_width)
        y = torch.abs(x) * torch.exp(-(a+self.tau_s)) * b # (batch_size, 1, in_channels*input_1d_width)

        y = self.unflatten(y)
                
        return y


class EVOLVER(nn.Module):
    
    def __init__(self, in_channels, input_1d_width, embed_dim, slow_dim, tau_s, device):
        super(EVOLVER, self).__init__()
        
        # (batchsize,1,1,3)-->(batchsize, embed_dim)
        self.encoder_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels*input_1d_width, 16, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.01),
            nn.Linear(16, embed_dim, bias=True),
            nn.Tanh(),
        )
        
        # (batchsize, embed_dim)-->(batchsize, slow_dim)
        self.encoder_2 = nn.Sequential(
            nn.Linear(embed_dim, 16, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.01),
            nn.Linear(16, slow_dim, bias=True),
            nn.Tanh(),
        )
        
        # (batchsize, slow_dim)-->(batchsize,1,1,3)
        self.decoder = nn.Sequential(
            nn.Linear(slow_dim, 64, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.01),
            nn.Linear(64, embed_dim, bias=True),
            nn.Tanh(),
            nn.Linear(embed_dim, 64, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.01),
            nn.Linear(64, in_channels*input_1d_width, bias=True),
            nn.Tanh(),
            nn.Unflatten(-1, (1, in_channels, input_1d_width))
        )
        
        self.K_opt = Koopman_OPT(slow_dim)
        
        self.lstm = LSTM_OPT(in_channels, input_1d_width, hidden_dim=64, layer_num=2, tau_s=tau_s, device=device)

        # scale inside the model
        self.register_buffer('min', torch.zeros(in_channels, input_1d_width, dtype=torch.float32))
        self.register_buffer('max', torch.ones(in_channels, input_1d_width, dtype=torch.float32))

    def extract(self, x):
        embed = self.encoder_1(x)
        slow_var = self.encoder_2(embed)
        return slow_var, embed

    def recover(self, x):
        x = self.decoder(x)
        return x

    def koopman_evolve(self, x, tau=1., T=1):
        
        K = self.K_opt(tau)
        y = [torch.matmul(K, x.unsqueeze(-1)).squeeze(-1)]
        for _ in range(1, T-1): 
            y.append(torch.matmul(K, y[-1].unsqueeze(-1)).squeeze(-1))
        
        return y[-1], y[:-1]
    
    def lstm_evolve(self, x, T=1):
        
        y = [self.lstm(x)]
        for _ in range(1, T-1): 
            y.append(self.lstm(y[-1]))
            
        return y[-1], y[:-1]
    
    def scale(self, x):
        return (x-self.min) / (self.max-self.min+1e-6)
    
    def descale(self, x):
        return x * (self.max-self.min+1e-6) + self.min