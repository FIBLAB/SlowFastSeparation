# -*- coding: utf-8 -*-
import os
import time
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Process
from pytorch_lightning import seed_everything
import warnings;warnings.simplefilter('ignore')

import models
from Data.dataset import _1S2FDataset
from Data.generator import generate_dataset, generate_original_data
from util import set_cpu_num
from util.plot import plot_epoch_test_log, plot_slow_ae_loss, plot_id_per_tau
from util.intrinsic_dimension import eval_id_embedding


def train_time_lagged(tau, is_print=False, random_seed=729):
    
    # prepare
    device = torch.device('cpu')
    data_filepath = 'Data/data/tau_' + str(tau)
    log_dir = 'logs/time-lagged/tau_' + str(tau) + f'/seed{random_seed}'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_dir+"/checkpoints/", exist_ok=True)
    
    # init model
    model = models.TIME_LAGGED_AE(in_channels=1, input_1d_width=3, embed_dim=64)
    model.apply(models.weights_normal_init)
    model.min = torch.from_numpy(np.loadtxt(data_filepath+"/data_min.txt").astype(np.float32)).unsqueeze(0)
    model.max = torch.from_numpy(np.loadtxt(data_filepath+"/data_max.txt").astype(np.float32)).unsqueeze(0)
    model.to(device)
    
    # training params
    lr = 0.001
    batch_size = 32
    max_epoch = 30
    weight_decay = 0.001
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_func = nn.MSELoss()

    # dataset
    train_dataset = _1S2FDataset(data_filepath, 'train')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataset = _1S2FDataset(data_filepath, 'val')
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    # training pipeline
    losses = []
    loss_curve = []
    for epoch in range(1, max_epoch+1):
        
        # train
        model.train()
        for input, target in train_loader:
            input = model.scale(input.to(device)) # (batchsize,1,1,3)
            target = model.scale(target.to(device))
            
            output, _ = model.forward(input)
            
            loss = loss_func(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.detach().item())
            
        loss_curve.append(np.mean(losses))
        
        # validate
        with torch.no_grad():
            targets = []
            outputs = []
            
            model.eval()
            for input, target in val_loader:
                input = model.scale(input.to(device))
                target = model.scale(target.to(device))
            
                output, _ = model.forward(input)
                outputs.append(output.cpu())
                targets.append(target.cpu())
                
            targets = torch.concat(targets, axis=0)
            outputs = torch.concat(outputs, axis=0)
            mse = loss_func(outputs, targets)
            if is_print: print(f'\rTau[{tau}] | epoch[{epoch}/{max_epoch}] val-MSE={mse:.5f}', end='')
        
        # save each epoch model
        model.train()
        torch.save(
            {'model': model.state_dict(),
             'encoder': model.encoder.state_dict(),}, 
            log_dir+f"/checkpoints/epoch-{epoch}.ckpt")
        
    # plot loss curve
    plt.figure()
    plt.plot(loss_curve)
    plt.xlabel('epoch')
    plt.title('Train MSELoss Curve')
    plt.savefig(log_dir+'/loss_curve.pdf', dpi=300)
    np.save(log_dir+'/loss_curve.npy', loss_curve)
    
    if is_print: print()
    

def test_and_save_embeddings_of_time_lagged(tau, checkpoint_filepath=None, is_print=False, random_seed=729):
    
    # prepare
    device = torch.device('cpu')
    data_filepath = 'Data/data/tau_' + str(tau)
    log_dir = 'logs/time-lagged/tau_' + str(tau) + f'/seed{random_seed}'
    os.makedirs(log_dir+'/test', exist_ok=True)
    
    # testing params
    batch_size = 32
    max_epoch = 30
    loss_func = nn.MSELoss()
    
    # init model
    model = models.TIME_LAGGED_AE(in_channels=1, input_1d_width=3, embed_dim=64)
    if checkpoint_filepath is None: # not trained
        model.apply(models.weights_normal_init)
        model.min = torch.from_numpy(np.loadtxt(data_filepath+"/data_min.txt").astype(np.float32)).unsqueeze(0)
        model.max = torch.from_numpy(np.loadtxt(data_filepath+"/data_max.txt").astype(np.float32)).unsqueeze(0)

    # dataset
    train_dataset = _1S2FDataset(data_filepath, 'train')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_dataset = _1S2FDataset(data_filepath, 'test')
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # testing pipeline
    fp = open('logs/time-lagged/tau_' + str(tau) + '/test_log.txt', 'a')
    for ep in range(max_epoch):
        
        # load weight file
        epoch = ep
        if checkpoint_filepath is not None:
            epoch = ep + 1
            ckpt_path = checkpoint_filepath + f"/checkpoints/" + f'epoch-{epoch}.ckpt'
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt['model'])
        model = model.to(device)
        model.eval()
        
        all_embeddings = []
        test_outputs = np.array([])
        test_targets = np.array([])
        train_outputs = np.array([])
        train_targets = np.array([])
        var_log_dir = log_dir + f'/test/epoch-{epoch}'
        os.makedirs(var_log_dir, exist_ok=True)
        
        # testing
        with torch.no_grad():
            
            # train-dataset
            for batch_idx, (input, target) in enumerate(train_loader):
                input = model.scale(input.to(device)) # (batchsize,1,1,3)
                target = model.scale(target.to(device))
                
                output, _ = model.forward(input)
                
                train_outputs = output.cpu() if not len(train_outputs) else torch.concat((train_outputs, output.cpu()), axis=0)
                train_targets = target.cpu() if not len(train_targets) else torch.concat((train_targets, target.cpu()), axis=0)
                                
                if batch_idx >= len(test_loader): break

            # test-dataset
            for input, target in test_loader:
                input = model.scale(input.to(device)) # (batchsize,1,1,3)
                target = model.scale(target.to(device))
                
                output, embedding = model.forward(input)
                # save the embedding vectors
                # TODO: 这里的代码好奇怪？
                for idx in range(input.shape[0]):
                    embedding_tmp = embedding[idx].view(1, -1)[0]
                    embedding_tmp = embedding_tmp.cpu().numpy()
                    all_embeddings.append(embedding_tmp)

                test_outputs = output.cpu() if not len(test_outputs) else torch.concat((test_outputs, output.cpu()), axis=0)
                test_targets = target.cpu() if not len(test_targets) else torch.concat((test_targets, target.cpu()), axis=0)
                                
            # test mse
            mse_x = loss_func(test_outputs[:,0,0,0], test_targets[:,0,0,0])
            mse_y = loss_func(test_outputs[:,0,0,1], test_targets[:,0,0,1])
            mse_z = loss_func(test_outputs[:,0,0,2], test_targets[:,0,0,2])
        
        # plot
        test_plot, train_plot = [[], [], []], [[], [], []]
        for i in range(len(test_outputs)):
            for j in range(len(test_plot)):
                test_plot[j].append([test_outputs[i,0,0,j], test_targets[i,0,0,j]])
                train_plot[j].append([train_outputs[i,0,0,j], train_targets[i,0,0,j]])
        plt.figure(figsize=(16,9))
        for i, item in enumerate(['test', 'train']):
            plot_data = test_plot if i == 0 else train_plot
            for j in range(len(test_plot)):
                ax = plt.subplot(2,3,j+1+3*i)
                ax.set_title(item+'_'+['X','Y','Z'][j])
                plt.plot(np.array(plot_data[j])[:,1], label='true')
                plt.plot(np.array(plot_data[j])[:,0], label='predict')
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15, wspace=0.2, hspace=0.35)
        plt.savefig(var_log_dir+"/result.pdf", dpi=300)
        plt.close()

        # save embedding
        np.save(var_log_dir+'/embedding.npy', all_embeddings)
        
        # calculae ID
        def cal_id_embedding(tau, epoch, method='MLE', is_print=False):
            var_log_dir = log_dir + f'/test/epoch-{epoch}'
            eval_id_embedding(var_log_dir, method=method, is_print=is_print, max_point=100)
            dims = np.load(var_log_dir+f'/id_{method}.npy')
            return np.mean(dims)
        LB_id = cal_id_embedding(tau, epoch, 'MLE')
        # MiND_id = cal_id_embedding(tau, epoch, 'MiND_ML')
        # MADA_id = cal_id_embedding(tau, epoch, 'MADA')
        # PCA_id = cal_id_embedding(tau, epoch, 'PCA')

        # logging
        # fp.write(f"{tau},0,{mse_x},{mse_y},{mse_z},{epoch},{LB_id},{MiND_id},{MADA_id},{PCA_id}\n")
        fp.write(f"{tau},0,{mse_x},{mse_y},{mse_z},{epoch},{LB_id},{0},{0},{0}\n")
        fp.flush()

        # if is_print: print(f'\rTau[{tau}] | Test epoch[{epoch}/{max_epoch}] | MLE={LB_id:.1f}, MinD={MiND_id:.1f}, MADA={MADA_id:.1f}, PCA={PCA_id:.1f}   ', end='')
        if is_print: print(f'\rTau[{tau}] | Test epoch[{epoch}/{max_epoch}] | MLE={LB_id:.1f}, MinD={0:.1f}, MADA={0:.1f}, PCA={0:.1f}   ', end='')
        
        if checkpoint_filepath is None: break
        
    fp.close()
    if is_print: print()
    
    
def train_slow_extract_and_evolve(tau, pretrain_epoch, slow_id, delta_t, n, is_print=False, random_seed=729):
        
    # prepare
    device = torch.device('cuda:0')
    data_filepath = 'Data/data/tau_' + str(delta_t)
    log_dir = f'logs/slow_extract_and_evolve/tau_{tau}/pretrain_epoch{pretrain_epoch}/id{slow_id}/seed{random_seed}'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_dir+"/checkpoints/", exist_ok=True)

    # init model
    model = models.EVOLVER(in_channels=1, input_1d_width=3, embed_dim=64, slow_dim=slow_id, tau_s=3.0, device=device)
    model.apply(models.weights_normal_init)
    model.min = torch.from_numpy(np.loadtxt(data_filepath+"/data_min.txt").astype(np.float32)).unsqueeze(0)
    model.max = torch.from_numpy(np.loadtxt(data_filepath+"/data_max.txt").astype(np.float32)).unsqueeze(0)
    
    # load pretrained time-lagged AE
    ckpt_path = f'logs/time-lagged/tau_{tau}/seed1/checkpoints/epoch-{pretrain_epoch}.ckpt'
    ckpt = torch.load(ckpt_path)
    model.encoder_1.load_state_dict(ckpt['encoder'])
    model = model.to(device)
    
    # training params
    lr = 0.001
    batch_size = 32
    max_epoch = 50
    weight_decay = 0.001
    L1_loss = nn.L1Loss()
    MSE_loss = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        [{'params': model.encoder_2.parameters()},
         {'params': model.decoder.parameters()}, 
         {'params': model.K_opt.parameters()},
         {'params': model.lstm.parameters()}],
        lr=lr, weight_decay=weight_decay) # not involve encoder_1 (freezen)
    
    # dataset
    train_dataset = _1S2FDataset(data_filepath, 'train', length=n)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dataset = _1S2FDataset(data_filepath, 'val', length=n)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    # training pipeline
    train_loss = []
    val_loss = []
    lambda_curve = [[] for _ in range(slow_id)]
    for epoch in range(1, max_epoch+1):
        
        losses = [[],[],[],[]]
        
        # train
        model.train()
        [lambda_curve[i].append(model.K_opt.Lambda[i].detach().cpu()) for i in range(slow_id)]
        for input, _, internl_units in train_loader:
            
            input = model.scale(input.to(device)) # (batchsize,1,1,3)
            
            ###############
            # slow extract
            ###############
            slow_var, embed = model.extract(input)
            slow_info = model.recover(slow_var)
            _, embed_from_info = model.extract(slow_info)
            
            adiabatic_loss = L1_loss(embed, embed_from_info)
            slow_reconstruct_loss = MSE_loss(slow_info, input)
            
            ################
            # n-step evolve
            ################
            fast_info = input - slow_info.detach()
            koopman_loss, evolve_loss = 0, 0
            for i in range(1, len(internl_units)):
                
                unit = model.scale(internl_units[i].to(device)) # t+i
                
                # extract to slow variables
                unit_slow_var, _ = model.extract(unit)

                # slow evolve
                t = delta_t * i # delta_t between 0 and t+i
                unit_slow_var_pred, _ = model.koopman_evolve(slow_var, tau=torch.tensor([t], device=device), T=1) # 0 ——> t+i
                unit_slow_info_pred = model.recover(unit_slow_var_pred)
                
                # fast evolve
                unit_fast_info_pred, _ = model.lstm_evolve(fast_info, T=i) # t ——> t+i
                
                # total evolve
                unit_info_pred = unit_slow_info_pred + unit_fast_info_pred
                
                # evolve loss
                koopman_loss += MSE_loss(unit_slow_var_pred, unit_slow_var)
                evolve_loss += MSE_loss(unit_info_pred, unit)
            
            ###########
            # optimize
            ###########
            all_loss = (slow_reconstruct_loss + 0.05*adiabatic_loss) + (0.1*koopman_loss + evolve_loss) / n
            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()
            
            # record loss
            losses[0].append(adiabatic_loss.detach().item())
            losses[1].append(slow_reconstruct_loss.detach().item())
            losses[2].append(koopman_loss.detach().item())
            losses[3].append(evolve_loss.detach().item())
            
        train_loss.append([np.mean(losses[0]), np.mean(losses[1]), np.mean(losses[2]), np.mean(losses[3])])
        
        # validate
        with torch.no_grad():
            inputs = []
            slow_vars = []
            targets = []
            slow_infos = []
            slow_infos_next = []
            fast_infos = []
            fast_infos_next = []
            total_infos_next = []
            embeds = []
            embed_from_infos = []
            
            model.eval()
            for input, target, _ in val_loader:
                
                input = model.scale(input.to(device)) # (batchsize,1,1,3)
                target = model.scale(target.to(device))
                
                # slow extract
                slow_var, embed = model.extract(input)
                slow_info = model.recover(slow_var)
                _, embed_from_info = model.extract(slow_info)
                
                # slow evolve
                slow_var_next, _ = model.koopman_evolve(slow_var, tau=torch.tensor([tau-delta_t], device=device), T=1)
                slow_info_next = model.recover(slow_var_next)
                
                # fast evolve
                fast_info = input - slow_info
                fast_info_next, _ = model.lstm_evolve(fast_info, T=n)
                
                # total evolve
                total_info_next = slow_info_next + fast_info_next

                # record results
                inputs.append(input.cpu())
                slow_vars.append(slow_var.cpu())
                targets.append(target.cpu())
                slow_infos.append(slow_info.cpu())
                slow_infos_next.append(slow_info_next.cpu())
                fast_infos.append(fast_info.cpu())
                fast_infos_next.append(fast_info_next.cpu())
                total_infos_next.append(total_info_next.cpu())
                embeds.append(embed.cpu())
                embed_from_infos.append(embed_from_info.cpu())
            
            # trans to tensor
            inputs = torch.concat(inputs, axis=0)
            slow_vars = torch.concat(slow_vars, axis=0)
            targets = torch.concat(targets, axis=0)
            slow_infos = torch.concat(slow_infos, axis=0)
            slow_infos_next = torch.concat(slow_infos_next, axis=0)
            fast_infos = torch.concat(fast_infos, axis=0)
            fast_infos_next = torch.concat(fast_infos_next, axis=0)
            total_infos_next = torch.concat(total_infos_next, axis=0)
            embeds = torch.concat(embeds, axis=0)
            embed_from_infos = torch.concat(embed_from_infos, axis=0)
            
            # cal loss
            adiabatic_loss = L1_loss(embeds, embed_from_infos)
            slow_reconstruct_loss = MSE_loss(slow_infos, inputs)
            evolve_loss = MSE_loss(total_infos_next, targets)
            all_loss = 0.5*slow_reconstruct_loss + 0.5*evolve_loss + 0.05*adiabatic_loss
            if is_print: print(f'\rTau[{tau}] | epoch[{epoch}/{max_epoch}] | val: adiab_loss={adiabatic_loss:.5f}, recons_loss={slow_reconstruct_loss:.5f}, evol_loss={evolve_loss:.5f}', end='')
            
            val_loss.append(all_loss.detach().item())
            
            # plot per 5 epoch
            if epoch % 5 == 0:
                
                os.makedirs(log_dir+f"/val/epoch-{epoch}/", exist_ok=True)
                
                # TODO: 把类似的plot写进for循环，压缩行数
                # plot slow variable vs input
                plt.figure(figsize=(12,5+2*(slow_id-1)))
                plt.title('Val Reconstruction Curve')
                for id_var in range(slow_id):
                    for index, item in enumerate(['X', 'Y', 'Z']):
                        plt.subplot(slow_id, 3, index+1+3*(id_var))
                        plt.scatter(inputs[:,0,0,index], slow_vars[:, id_var], s=5)
                        plt.xlabel(item)
                        plt.ylabel(f'U{id_var+1}')
                plt.subplots_adjust(wspace=0.35, hspace=0.35)
                plt.savefig(log_dir+f"/val/epoch-{epoch}/slow_vs_input.pdf", dpi=300)
                plt.close()
                
                # plot slow variable
                plt.figure(figsize=(12,5+2*(slow_id-1)))
                plt.title('Slow variable Curve')
                for id_var in range(slow_id):
                    plt.subplot(slow_id, 1, 1+id_var)
                    plt.plot(inputs[:,0,0,0], label='X')
                    plt.plot(slow_vars[:, id_var], label=f'U{id_var+1}')
                    plt.xlabel(item)
                plt.legend()
                plt.subplots_adjust(wspace=0.35, hspace=0.35)
                plt.savefig(log_dir+f"/val/epoch-{epoch}/slow_variable.pdf", dpi=300)
                plt.close()
                
                # plot slow infomation reconstruction curve
                plt.figure(figsize=(16,5))
                for j, item in enumerate(['X','Y','Z']):
                    ax = plt.subplot(1,3,j+1)
                    ax.set_title(item)
                    plt.plot(inputs[:,0,0,j], label='all_info')
                    plt.plot(slow_infos[:,0,0,j], label='slow_info')
                plt.subplots_adjust(wspace=0.2)
                plt.savefig(log_dir+f"/val/epoch-{epoch}/slow_info.pdf", dpi=300)
                plt.close()
                
                # plot fast infomation curve (== origin_data - slow_info_recons)
                plt.figure(figsize=(16,5))
                for j, item in enumerate(['X','Y','Z']):
                    ax = plt.subplot(1,3,j+1)
                    ax.set_title(item)
                    plt.plot(inputs[:,0,0,j], label='all_info')
                    plt.plot(fast_infos[:,0,0,j], label='fast_info')
                plt.subplots_adjust(wspace=0.2)
                plt.savefig(log_dir+f"/val/epoch-{epoch}/fast_info.pdf", dpi=300)
                plt.close()
                
                # plot slow infomation one-step prediction curve
                plt.figure(figsize=(16,5))
                for j, item in enumerate(['X','Y','Z']):
                    ax = plt.subplot(1,3,j+1)
                    ax.set_title(item)
                    plt.plot(targets[:,0,0,j], label='all_true')
                    plt.plot(slow_infos_next[:,0,0,j], label='slow_predict')
                plt.subplots_adjust(wspace=0.2)
                plt.savefig(log_dir+f"/val/epoch-{epoch}/slow_predict.pdf", dpi=300)
                plt.close()
                
                # plot fast infomation one-step prediction curve
                plt.figure(figsize=(16,5))
                for j, item in enumerate(['X','Y','Z']):
                    ax = plt.subplot(1,3,j+1)
                    ax.set_title(item)
                    plt.plot(targets[:,0,0,j], label='all_true')
                    plt.plot(fast_infos_next[:,0,0,j], label='fast_predict')
                plt.subplots_adjust(wspace=0.2)
                plt.savefig(log_dir+f"/val/epoch-{epoch}/fast_predict.pdf", dpi=300)
                plt.close()
                
                # plot total infomation one-step prediction curve
                plt.figure(figsize=(16,5))
                for j, item in enumerate(['X','Y','Z']):
                    ax = plt.subplot(1,3,j+1)
                    ax.set_title(item)
                    plt.plot(targets[:,0,0,j], label='all_true')
                    plt.plot(total_infos_next[:,0,0,j], label='all_predict')
                plt.subplots_adjust(wspace=0.2)
                plt.savefig(log_dir+f"/val/epoch-{epoch}/all_predict.pdf", dpi=300)
                plt.close()
                
                # save model
                torch.save(model.state_dict(), log_dir+f"/checkpoints/epoch-{epoch}.ckpt")
    
    # plot loss curve
    train_loss = np.array(train_loss)
    plt.figure()
    for i, item in enumerate(['adiabatic','slow_reconstruct','koopman_evolve','total_evolve']):
        plt.plot(train_loss[:, i], label=item)
    plt.xlabel('epoch')
    plt.legend()
    plt.title('Training Loss Curve')
    plt.savefig(log_dir+'/train_loss_curve.pdf', dpi=300)
    np.save(log_dir+'/val_loss_curve.npy', val_loss)

    # plot Koopman Lambda curve
    plt.figure()
    for i in range(slow_id):
        plt.plot(lambda_curve[i], label=f'lambda[{i}]')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(log_dir+'/K_lambda_curve.pdf', dpi=300)


def test_evolve(tau, pretrain_epoch, ckpt_epoch, slow_id, delta_t, n, is_print=False, random_seed=729):
        
    # prepare
    device = torch.device('cuda:0')
    data_filepath = 'Data/data/tau_' + str(delta_t)
    log_dir = f'logs/slow_extract_and_evolve/tau_{tau}/pretrain_epoch{pretrain_epoch}/id{slow_id}/seed{random_seed}'

    # load model
    batch_size = 32
    model = models.EVOLVER(in_channels=1, input_1d_width=3, embed_dim=64, slow_dim=slow_id, tau_s=3.0, device=device)
    ckpt_path = log_dir+f'/checkpoints/epoch-{ckpt_epoch}.ckpt'
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt)
    model = model.to(device)
    
    # dataset
    test_dataset = _1S2FDataset(data_filepath, 'test')
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    # testing pipeline        
    with torch.no_grad():
        inputs = []
        targets = []
        slow_vars = []
        slow_obses = []
        slow_infos_next = []
        fast_infos_next = []
        total_infos_next = []
        
        model.eval()
        for input, target in test_loader:
            input = model.scale(input.to(device))
            target = model.scale(target.to(device))
        
            # slow extract
            slow_var, _ = model.extract(input)
            slow_info = model.recover(slow_var)
            
            # slow evolve
            slow_var_next, _ = model.koopman_evolve(slow_var, tau=torch.tensor([delta_t], device=device), T=1)
            slow_info_next = model.recover(slow_var_next)
            
            # fast evolve
            fast_info = input - slow_info
            fast_info_next, _ = model.lstm_evolve(fast_info, T=n)
            
            # total evolve
            total_info_next = slow_info_next + fast_info_next

            inputs.append(input)
            targets.append(target)
            slow_vars.append(slow_var)
            slow_obses.append(slow_info)
            slow_infos_next.append(slow_info_next)
            fast_infos_next.append(fast_info_next)
            total_infos_next.append(total_info_next)
        
        inputs = model.descale(torch.concat(inputs, axis=0)).cpu()
        slow_obses = model.descale(torch.concat(slow_obses, axis=0)).cpu()
        slow_vars = torch.concat(slow_vars, axis=0).cpu()
        slow_infos_next = model.descale(torch.concat(slow_infos_next, axis=0)).cpu()
        fast_infos_next = model.descale(torch.concat(fast_infos_next, axis=0)).cpu()
        
        targets = torch.concat(targets, axis=0)
        total_infos_next = torch.concat(total_infos_next, axis=0)
    
    # metrics
    pred = total_infos_next.detach().cpu().numpy()
    true = targets.detach().cpu().numpy()
    MAPE = np.mean(np.abs((pred - true) / true))
    targets = model.descale(targets)
    total_infos_next = model.descale(total_infos_next)
    pred = total_infos_next.detach().cpu().numpy()
    true = targets.detach().cpu().numpy()
    MSE = np.mean((pred - true) ** 2)
    RMSE = np.sqrt(MSE)
    MAE = np.mean(np.abs(pred - true))
        
    os.makedirs(log_dir+f"/test/{delta_t}/", exist_ok=True)

    import scienceplots
    plt.style.use(['science'])

    # # plot slow extract from original data
    # start = 4300 - 1
    # end = 10300 - 1
    # sample = 20
    # plt.rcParams.update({'font.size':16})
    # plt.figure(figsize=(16,5))
    # for j, item in enumerate([r'$X$', r'$Y$', r'$Z$']):
    #     ax = plt.subplot(1,3,j+1)
    #     t = torch.range(0,end-start-1)*delta_t
    #     ax.plot(t[::sample], inputs[start:end:sample,0,0,j], label=r'$X$')
    #     ax.plot(t[::sample], slow_obses[start:end:sample,0,0,j], marker="^", markersize=4, label=r'$X_s$')
    #     ax.legend()
    #     plt.xticks(fontsize=16)
    #     plt.yticks(fontsize=16)
    #     plt.xlabel('t / s', fontsize=18)
    #     plt.ylabel(item, fontsize=18)
    # plt.subplots_adjust(wspace=0.35)
    # plt.savefig(log_dir+f"/test/{delta_t}/slow_extract.pdf", dpi=300)
    # plt.savefig(log_dir+f"/test/{delta_t}/slow_extract.jpg", dpi=300)
    # plt.close()

    # plot slow variable vs input
    sample = 4
    plt.figure(figsize=(16,4))
    for id_var in range(slow_id):
        for index, item in enumerate([r'$X$', r'$Y$', r'$Z$']):
            ax = plt.subplot(slow_id, 3, index+1+4*(id_var))
            ax.scatter(inputs[::sample,0,0,index], slow_vars[::sample, id_var], s=2)
            plt.xlabel(item, fontsize=18)
            plt.ylabel(rf'$u$', fontsize=18)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            if index==2:
                ax.xaxis.set_major_locator(plt.MultipleLocator(5000))
    plt.subplots_adjust(wspace=0.55, hspace=0.35, bottom=0.15)
    plt.savefig(log_dir+f"/test/{delta_t}/slow_vs_input.pdf", dpi=150)
    plt.savefig(log_dir+f"/test/{delta_t}/slow_vs_input.jpg", dpi=150)
    plt.close()
    
    # # plot slow infomation prediction curve
    # plt.figure(figsize=(16,5))
    # for j, item in enumerate(['X','Y','Z']):
    #     ax = plt.subplot(1,3,j+1)
    #     ax.set_title(item)
    #     plt.plot(true[:,0,0,j], label='true')
    #     plt.plot(slow_infos_next[:,0,0,j], label='predict')
    # plt.subplots_adjust(wspace=0.2)
    # plt.savefig(log_dir+f"/test/{delta_t}/slow_pred.pdf", dpi=300)
    # plt.close()
    
    # # plot fast infomation prediction curve
    # plt.figure(figsize=(16,5))
    # for j, item in enumerate(['X','Y','Z']):
    #     ax = plt.subplot(1,3,j+1)
    #     ax.set_title(item)
    #     plt.plot(true[:,0,0,j], label='true')
    #     plt.plot(fast_infos_next[:,0,0,j], label='predict')
    # plt.subplots_adjust(wspace=0.2)
    # plt.savefig(log_dir+f"/test/{delta_t}/fast_pred.pdf", dpi=300)
    # plt.close()
    
    # # plot total infomation prediction curve
    # plt.figure(figsize=(16,5))
    # for j, item in enumerate(['X','Y','Z']):
    #     ax = plt.subplot(1,3,j+1)
    #     ax.set_title(item)
    #     plt.plot(true[:,0,0,j], label='true')
    #     plt.plot(pred[:,0,0,j], label='predict')
    # plt.subplots_adjust(wspace=0.2)
    # plt.savefig(log_dir+f"/test/{delta_t}/total.pdf", dpi=300)
    # plt.close()
    
    return MSE, RMSE, MAE, MAPE
        
    
def worker_1(tau, trace_num=256+32+32, random_seed=729, cpu_num=1, is_print=False):
    
    time.sleep(0.1)
    seed_everything(random_seed)
    set_cpu_num(cpu_num)
    
    # train
    train_time_lagged(tau, is_print, random_seed)
    # test and calculating ID
    test_and_save_embeddings_of_time_lagged(tau, None, is_print, random_seed)
    test_and_save_embeddings_of_time_lagged(tau, f"logs/time-lagged/tau_{tau}/seed{random_seed}", is_print, random_seed)


def worker_2(tau, pretrain_epoch, slow_id, n, random_seed=729, cpu_num=1, is_print=False, id_list=[1,2,3,4], long_test=False):
    
    time.sleep(0.1)
    seed_everything(random_seed)
    set_cpu_num(cpu_num)

    ckpt_epoch = 50

    if not long_test:
        # train
        train_slow_extract_and_evolve(tau, pretrain_epoch, slow_id, round(tau/n,3), n, is_print=is_print, random_seed=random_seed)
        # plot mse curve of each id
        try: plot_slow_ae_loss(tau, pretrain_epoch, delta_t, id_list) 
        except: pass
    else:
        # test evolve
        for i in tqdm(range(1, 5*n+1)):
            delta_t = round(tau/n*i, 3)
            MSE, RMSE, MAE, MAPE = test_evolve(tau, pretrain_epoch, ckpt_epoch, slow_id, delta_t, i, is_print, random_seed)
            with open(f'results/evolve_test_{tau}.txt','a') as f:
                f.writelines(f'{delta_t}, {random_seed}, {MSE}, {RMSE}, {MAE}, {MAPE}\n')
    
    
def data_generator_pipeline(trace_num=256+32+32, total_t=9):
    
    # generate original data
    generate_original_data(trace_num=trace_num, total_t=total_t)
    
    
def id_esitimate_pipeline(cpu_num=1, trace_num=256+32+32):
    
    tau_list = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]
    workers = []
    
    # id esitimate sub-process
    for tau in tau_list:

        generate_dataset(trace_num, tau, None, is_print=True)

        for seed in range(1,5+1):
            is_print = True if len(workers)==0 else False
            workers.append(Process(target=worker_1, args=(tau, trace_num, seed, cpu_num, is_print), daemon=True))
            workers[-1].start()
    while any([sub.exitcode==None for sub in workers]):
        pass

    [plot_epoch_test_log(tau, max_epoch=30+1) for tau in tau_list]
    plot_id_per_tau(tau_list, np.arange(15,15+1))
    
    print('ID Esitimate Over!')


def slow_evolve_pipeline(trace_num=256+32+32, n=10, cpu_num=1, long_test=False):
    
    tau_list = [3.0]
    id_list = [1]
    workers = []
    
    # generate dataset sub-process
    sample_num = None
    for tau in tau_list:
        if not long_test:
            # dataset for training
            workers.append(Process(target=generate_dataset, args=(trace_num, round(tau/n,3), sample_num, True, n), daemon=True))
            workers[-1].start()
        
        # dataset for testing
        for i in range(1, 5*n+1):
            print(f'processing testing dataset [{i}/{5*n}]')
            delta_t = round(tau/n*i, 3)
            generate_dataset(trace_num, delta_t, sample_num, True)
            # workers.append(Process(target=generate_dataset, args=(trace_num, delta_t, sample_num, False), daemon=True))
            # workers[-1].start()
    while any([sub.exitcode==None for sub in workers]):
        pass
    workers = []
    
    # slow evolve sub-process
    for tau in tau_list:
        for pretrain_epoch in [15]:
            for slow_id in id_list:
                for random_seed in range(1,10+1):
                    is_print = True if len(workers)==0 else False
                    workers.append(Process(target=worker_2, args=(tau, pretrain_epoch, slow_id, n, random_seed, cpu_num, is_print, id_list, long_test), daemon=True))
                    workers[-1].start()
    while any([sub.exitcode==None for sub in workers]):
        pass
    
    print('Slow-Infomation Evolve Over!')
    

if __name__ == '__main__':
    
    trace_num = 100
    
    data_generator_pipeline(trace_num, total_t=15.1)
    id_esitimate_pipeline(trace_num=trace_num)
    slow_evolve_pipeline(trace_num=trace_num, n=10, long_test=False)
    slow_evolve_pipeline(trace_num=trace_num, n=10, long_test=True)
    
    torch.cuda.empty_cache()