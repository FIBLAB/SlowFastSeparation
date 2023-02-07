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
from Data.dataset import _2S2FDataset
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
    model = models.TIME_LAGGED_AE(in_channels=1, input_1d_width=4, embed_dim=64)
    model.apply(models.weights_normal_init)
    model.min = torch.from_numpy(np.loadtxt(data_filepath+"/data_min.txt").astype(np.float32)).unsqueeze(0)
    model.max = torch.from_numpy(np.loadtxt(data_filepath+"/data_max.txt").astype(np.float32)).unsqueeze(0)
    model.to(device)
    
    # training params
    lr = 0.001
    batch_size = 128
    max_epoch = 100
    weight_decay = 0.001
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_func = nn.MSELoss()

    # dataset
    train_dataset = _2S2FDataset(data_filepath, 'train')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dataset = _2S2FDataset(data_filepath, 'val')
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    # training pipeline
    losses = []
    loss_curve = []
    for epoch in range(1, max_epoch+1):
        
        # train
        model.train()
        for input, target in train_loader:
            input = model.scale(input.to(device)) # (batchsize,1,1,4)
            target = model.scale(target.to(device))
            
            output, _ = model.forward(input)
            
            loss = loss_func(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.detach().item())
            
        loss_curve.append(np.mean(losses))
        
        # # validate
        # with torch.no_grad():
        #     targets = []
        #     outputs = []
            
        #     model.eval()
        #     for input, target in val_loader:
        #         input = model.scale(input.to(device))
        #         target = model.scale(target.to(device))
            
        #         output, _ = model.forward(input)
        #         outputs.append(output.cpu())
        #         targets.append(target.cpu())
                
        #     targets = torch.concat(targets, axis=0)
        #     outputs = torch.concat(outputs, axis=0)
        #     mse = loss_func(outputs, targets)
        #     if is_print: print(f'\rTau[{tau}] | epoch[{epoch}/{max_epoch}] val-MSE={mse:.5f}', end='')
        if is_print: print(f'\rTau[{tau}] | epoch[{epoch}/{max_epoch}] train-MSE={loss:.5f}', end='')
        
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
    batch_size = 128
    max_epoch = 100
    loss_func = nn.MSELoss()
    
    # init model
    model = models.TIME_LAGGED_AE(in_channels=1, input_1d_width=4, embed_dim=64)
    if checkpoint_filepath is None: # not trained
        model.apply(models.weights_normal_init)
        model.min = torch.from_numpy(np.loadtxt(data_filepath+"/data_min.txt").astype(np.float32)).unsqueeze(0)
        model.max = torch.from_numpy(np.loadtxt(data_filepath+"/data_max.txt").astype(np.float32)).unsqueeze(0)

    # dataset
    train_dataset = _2S2FDataset(data_filepath, 'train')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_dataset = _2S2FDataset(data_filepath, 'test')
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # testing pipeline
    fp = open('logs/time-lagged/tau_' + str(tau) + '/test_log.txt', 'a')
    for ep in range(1,max_epoch):
        
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
            
            # # train-dataset
            # for batch_idx, (input, target) in enumerate(train_loader):
            #     input = model.scale(input.to(device)) # (batchsize,1,1,4)
            #     target = model.scale(target.to(device))
                
            #     output, _ = model.forward(input)
                
            #     train_outputs = output.cpu() if not len(train_outputs) else torch.concat((train_outputs, output.cpu()), axis=0)
            #     train_targets = target.cpu() if not len(train_targets) else torch.concat((train_targets, target.cpu()), axis=0)
                                
            #     if batch_idx >= len(test_loader): break

            # test-dataset
            for input, target in test_loader:
                input = model.scale(input.to(device)) # (batchsize,1,1,4)
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
            mse_c1 = loss_func(test_outputs[:,0,0,0], test_targets[:,0,0,0])
            mse_c2 = loss_func(test_outputs[:,0,0,1], test_targets[:,0,0,1])
            mse_c3 = loss_func(test_outputs[:,0,0,2], test_targets[:,0,0,2])
            mse_c4 = loss_func(test_outputs[:,0,0,3], test_targets[:,0,0,3])
        
        # plot
        test_plot, train_plot = [[], [], [], []], [[], [], [], []]
        for i in range(len(test_outputs)):
            for j in range(4):
                test_plot[j].append([test_outputs[i,0,0,j], test_targets[i,0,0,j]])
                # train_plot[j].append([train_outputs[i,0,0,j], train_targets[i,0,0,j]])
        # plt.figure(figsize=(16,9))
        # for i, item in enumerate(['test', 'train']):
        plt.figure(figsize=(16,5))
        for i, item in enumerate(['test']):
            plot_data = test_plot if i == 0 else train_plot
            for j in range(4):
                # ax = plt.subplot(2,4,j+1+4*i)
                ax = plt.subplot(1,4,j+1+4*i)
                ax.set_title(item+'_'+['c1','c2','c3','c4'][j])
                ax.plot(np.array(plot_data[j])[:,1], label='true')
                ax.plot(np.array(plot_data[j])[:,0], label='predict')
                ax.legend()
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
        LB_id = cal_id_embedding(tau, epoch, 'MLE', is_print)
        # MiND_id = cal_id_embedding(tau, epoch, 'MiND_ML')
        # MADA_id = cal_id_embedding(tau, epoch, 'MADA')
        # PCA_id = cal_id_embedding(tau, epoch, 'PCA')

        # logging
        # fp.write(f"{tau},{random_seed},{mse_c1},{mse_c2},{mse_c3},{mse_c4},{epoch},{LB_id},{MiND_id},{MADA_id},{PCA_id}\n")
        fp.write(f"{tau},{random_seed},{mse_c1},{mse_c2},{mse_c3},{mse_c4},{epoch},{LB_id},{0},{0},{0}\n")
        fp.flush()

        # if is_print: print(f'\rTau[{tau}] | Test epoch[{epoch}/{max_epoch}] | MSE: {loss_func(test_outputs, test_targets):.6f} | MLE={LB_id:.1f}, MinD={MiND_id:.1f}, MADA={MADA_id:.1f}, PCA={PCA_id:.1f}   ', end='')
        if is_print: print(f'\rTau[{tau}] | Test epoch[{epoch}/{max_epoch}] | MSE: {loss_func(test_outputs, test_targets):.6f} | MLE={LB_id:.1f}   ', end='')
        
        if checkpoint_filepath is None: break
        
    fp.close()
    if is_print: print()
    
    
def train_slow_extract_and_evolve(tau, pretrain_epoch, slow_id, delta_t, n, is_print=False, random_seed=729):
        
    # prepare
    device = torch.device('cuda:0')
    data_filepath = 'Data/data/tau_' + str(delta_t)
    log_dir = f'logs/slow_extract_and_evolve/tau_{tau}/pretrain_epoch{pretrain_epoch}/id{slow_id}/random_seed{random_seed}'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(log_dir+"/checkpoints/", exist_ok=True)

    # init model
    # model = models.EVOLVER(in_channels=1, input_1d_width=4, embed_dim=64, slow_dim=slow_id, redundant_dim=10, device=device)
    model = models.EVOLVER(in_channels=1, input_1d_width=4, embed_dim=64, slow_dim=slow_id, redundant_dim=10, tau_s=0.8, device=device)
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
    batch_size = 128
    max_epoch = 100
    weight_decay = 0.001
    L1_loss = nn.L1Loss()
    MSE_loss = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        [{'params': model.encoder_2.parameters()},
         {'params': model.decoder.parameters()}, 
         {'params': model.K_opt.parameters(), 'lr': 0.005},
         {'params': model.lstm.parameters()}],
        lr=lr, weight_decay=weight_decay) # not involve encoder_1 (freezen)
    
    # dataset
    train_dataset = _2S2FDataset(data_filepath, 'train', length=n)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dataset = _2S2FDataset(data_filepath, 'val', length=n)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    # training pipeline
    train_loss = []
    val_loss = []
    lambda_curve = [[] for _ in range(slow_id)]
    for epoch in range(1, max_epoch+1):
        
        losses = [[],[],[],[]]
        
        # train
        model.train()
        [lambda_curve[i].append(model.K_opt.Lambda[i].detach().cpu()) for i in range(slow_id) ]
        for input, _, internl_units in train_loader:
            
            input = model.scale(input.to(device)) # (batchsize,1,1,4)
            
            ####################################
            # obs —— slow —— obs(reconstruction)
            #         |
            #      koopman
            ####################################
            slow_var, embed = model.obs2slow(input)
            # koop_var = model.slow2koopman(slow_var)
            slow_obs = model.slow2obs(slow_var)
            _, embed_from_obs = model.obs2slow(slow_obs)
            
            adiabatic_loss = L1_loss(embed, embed_from_obs)
            slow_reconstruct_loss = MSE_loss(slow_obs, input)

            ################
            # n-step evolve
            ################
            fast_obs = input - slow_obs.detach()
            # obs_evol_loss, slow_evol_loss, koopman_evol_loss = 0, 0, 0
            obs_evol_loss, slow_evol_loss = 0, 0
            for i in range(1, len(internl_units)):
                
                unit = model.scale(internl_units[i].to(device)) # t+i
                
                #######################
                # slow component evolve
                #######################
                # obs ——> slow ——> koopman
                unit_slow_var, _ = model.obs2slow(unit)
                # unit_koop_var = model.slow2koopman(unit_slow_var)

                # # koopman evolve
                # t = torch.tensor([delta_t * i], device=device) # delta_t
                # unit_koop_var_next = model.koopman_evolve(koop_var, tau=t) # t ——> t + i*delta_t
                
                # slow evolve
                t = torch.tensor([delta_t * i], device=device) # delta_t
                unit_slow_var_next = model.koopman_evolve(slow_var, tau=t) # t ——> t + i*delta_t

                # koopman ——> slow ——> obs
                # unit_slow_var_next = model.koopman2slow(unit_koop_var_next)
                unit_slow_obs_next = model.slow2obs(unit_slow_var_next)
                
                #######################
                # fast component evolve
                #######################
                # fast obs evolve
                unit_fast_obs_next, _ = model.lstm_evolve(fast_obs, T=i) # t ——> t + i*delta_t
                
                ################
                # calculate loss
                ################
                # total obs evolve
                unit_obs_next = unit_slow_obs_next + unit_fast_obs_next
                
                # evolve loss
                slow_evol_loss += MSE_loss(unit_slow_var_next, unit_slow_var)
                # koopman_evol_loss += MSE_loss(unit_koop_var_next, unit_koop_var)
                obs_evol_loss += MSE_loss(unit_obs_next, unit)
            
            ###########
            # optimize
            ###########
            # all_loss = (slow_reconstruct_loss + 0.05*adiabatic_loss) + (0.5*koopman_evol_loss + 0.5*obs_evol_loss) / n
            all_loss = (slow_reconstruct_loss + 0.1*adiabatic_loss) + (0.5*slow_evol_loss + 0.5*obs_evol_loss) / n
            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()
            
            # record loss
            losses[0].append(adiabatic_loss.detach().item())
            losses[1].append(slow_reconstruct_loss.detach().item())
            # losses[2].append(koopman_evol_loss.detach().item())
            losses[2].append(slow_evol_loss.detach().item())
            losses[3].append(obs_evol_loss.detach().item())
        
        train_loss.append([np.mean(losses[0]), np.mean(losses[1]), np.mean(losses[2]), np.mean(losses[3])])
        
        # validate 
        with torch.no_grad():
            inputs = []
            slow_vars = []
            targets = []
            slow_obses = []
            slow_obses_next = []
            fast_obses = []
            fast_obses_next = []
            total_obses_next = []
            embeds = []
            embed_from_obses = []
            
            model.eval()
            for input, target, _ in val_loader:
                
                input = model.scale(input.to(device)) # (batchsize,1,1,4)
                target = model.scale(target.to(device))
                
                # obs ——> slow ——> koopman
                slow_var, embed = model.obs2slow(input)
                # koop_var = model.slow2koopman(slow_var)
                slow_obs = model.slow2obs(slow_var)
                _, embed_from_obs = model.obs2slow(slow_obs)
                
                # koopman evolve
                t = torch.tensor([tau-delta_t], device=device)
                # koop_var_next = model.koopman_evolve(koop_var, tau=t)
                slow_var_next = model.koopman_evolve(slow_var, tau=t)
                # slow_var_next = model.koopman2slow(koop_var_next)
                slow_obs_next = model.slow2obs(slow_var_next)
                
                # fast obs evolve
                fast_obs = input - slow_obs
                fast_obs_next, _ = model.lstm_evolve(fast_obs, T=n)
                
                # total obs evolve
                total_obs_next = slow_obs_next + fast_obs_next

                # record results
                inputs.append(input.cpu())
                slow_vars.append(slow_var.cpu())
                targets.append(target.cpu())
                slow_obses.append(slow_obs.cpu())
                slow_obses_next.append(slow_obs_next.cpu())
                fast_obses.append(fast_obs.cpu())
                fast_obses_next.append(fast_obs_next.cpu())
                total_obses_next.append(total_obs_next.cpu())
                embeds.append(embed.cpu())
                embed_from_obses.append(embed_from_obs.cpu())
            
            # trans to tensor
            inputs = torch.concat(inputs, axis=0)
            slow_vars = torch.concat(slow_vars, axis=0)
            targets = torch.concat(targets, axis=0)
            slow_obses = torch.concat(slow_obses, axis=0)
            slow_obses_next = torch.concat(slow_obses_next, axis=0)
            fast_obses = torch.concat(fast_obses, axis=0)
            fast_obses_next = torch.concat(fast_obses_next, axis=0)
            total_obses_next = torch.concat(total_obses_next, axis=0)
            embeds = torch.concat(embeds, axis=0)
            embed_from_obses = torch.concat(embed_from_obses, axis=0)
            
            # cal loss
            adiabatic_loss = L1_loss(embeds, embed_from_obses)
            slow_reconstruct_loss = MSE_loss(slow_obses, inputs)
            evolve_loss = MSE_loss(total_obses_next, targets)
            all_loss = 0.5*slow_reconstruct_loss + 0.5*evolve_loss + 0.1*adiabatic_loss
            if is_print: print(f'\rTau[{tau}] | epoch[{epoch}/{max_epoch}] | val: adiab_loss={adiabatic_loss:.5f}, recons_loss={slow_reconstruct_loss:.5f}, evol_loss={evolve_loss:.5f}', end='')
            
            val_loss.append(all_loss.detach().item())
            
            # plot per 5 epoch
            if epoch % 5 == 0:
                os.makedirs(log_dir+f"/val/epoch-{epoch}/", exist_ok=True)
                
                # plot slow variable vs input
                plt.figure(figsize=(16,5+2*(slow_id-1)))
                plt.title('Val Reconstruction Curve')
                for id_var in range(slow_id):
                    for index, item in enumerate(['c1', 'c2', 'c3', 'c4']):
                        plt.subplot(slow_id, 4, index+1+4*(id_var))
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
                    ax = plt.subplot(slow_id, 1, 1+id_var)
                    ax.plot(inputs[:,0,0,0], label='c1')
                    ax.plot(inputs[:,0,0,1], label='c2')
                    ax.plot(slow_vars[:, id_var], label=f'U{id_var+1}')
                    plt.xlabel(item)
                    ax.legend()
                plt.subplots_adjust(wspace=0.35, hspace=0.35)
                plt.savefig(log_dir+f"/val/epoch-{epoch}/slow_variable.pdf", dpi=300)
                plt.close()
                
                # plot fast & slow observation reconstruction curve
                plt.figure(figsize=(16,5))
                for j, item in enumerate(['c1', 'c2', 'c3', 'c4']):
                    ax = plt.subplot(1,4,j+1)
                    ax.set_title(item)
                    ax.plot(inputs[:,0,0,j], label='all_obs')
                    ax.plot(slow_obses[:,0,0,j], label='slow_obs')
                    ax.plot(fast_obses[:,0,0,j], label='fast_obs')
                    ax.legend()
                plt.subplots_adjust(wspace=0.2)
                plt.savefig(log_dir+f"/val/epoch-{epoch}/fast_slow_obs.pdf", dpi=300)
                plt.close()
                
                # plot slow observation one-step prediction curve
                plt.figure(figsize=(16,5))
                for j, item in enumerate(['c1', 'c2', 'c3', 'c4']):
                    ax = plt.subplot(1,4,j+1)
                    ax.set_title(item)
                    ax.plot(targets[:,0,0,j], label='all_true')
                    ax.plot(slow_obses_next[:,0,0,j], label='slow_predict')
                    ax.legend()
                plt.subplots_adjust(wspace=0.2)
                plt.savefig(log_dir+f"/val/epoch-{epoch}/slow_predict.pdf", dpi=300)
                plt.close()
                
                # plot fast observation one-step prediction curve
                plt.figure(figsize=(16,5))
                for j, item in enumerate(['c1', 'c2', 'c3', 'c4']):
                    ax = plt.subplot(1,4,j+1)
                    ax.set_title(item)
                    ax.plot(targets[:,0,0,j], label='all_true')
                    ax.plot(fast_obses_next[:,0,0,j], label='fast_predict')
                    ax.legend()
                plt.subplots_adjust(wspace=0.2)
                plt.savefig(log_dir+f"/val/epoch-{epoch}/fast_predict.pdf", dpi=300)
                plt.close()
                
                # plot total observation one-step prediction curve
                plt.figure(figsize=(16,5))
                for j, item in enumerate(['c1', 'c2', 'c3', 'c4']):
                    ax = plt.subplot(1,4,j+1)
                    ax.set_title(item)
                    ax.plot(targets[:,0,0,j], label='all_true')
                    ax.plot(total_obses_next[:,0,0,j], label='all_predict')
                    ax.legend()
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
    plt.figure(figsize=(6,6))
    for i in range(slow_id):
        plt.plot(lambda_curve[i], label=f'lambda[{i}]')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(log_dir+'/K_lambda_curve.pdf', dpi=300)
    np.savez(log_dir+'/K_lambda_curve.npz',lambda_curve=lambda_curve)

    # plot Koopman Lambda curve
    log_dir = f'logs/slow_extract_and_evolve/tau_{tau}/pretrain_epoch{pretrain_epoch}/id{slow_id}/random_seed{random_seed}'
    lambda_curve = np.load(log_dir+'/K_lambda_curve.npz')['lambda_curve']
    import scienceplots
    plt.style.use(['science'])
    plt.figure(figsize=(6,6))
    plt.rcParams.update({'font.size':16})
    plt.plot(np.arange(int(lambda_curve.shape[-1]/3)+1)*3, lambda_curve[0,::3], marker="o", markersize=6, label=rf'$\lambda_1$')
    plt.plot(np.arange(int(lambda_curve.shape[-1]/3)+1)*3, lambda_curve[1,::3], marker="^", markersize=6, label=rf'$\lambda_2$')
    plt.xlabel('epoch', fontsize=18)
    plt.ylabel(r'$\Lambda$', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(log_dir+'/K_lambda_curve.pdf', dpi=300)
    exit(0)


def test_evolve(tau, pretrain_epoch, ckpt_epoch, slow_id, delta_t, n, is_print=False, random_seed=729):
        
    # prepare
    device = torch.device('cuda:0')
    data_filepath = 'Data/data/tau_' + str(delta_t)
    log_dir = f'logs/slow_extract_and_evolve/tau_{tau}/pretrain_epoch{pretrain_epoch}/id{slow_id}/random_seed{random_seed}'

    # load model
    batch_size = 128
    # model = models.EVOLVER(in_channels=1, input_1d_width=4, embed_dim=64, slow_dim=slow_id, redundant_dim=10, device=device)
    model = models.EVOLVER(in_channels=1, input_1d_width=4, embed_dim=64, slow_dim=slow_id, redundant_dim=10, tau_s=0.8, device=device)
    ckpt_path = log_dir+f'/checkpoints/epoch-{ckpt_epoch}.ckpt'
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt)
    model = model.to(device)

    if is_print and delta_t==0.1:
        print('Koopman V:')
        print(model.K_opt.V.detach().cpu().numpy())
        print('Koopman Lambda:')
        print(model.K_opt.Lambda.detach().cpu().numpy())

    # # record koopman param
    # np.savetxt('koopman_params.txt', (model.K_opt.V.detach().cpu().numpy(), model.K_opt.Lambda.detach().cpu().numpy()))
    
    # dataset
    test_dataset = _2S2FDataset(data_filepath, 'test')
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    # testing pipeline        
    with torch.no_grad():
        inputs = []
        targets = []
        slow_vars = []
        slow_obses = []
        slow_obses_next = []
        fast_obses_next = []
        total_obses_next = []
        slow_vars_next = []
        slow_vars_truth = []
        
        model.eval()
        for input, target in test_loader:
            
            input = model.scale(input.to(device))
            target = model.scale(target.to(device))
        
            # obs ——> slow ——> koopman
            slow_var, _ = model.obs2slow(input)
            # koop_var = model.slow2koopman(slow_var)
            slow_obs = model.slow2obs(slow_var)
            
            # koopman evolve
            t = torch.tensor([delta_t], device=device)
            # koop_var_next = model.koopman_evolve(koop_var, tau=t)
            slow_var_next = model.koopman_evolve(slow_var, tau=t)
            # slow_var_next = model.koopman2slow(koop_var_next)
            slow_obs_next = model.slow2obs(slow_var_next)
            
            # fast obs evolve
            fast_obs = input - slow_obs
            fast_obs_next, _ = model.lstm_evolve(fast_obs, T=n)
            
            # total obs evolve
            total_obs_next = slow_obs_next + fast_obs_next

            inputs.append(input)
            targets.append(target)
            slow_vars.append(slow_var)
            slow_obses.append(slow_obs)
            slow_obses_next.append(slow_obs_next)
            fast_obses_next.append(fast_obs_next)
            total_obses_next.append(total_obs_next)   
            slow_vars_next.append(slow_var_next)   
            slow_vars_truth.append(model.obs2slow(target)[0])  
        
        inputs = model.descale(torch.concat(inputs, axis=0)).cpu()
        slow_obses = model.descale(torch.concat(slow_obses, axis=0)).cpu()
        slow_obses_next = model.descale(torch.concat(slow_obses_next, axis=0)).cpu()
        fast_obses_next = model.descale(torch.concat(fast_obses_next, axis=0)).cpu()
        slow_vars = torch.concat(slow_vars, axis=0).cpu()
        slow_vars_next = torch.concat(slow_vars_next, axis=0).cpu()
        slow_vars_truth = torch.concat(slow_vars_truth, axis=0).cpu()
        
        targets = torch.concat(targets, axis=0)
        total_obses_next = torch.concat(total_obses_next, axis=0)
    
    # metrics
    pred = total_obses_next.detach().cpu().numpy()
    true = targets.detach().cpu().numpy()
    MAPE = np.mean(np.abs((pred - true) / true))
    
    targets = model.descale(targets)
    total_obses_next = model.descale(total_obses_next)
    pred = total_obses_next.detach().cpu().numpy()
    true = targets.detach().cpu().numpy()
    MSE = np.mean((pred - true) ** 2)
    RMSE = np.sqrt(MSE)
    MAE = np.mean(np.abs(pred - true))
                
    os.makedirs(log_dir+f"/test/{delta_t}/", exist_ok=True)

    import scienceplots
    plt.style.use(['science'])

    # plot slow extract from original data
    start = 1510 - 1
    end = 3500 - 1
    sample = 10
    plt.rcParams.update({'font.size':16})
    plt.figure(figsize=(16,5))
    for j, item in enumerate([r'$c_1$', r'$c_2$', r'$c_3$', r'$c_4$']):
        ax = plt.subplot(1,4,j+1)
        t = torch.range(0,end-start-1) * 0.01
        ax.plot(t[::sample], inputs[start:end:sample,0,0,j], label=r'$X$')
        ax.plot(t[::sample], slow_obses[start:end:sample,0,0,j], marker="^", markersize=4, label=r'$X_s$')
        ax.legend()
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel('t / s', fontsize=20)
        plt.ylabel(item, fontsize=20)
    plt.subplots_adjust(wspace=0.35)
    plt.savefig(log_dir+f"/test/{delta_t}/slow_extract.pdf", dpi=300)
    plt.savefig(log_dir+f"/test/{delta_t}/slow_extract.jpg", dpi=300)
    plt.close()

    # plot slow variable vs input
    sample = 4
    plt.figure(figsize=(16,5+2*(slow_id-1)))
    for id_var in range(slow_id):
        for index, item in enumerate([r'$c_1$', r'$c_2$', r'$c_3$', r'$c_4$']):
            plt.subplot(slow_id, 4, index+1+4*(id_var))
            plt.scatter(inputs[::sample,0,0,index], slow_vars[::sample, id_var], s=2)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.xlabel(item, fontsize=20)
            plt.ylabel(rf'$u_{id_var+1}$', fontsize=20)
    plt.subplots_adjust(wspace=0.55, hspace=0.35)
    plt.savefig(log_dir+f"/test/{delta_t}/slow_vs_input.pdf", dpi=150)
    plt.savefig(log_dir+f"/test/{delta_t}/slow_vs_input.jpg", dpi=150)
    plt.close()
    
    # plot koopman space prediction curve
    plt.figure(figsize=(16,5))
    for j, item in enumerate(['u1', 'u2']):
        ax = plt.subplot(1,4,j+1)
        ax.set_title(item)
        ax.plot(slow_vars_truth[:,j], label='"true"')
        ax.plot(slow_vars_next[:,j], label='predict')
        ax.legend()
    plt.subplots_adjust(wspace=0.2)
    plt.savefig(log_dir+f"/test/{delta_t}/koopman_pred.pdf", dpi=300)
    plt.close()

    # plot slow observation prediction curve
    plt.figure(figsize=(16,5))
    for j, item in enumerate(['c1', 'c2', 'c3', 'c4']):
        ax = plt.subplot(1,4,j+1)
        ax.set_title(item)
        ax.plot(true[:,0,0,j], label='true')
        ax.plot(slow_obses_next[:,0,0,j], label='predict')
        ax.legend()
    plt.subplots_adjust(wspace=0.2)
    plt.savefig(log_dir+f"/test/{delta_t}/slow_pred.pdf", dpi=300)
    plt.close()
    
    # plot fast observation prediction curve
    plt.figure(figsize=(16,5))
    for j, item in enumerate(['c1', 'c2', 'c3', 'c4']):
        ax = plt.subplot(1,4,j+1)
        ax.set_title(item)
        ax.plot(true[:,0,0,j], label='true')
        ax.plot(fast_obses_next[:,0,0,j], label='predict')
        ax.legend()
    plt.subplots_adjust(wspace=0.2)
    plt.savefig(log_dir+f"/test/{delta_t}/fast_pred.pdf", dpi=300)
    plt.close()

    # plot total observation prediction curve
    plt.figure(figsize=(16,5))
    for j, item in enumerate(['c1', 'c2', 'c3', 'c4']):
        ax = plt.subplot(1,4,j+1)
        ax.set_title(item)
        ax.plot(true[:,0,0,j], label='true')
        ax.plot(pred[:,0,0,j], label='predict')
        ax.legend()
    plt.subplots_adjust(wspace=0.2)
    plt.savefig(log_dir+f"/test/{delta_t}/total_pred.pdf", dpi=300)
    plt.close()

    c1_evolve_mae = torch.mean(torch.abs(slow_obses_next[:,0,0,0] - true[:,0,0,0]))
    c2_evolve_mae = torch.mean(torch.abs(slow_obses_next[:,0,0,1] - true[:,0,0,1]))
    
    return MSE, RMSE, MAE, MAPE, c1_evolve_mae.item(), c2_evolve_mae.item()
        
    
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

    ckpt_epoch = 100

    if not long_test:
        # train
        train_slow_extract_and_evolve(tau, pretrain_epoch, slow_id, round(tau/n,3), n, is_print=is_print, random_seed=random_seed)
        # plot mse curve of each id
        try: plot_slow_ae_loss(tau, pretrain_epoch, delta_t, id_list) 
        except: pass
    else:
        # test evolve
        for i in tqdm(range(1, 6*n+1+2)):
            delta_t = round(tau/n*i, 3)
            MSE, RMSE, MAE, MAPE, c1_mae, c2_mae = test_evolve(tau, pretrain_epoch, ckpt_epoch, slow_id, delta_t, i, is_print, random_seed)
            with open(f'results/pretrain{pretrain_epoch}_evolve_test_{tau}.txt','a') as f:
                f.writelines(f'{delta_t}, {random_seed}, {MSE}, {RMSE}, {MAE}, {MAPE}, {c1_mae}, {c2_mae}\n')
    
    
def data_generator_pipeline(trace_num=256+32+32, total_t=9, dt=0.0001):
    
    # generate original data
    generate_original_data(trace_num=trace_num, total_t=total_t, dt=dt)
    
    
def id_esitimate_pipeline(cpu_num=1, trace_num=256+32+32):
    
    tau_list = np.arange(0., 3.1, 0.1)
    workers = []
    
    # id esitimate sub-process
    for tau in tau_list:

        # generate dataset
        tau = round(tau, 2)
        generate_dataset(trace_num, tau, None, is_print=True)

        for seed in range(10, 15+1):
            is_print = True if len(workers)==0 else False
            workers.append(Process(target=worker_1, args=(tau, trace_num, seed, cpu_num, is_print), daemon=True))
            workers[-1].start()
    while any([sub.exitcode==None for sub in workers]):
        pass

    [plot_epoch_test_log(round(tau,2), max_epoch=100+1) for tau in tau_list]
    plot_id_per_tau(tau_list, np.arange(95,100+1,1))
    
    print('ID Esitimate Over!')


def slow_evolve_pipeline(trace_num=256+32+32, n=10, cpu_num=1, long_test=False):
    
    tau_list = [0.8]
    id_list = [2]
    workers = []
    
    # generate dataset sub-process
    sample_num = None
    for tau in tau_list:
        if not long_test:
            # dataset for training
            workers.append(Process(target=generate_dataset, args=(trace_num, round(tau/n,3), sample_num, True, n), daemon=True))
            workers[-1].start()
            
        # dataset for testing
        for i in range(1, 6*n+1+2):
            print(f'processing testing dataset [{i}/{6*n+2}]')
            delta_t = round(tau/n*i, 3)
            generate_dataset(trace_num, delta_t, sample_num, True)
            # workers.append(Process(target=generate_dataset, args=(trace_num, delta_t, sample_num, False), daemon=True))
            # workers[-1].start()
    while any([sub.exitcode==None for sub in workers]):
        pass
    workers = []
    
    # slow evolve sub-process
    for tau in tau_list:
        for pretrain_epoch in [100]:
            for slow_id in id_list:
                for random_seed in range(1,10+1):
                    is_print = True if len(workers)==0 else False
                    workers.append(Process(target=worker_2, args=(tau, pretrain_epoch, slow_id, n, random_seed, cpu_num, is_print, id_list, long_test), daemon=True))
                    workers[-1].start()
    while any([sub.exitcode==None for sub in workers]):
        pass
    
    print('Slow-observation Evolve Over!')


def test_koopman_evolve():
        
    # prepare
    device = torch.device('cpu')

    # load model
    model = models.EVOLVER(in_channels=1, input_1d_width=4, embed_dim=64, slow_dim=2, redundant_dim=10, device=device)
    ckpt_path = f'logs/evolve/tau_1.0/pretrain_epoch30/id2/random_seed2/checkpoints/epoch-10.ckpt'
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt)
    model = model.to(device)

    print(model.K_opt.V)
    print(model.K_opt.Lambda)
    model.K_opt.V = torch.nn.Parameter(torch.tensor([[1.0,0.0],[0.0,1.0]]).to(device))
    model.K_opt.Lambda = torch.nn.Parameter(torch.tensor([-1.0,-2.0]).to(device))

    # obs ——> slow
    input = torch.tensor([[[[2.0,2.0,2.0,2.0]]]]).to(device)
    input = model.scale(input)
    slow_var, _ = model.obs2slow(input)
    
    # koopman evolve
    t_s = [0.0]
    slow_vars = [slow_var.detach().cpu().numpy()]
    obs_vars = [torch.tensor([[[[2.0,2.0,2.0,2.0]]]]).numpy()]
    for delta_t in np.linspace(0.1, 5.0, 50):
        t = torch.tensor([delta_t], device=device).float()
        slow_var_next = model.koopman_evolve(slow_var, tau=t)
        obs_var_next = model.slow2obs(slow_var_next)

        t_s.append(delta_t)
        slow_vars.append(slow_var_next.detach().cpu().numpy())
        obs_vars.append(model.descale(obs_var_next).detach().cpu().numpy())
    t_s = np.array(t_s)
    slow_vars = np.array(slow_vars)
    obs_vars = np.array(obs_vars)

    os.makedirs('test/', exist_ok=True)

    # plot koopman space
    plt.figure(figsize=(16,5))
    plt.plot(t_s, slow_vars[:,0,0], label='u1')
    plt.plot(t_s, slow_vars[:,0,1], label='u2')
    plt.legend()
    plt.savefig('test/koopman_evol.pdf', dpi=300)

    # plot obs space
    plt.figure(figsize=(16,5))
    plt.plot(t_s, obs_vars[:,0,0,0,0], label='c1')
    plt.plot(t_s, obs_vars[:,0,0,0,1], label='c2')
    plt.plot(t_s, obs_vars[:,0,0,0,2], label='c3')
    plt.plot(t_s, obs_vars[:,0,0,0,3], label='c4')
    plt.legend()
    plt.savefig('test/obs_evol.pdf', dpi=300)
    

if __name__ == '__main__':
    
    trace_num = 200
    
    data_generator_pipeline(trace_num=trace_num, total_t=5.1, dt=0.01)
    # id_esitimate_pipeline(trace_num=trace_num)
    slow_evolve_pipeline(trace_num=trace_num, n=8, long_test=False)
    # slow_evolve_pipeline(trace_num=trace_num, n=8, long_test=True)

    # test_koopman_evolve()

    torch.cuda.empty_cache()