import os
import copy
from collections import defaultdict
import numpy as np
import torch
import einops
import pdb
from torch.utils.data import DataLoader, random_split
import wandb

from .arrays import batch_to_device, to_np, to_device, apply_dict
from .timer import Timer
from .cloud import sync_logs

def cycle(dl):
    while True:
        for data in dl:
            yield data
            
def get_dataset(dataset,
                batch_size=2,
                val_set_size=0.05,
                results_dir=None,
                save_indices=False,
                ):
    
    full_dataset = dataset
    print(full_dataset)

    # split into train and validation
    train_size = len(full_dataset) - len(full_dataset)*val_set_size
    valid_size = len(full_dataset)*val_set_size
    
    train_subset, val_subset = random_split(full_dataset, [int(train_size+1), int(valid_size)])
    
    #By using cycle the dataloader's iterator is infinite since cyclic
    train_dataloader = cycle(DataLoader(
        train_subset, batch_size=batch_size, num_workers=1, shuffle=True, pin_memory=True)) 
    val_dataloader = cycle(DataLoader(
        val_subset, batch_size=batch_size, num_workers=1, shuffle=True, pin_memory=True)) 

    if save_indices:
        # save the indices of training and validation sets (for later evaluation)
        torch.save(train_subset.indices, os.path.join(results_dir, f'train_subset_indices.pt'))
        torch.save(val_subset.indices, os.path.join(results_dir, f'val_subset_indices.pt'))

    return train_subset, train_dataloader, val_subset, val_dataloader

class EarlyStopper:
    # https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch

    def __init__(self, patience=10, min_delta=0):
        self.patience = patience  # use -1 to deactivate it
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = torch.inf

    def early_stop(self, validation_loss):
        if self.patience == -1:
            return
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer = None,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        n_samples=2,
        bucket=None,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        train_subset, train_dataloader, valid_subset, valid_dataloader = get_dataset(
            dataset=dataset, 
            batch_size=train_batch_size, 
            results_dir=results_folder,
            save_indices=True
            )
        self.dataset = dataset
        self.dataloader_train = train_dataloader
        self.dataloader_valid = valid_dataloader
        # self.dataloader = cycle(torch.utils.data.DataLoader(
        #     self.dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        # ))
        # self.dataloader_vis = cycle(torch.utils.data.DataLoader(
        #     self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        # ))
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference
        self.n_samples = n_samples

        self.reset_parameters()
        self.step = 0

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, 
              n_train_steps, 
              steps_til_summary,
              epoch,
              early_stopper_patience=-1,
              ):
        
        early_stopper = EarlyStopper(patience=early_stopper_patience, min_delta=0)
        train_losses_l = []
        validation_losses_l = []
        n_valid_steps = 1000
        timer = Timer()
        stop_training = False
        for step in range(n_train_steps):
            train_steps_current = self.step  # Global
            train_step = step + 1            # On epoch
            #----------------------------------------------------------------------------------#
            #-----------------------------------TRAINING---------------------------------------#                 
            #----------------------------------------------------------------------------------#
            self.model.train()
            for i in range(self.gradient_accumulate_every):
                #Grab next batch with "next"
                batch = next(self.dataloader_train)
                batch = batch_to_device(batch)
                raw_loss, infos = self.model.loss(*batch)
                train_losses_l.append(raw_loss.detach().cpu().numpy())         
                loss = raw_loss / self.gradient_accumulate_every          
                # Backpropagation
                loss.backward()
            #Optimizer
            self.optimizer.step()
            self.optimizer.zero_grad()  
            
            #---------------------------------------------------------------------------------------#
            #----------------------------------VALIDATION-------------------------------------------#                 
            #---------------------------------------------------------------------------------------#
            if train_step % steps_til_summary == 0:
                # TRAINING
                train_loss = np.mean(train_losses_l)
                train_losses_log = {'TRAINING Diffusion_loss': train_loss}   
                                              
                #####################################################################################
                # VALIDATION LOSS and SUMMARY
                self.model.eval()
                if self.dataloader_valid is not None:
                    print("Running validation...")
                    total_val_loss = 0.
                    with torch.no_grad():
                        for step_val, batch_dict_val in enumerate(self.dataloader_valid):
                            batch_valid = batch_to_device(batch_dict_val)
                            loss_valid, infos_valid = self.model.loss(*batch_valid)
                            validation_losses_l.append(loss_valid.detach().cpu().numpy()) 
                            total_val_loss += loss_valid
                            step_val = step_val + 1
                            if step_val % n_valid_steps == 0:
                                break
                             
                    print("... finished validation.")
                    valid_loss = np.mean(validation_losses_l)
                    validation_losses_log = {f'VALIDATION Diffusion_loss': valid_loss}
                    
                wandb.log({**train_losses_log, **validation_losses_log}, step=train_steps_current)
                if early_stopper.early_stop(total_val_loss):
                    print(f'Early stopped training at {train_steps_current} steps.')
                    stop_training = True
           
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                self.save(label)

            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                #import ipdb; ipdb.set_trace()
                print(f'{self.step}: {raw_loss:8.4f} | {infos_str} | t: {timer():8.4f}')

            if self.step == 0 and self.sample_freq:
                self.render_reference(self.n_reference)

            if self.sample_freq and self.step % self.sample_freq == 0:
                self.render_samples(n_samples=self.n_samples)

            self.step += 1
            if stop_training:
                    return stop_training
        return stop_training
        

    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}')
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10):
        '''
            renders training points
        '''
        if self.renderer == None:
            return
        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch.trajectories)
        conditions = to_np(batch.conditions[0])[:,None]

        ## [ batch_size x horizon x observation_dim ]
        normed_observations = trajectories[:, :, self.dataset.action_dim:]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

        # from diffusion.datasets.preprocessing import blocks_cumsum_quat
        # # observations = conditions + blocks_cumsum_quat(deltas)
        # observations = conditions + deltas.cumsum(axis=1)

        #### @TODO: remove block-stacking specific stuff
        # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
        # observations = blocks_add_kuka(observations)
        ####

        savepath = os.path.join(self.logdir, f'_sample-reference.png')
        self.renderer.composite(savepath, observations)

    def render_samples(self, batch_size=2, n_samples=2):
        '''
            renders samples from (ema) diffusion model
        '''
        if self.renderer == None:
            return
        for i in range(batch_size):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.conditions, 'cuda:0')

            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            samples = self.ema_model.conditional_sample(conditions)
            trajectories = to_np(samples.trajectories)
            #samples = to_np(samples)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = trajectories[:, :, self.dataset.action_dim:]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(batch.conditions[0])[:,None]

            # from diffusion.datasets.preprocessing import blocks_cumsum_quat
            # observations = conditions + blocks_cumsum_quat(deltas)
            # observations = conditions + deltas.cumsum(axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            normed_observations = np.concatenate([
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_observations
            ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            #### @TODO: remove block-stacking specific stuff
            # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
            # observations = blocks_add_kuka(observations)
            ####

            savepath = os.path.join(self.logdir, f'sample-{self.step}-{i}.png')
            self.renderer.composite(savepath, observations)
