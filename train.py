from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
from dataset import load_PairSpecDataset
from util import rescale, find_max_epoch, print_size
from logger import Logger
from util import LinearWarmupCosineDecay
from model import CleanSpecNet
from losses import l1_loss as loss_fn
#from losses import cleanspecnet_official_loss as loss_fn
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()  # Disables interactivity

import numpy as np
import time
import argparse
import random
import json
import os
from torch.utils.tensorboard import SummaryWriter

#loss_fn = nn.MSELoss()

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

def prepare_directories_and_logger(output_dir, log_dir, ckpt_dir, rank):
    if rank == 0:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
            os.chmod(output_dir, 0o775)

        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
            os.chmod(log_dir, 0o775)    

        if not os.path.isdir(ckpt_dir):
            os.makedirs(ckpt_dir)
            os.chmod(ckpt_dir, 0o775)                        

        logger = Logger(log_dir)
    else:
        logger = None
    return logger

def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model_dict = checkpoint_dict['state_dict']
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    #model.load_state_dict(checkpoint_dict['state_dict'], strict=False)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint_dict['state_dict'], strict=False)

    if len(missing_keys) > 0:
        print("Missing keys:", missing_keys)
    if len(unexpected_keys) > 0:
        print("Unexpected keys:", unexpected_keys)
            
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def validate(model, val_loader, iteration, trainset_config, logger, device):
    """
    Handles all the validation scoring and logging.

    Parameters:
    - model (nn.Module): The neural network model to validate.
    - val_loader (Dataset): The validation dataloader.
    - iteration (int): Current training iteration.
    - logger (Logger): Logger instance for TensorBoard logging.
    - device (torch.device): Device to perform computations on.
    """
    model.eval()
    val_loss = 0.0
    num_batches = 0

    with torch.no_grad():

        for i, (x, y) in enumerate(val_loader):
            x, y = x.to(device), y.to(device)

            # Forward pass
            y_pred = model(x)  # Assumindo que model retorna apenas y_pred

            # Calcular a perda
            loss = loss_fn(y_pred, y)

            # Reduzir a perda se estiver usando múltiplas GPUs
            #if distributed_run:
            #    reduced_val_loss = reduce_tensor(loss.data, n_gpus).item()
            #else:
            #    reduced_val_loss = loss.item()
            reduced_val_loss = loss.item()

            val_loss += reduced_val_loss
            num_batches += 1

        # Calcular a perda média de validação
        val_loss /= num_batches

    model.train()

    #if rank == 0 and logger is not None:
    if logger is not None:
        print(f"Validation loss at iteration {iteration}: {val_loss:.6f}")

        # save to tensorboard
        logger.add_scalar("Validation/Loss", val_loss, iteration)

        num_samples = min(4, x.size(0))

        # Define Griffin-Lim transformer if not already defined
        if 'griffin_lim' not in locals():
            n_fft = trainset_config['n_fft']
            hop_length = trainset_config['hop_length']
            win_length = trainset_config['win_length']

            griffin_lim = torchaudio.transforms.GriffinLim(
                n_fft=n_fft, hop_length=hop_length, win_length=win_length,
                power=1.0, n_iter=32
            ).to(x.device)

        for i in range(num_samples):
            # Get spectrograms
            clean_spec = y[i]  # Shape: (freq_bins, time_steps)
            denoised_spec = y_pred[i]
            noisy_spec = x[i]
            
            # Convert spectrograms to numpy arrays for plotting
            clean_spec_np = clean_spec.cpu().numpy()
            denoised_spec_np = denoised_spec.cpu().detach().numpy()
            noisy_spec_np = noisy_spec.cpu().numpy()
            
            # Plot spectrograms
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(clean_spec_np, origin='lower', aspect='auto')
            axs[0].set_title('Clean Spectrogram')
            axs[1].imshow(denoised_spec_np, origin='lower', aspect='auto')
            axs[1].set_title('Denoised Spectrogram')
            axs[2].imshow(noisy_spec_np, origin='lower', aspect='auto')
            axs[2].set_title('Noisy Spectrogram')
            plt.tight_layout()
            logger.add_figure('Spectrograms/Sample_{}'.format(i), fig, iteration)
            plt.close(fig)
            
            # Reconstruct audio using Griffin-Lim
            # The spectrograms are magnitude spectrograms
            # Griffin-Lim expects inputs of shape (batch_size, freq_bins, time_steps)
            # So we need to unsqueeze(0) to add batch dimension
            clean_spec_batch = clean_spec.unsqueeze(0)
            denoised_spec_batch = denoised_spec.unsqueeze(0)
            noisy_spec_batch = noisy_spec.unsqueeze(0)
            
            # Reconstruct waveforms
            clean_waveform = griffin_lim(clean_spec_batch)
            denoised_waveform = griffin_lim(denoised_spec_batch)
            noisy_waveform = griffin_lim(noisy_spec_batch)

            # Log audio samples to TensorBoard
            sample_rate = trainset_config['sample_rate']
            logger.add_audio('Audio/Clean_{}'.format(i), clean_waveform.squeeze(), iteration, sample_rate=sample_rate)
            logger.add_audio('Audio/Denoised_{}'.format(i), denoised_waveform.squeeze(), iteration, sample_rate=sample_rate)
            logger.add_audio('Audio/Noisy_{}'.format(i), noisy_waveform.squeeze(), iteration, sample_rate=sample_rate)


def train(num_gpus, rank, group_name, 
          exp_path, checkpoint_path, log, optimization, device=None):
    
    device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # setup local experiment path
    if rank == 0:
        print('exp_path:', exp_path)
    
    # Create tensorboard logger.
    output_dir = os.path.join(log["directory"], exp_path)
    log_directory = os.path.join(output_dir, 'logs')
    ckpt_dir = os.path.join(output_dir, 'checkpoint')

    weight_decay = optimization["weight_decay"]
    learning_rate = optimization["learning_rate"]
    max_norm = optimization["max_norm"]
    batch_size = optimization["batch_size_per_gpu"]

    logger = prepare_directories_and_logger(
        output_dir, log_directory, ckpt_dir, rank)

    # distributed running initialization
    if num_gpus > 1:
        init_distributed(rank, num_gpus, group_name, **dist_config)    

    # load training data
    print('Loading training dataloader...')
    trainloader = load_PairSpecDataset(**trainset_config, 
                            subset='train',
                            batch_size=batch_size, 
                            num_gpus=num_gpus)
    
    print('Data loaded')
    
    print('Loading val dataloader...')    
    testloader = load_PairSpecDataset(**trainset_config, 
                            subset='test',
                            batch_size=batch_size, 
                            num_gpus=num_gpus)

    print('Data loaded')
            
    # predefine model
    model = CleanSpecNet(**network_config).to(device)
    print_size(model)

    # apply gradient all reduce
    if num_gpus > 1:
        model = apply_gradient_allreduce(model)

    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    '''
    time0 = time.time()
    if log["ckpt_iter"] == 'max':
        ckpt_iter = find_max_epoch(ckpt_directory)
    else:
        ckpt_iter = log["ckpt_iter"]
    '''

    # load checkpoint
    n_iter = 0
    if checkpoint_path is not None:
        try:
            model, optimizer, _learning_rate, iteration = load_checkpoint(
                checkpoint_path, model, optimizer)        
            print('Model at %s has been loaded' % (checkpoint_path))
            print('checkpoint model loaded successfully')        
            if False: #use_saved_learning_rate:
                learning_rate = _learning_rate             
            n_iter = iteration + 1                   
        except:
            print(f'No valid checkpoint model found at {checkpoint_path}, start training from initialization.')           

    # define learning rate scheduler
    scheduler = LinearWarmupCosineDecay(
                    optimizer,
                    lr_max=learning_rate,
                    n_iter=optimization["n_iters"],
                    iteration=n_iter,
                    divider=25,
                    warmup_proportion=0.05,
                    phase=('linear', 'cosine'),
                )
    # training 
    epoch = 1
    print('Starting training...')
    while n_iter < optimization["n_iters"] + 1:
        # for each epoch
        for noisy_spec, clean_spec in trainloader:             
            noisy_spec = noisy_spec.to(device)  # Shape: (batch_size, freq_bins, time_steps)
            clean_spec = clean_spec.to(device)

            # back-propagation
            optimizer.zero_grad()
            denoised_audio = model(noisy_spec)  # Output shape: (batch_size, freq_bins, time_steps)
            loss = loss_fn(clean_spec, denoised_audio)
            loss.backward()     

            if torch.isnan(loss).any():
                print("Loss contains NaN, terminating training")
                break

            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            scheduler.step()
            optimizer.step()

            print("Epoch: {} \t iteration: {} \t\tloss: {:.7f}".format(epoch, n_iter, loss.item()), flush=True)

            # output to log
            if n_iter > 0 and n_iter % 10 == 0 and rank == 0:
                # save to tensorboard
                logger.add_scalar("Train/Train-Loss", loss.item(), n_iter)
                logger.add_scalar("Train/Gradient-Norm", grad_norm, n_iter)
                logger.add_scalar("Train/learning-rate", optimizer.param_groups[0]["lr"], n_iter)

            if n_iter > 0 and n_iter % log["iters_per_valid"] == 0 and rank == 0:
                validate(model, testloader, n_iter, trainset_config, logger, device)

            # save checkpoint
            if n_iter > 0 and n_iter % log["iters_per_ckpt"] == 0 and rank == 0:
                checkpoint_name = '{}.pkl'.format(n_iter)
                checkpoint_path = os.path.join(ckpt_dir, checkpoint_name)
                save_checkpoint(model, optimizer, learning_rate, n_iter, checkpoint_path)
                print('model at iteration %s is saved' % n_iter)


            n_iter += 1

        epoch += 1

    # After training, close TensorBoard.
    if rank == 0:
        logger.close()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/config.json', 
                        help='JSON file for configuration')
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='rank of process for distributed')
    parser.add_argument('-g', '--group_name', type=str, default='',
                        help='name of group for distributed')
    args = parser.parse_args()

    # Parse configs. Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)

    train_config            = config["train_config"]        # training parameters
    dist_config             = config["dist_config"]         # to initialize distributed training
    network_config          = config["network_config"]      # to define network
    trainset_config         = config["trainset_config"]     # to load trainset

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        if args.group_name == '':
            print("WARNING: Multiple GPUs detected but no distributed group set")
            print("Only running 1 GPU. Use distributed.py for multiple GPUs")
            num_gpus = 1

    if num_gpus == 1 and args.rank != 0:
        raise Exception("Doing single GPU training on rank > 0")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    train(num_gpus, args.rank, args.group_name, **train_config)