import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics.classification import Dice
from utils.matthews_loss import MCC_Loss
from monai.optimizers import Novograd

import wandb
from evaluate import evaluate
from dataloader import load_data

def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 2,
        learning_rate: float = 1e-5,
        save_checkpoint: bool = True,
        amp: bool = False,
        weight_decay: float = 1e-8,
        gradient_clipping: float = 1.0,
        n_train: float = .8,
        val_freq: int = 2,
        weights_loss: tuple = (1, 1, 0),
        augment: bool = False
):
    alpha, betta, gamma = weights_loss
    # Load data
    n_train = .8
    n_val = round(1 - n_train,2)
    train_set, val_set = load_data(train_size=n_train, test_size=n_val)
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        **loader_args # Define some rigid body transformation
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        drop_last=True,
        **loader_args)

    # Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate, 
        weight_decay=weight_decay)#, momentum=momentum, foreach=True)
    """ optimizer = Novograd(
        model.parameters(),
        lr=learning_rate, 
        weight_decay=weight_decay)#, momentum=momentum, foreach=True) """
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    dice = Dice(zero_division=1e-6)
    mcc = MCC_Loss(zero_division=1)#e-6)
    global_step = 0
    
    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', entity="joansano")
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_freq=val_freq, save_checkpoint=save_checkpoint, amp=amp, weights_loss=weights_loss)
    )
    logging.basicConfig(level=logging.INFO)
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Validation freq: {val_freq}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Mixed Precision: {amp},
        Augmentation: {augment}
    ''')

    # Training loop
    model.to(device=device)
    dice.to(device)
    mcc.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=len(train_set), desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for (images, masks) in train_loader:
                images = images.to(device=device, dtype=torch.float32)
                masks = masks.to(device=device, dtype=torch.long)

                assert images.shape[1] == model.n_channels, \
                        f'Network has been defined with {model.n_channels} input channels, ' \
                        f'but loaded images have {images.shape[1]} channels. Please check that ' \
                        'the images are loaded correctly.'
                
                masks_pred = model(images)
                loss = alpha*criterion(masks_pred, masks.float())
                loss += betta*(1 - dice((F.sigmoid(masks_pred)>.5).int(), masks.int()))
                loss += gamma*mcc((F.sigmoid(masks_pred)>.5).int(), masks.int())

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                """ experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                }) """
                pbar.set_postfix(**{'loss (batch)': loss.item()})
        experiment.log({
            'train loss': epoch_loss/len(train_set),
            'step': global_step
        })

        # Evaluation round
        division_step = epoch % val_freq
        if division_step == 0 or epoch==(epochs+1):
            """ histograms = {}
            for tag, value in model.named_parameters():
                    tag = tag.replace('/', '.')
                    if not (torch.isinf(value) | torch.isnan(value)).any():
                            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                    if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu()) """

            val_score = evaluate(model, val_loader, device, amp, dice, epoch)
            scheduler.step(val_score)
            experiment.log({
                'dice score': val_score,
                'step': global_step
            }, step=global_step)
            logging.info('Validation Dice score: {}'.format(val_score))