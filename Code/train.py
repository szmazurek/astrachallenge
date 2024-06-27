import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics.classification import Dice, BinaryMatthewsCorrCoef
from copy import deepcopy
from sklearn.model_selection import KFold

from evaluate import evaluate
from dataloader import load_data, load_data_no_split, augmentation

def train_model(
        data_path,
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
    n_val = round(1 - n_train,2)
    train_set, val_set = load_data(data_path, train_size=n_train, test_size=n_val)
    if augment:
        train_set = augmentation(train_set)
    loader_args = dict(batch_size=batch_size, num_workers=6, pin_memory=True)
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        drop_last=False,
        **loader_args # Define some rigid body transformation
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        drop_last=False,
        **loader_args)

    # Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate, 
        weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    dice = Dice(zero_division=1e-6)
    mcc = BinaryMatthewsCorrCoef()
    global_step = 0
    
    # (Initialize logging)
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
                loss += gamma*(1 - mcc((F.sigmoid(masks_pred)>.5).int(), masks.int()))

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})

        # Evaluation round
        division_step = epoch % val_freq
        if division_step == 0 or epoch==(epochs+1):
            val_score = evaluate(model, val_loader, device, amp, dice, epoch)
            scheduler.step(val_score)
            logging.info('Validation Dice score: {}'.format(val_score))


def train_model_kfold(
        data_path,
        model : nn.Module,
        device,
        epochs: int = 5,
        n_folds: int = 5,
        batch_size: int = 2,
        learning_rate: float = 1e-5,
        save_checkpoint: bool = True,
        amp: bool = False,
        weight_decay: float = 1e-8,
        gradient_clipping: float = 1.0,
        weights_loss: tuple = (1, 1, 0),
        augment: bool = False,
        grad_accumulation_steps: int = 1
):
    alpha, betta, gamma = weights_loss
    model_initial = deepcopy(model)   
    
    # Load data
    train_set_full = load_data_no_split(data_path)
    kfold = KFold(n_splits=n_folds, shuffle=True)
    loader_args = dict(batch_size=batch_size, num_workers=6, pin_memory=True, prefetch_factor=3)
    for fold, (train_idx, test_idx) in enumerate(kfold.split(train_set_full)):
        model = deepcopy(model_initial)
        
        train_set = [train_set_full[i] for i in train_idx]
        val_set = [train_set_full[i] for i in test_idx]
        if augment: # 90 degree rotations for each one of the images in the training set
            train_set = augmentation(train_set)
            
        train_loader = DataLoader(
            train_set,
            shuffle=True,
            drop_last=False,
            **loader_args # Define some rigid body transformation
        )
        val_loader = DataLoader(
            val_set,
            shuffle=False,
            drop_last=False,
            **loader_args)

        # Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate, 
            weight_decay=weight_decay)
        
        #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
        grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
        criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
        dice = Dice(zero_division=1e-6)
        mcc = BinaryMatthewsCorrCoef()
        global_step = 0
        
        # (Initialize logging)
        logging.basicConfig(level=logging.INFO)
        logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Fold:            {fold+1}
            Learning rate:   {learning_rate}
            Checkpoints:     {save_checkpoint}
            Device:          {device.type}
            Mixed Precision: {amp},
            Augmentation: {augment}
        ''')

        # Training loop
        model.to(device=device)
        dice.to(device)
        mcc.to(device)
        with tqdm(total=epochs, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for epoch in range(1, epochs + 1):
                model.train()
                epoch_loss = 0
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
                    loss += gamma*(1 - mcc((F.sigmoid(masks_pred)>.5).int(), masks.int()))
                    #### GRAD ACCUMULATION

                    grad_scaler.scale(loss).backward()
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                    grad_scaler.unscale_(optimizer)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    optimizer.zero_grad(set_to_none=True)


                    global_step += 1
                    epoch_loss += loss.item()
            pbar.update(images.shape[0])
            pbar.set_postfix(**{'loss (batch)': epoch_loss/epochs})

        # Evaluation round
        val_score = evaluate(model, val_loader, device, amp, dice, epoch, fold=fold)
        logging.info('Validation Dice score: {}'.format(val_score))