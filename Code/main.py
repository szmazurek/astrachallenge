from train import train_model, train_model_kfold
from unet.unet_model import UNet
import torch
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--n_folds', type=int, default=None)
    n_folds = parser.parse_args().n_folds
    
    model = UNet(n_channels=1, n_classes=1)
    if n_folds:
        train_model_kfold(
        model,
        torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        epochs = 10,
        batch_size = 8,
        learning_rate = 1e-5,
        save_checkpoint = True,
        amp = False,
        weight_decay = 1e-6,
        gradient_clipping = 1.0,
        n_train = .85,
        val_freq = 20,
        weights_loss=(1,0,1),
        augment = False,
        n_folds=n_folds,
        grad_accumulation_steps=5
    )
    else:
        train_model(
            model,
            torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            epochs = 100,
            batch_size = 6,
            learning_rate = 1e-5,
            save_checkpoint = True,
            amp = False,
            weight_decay = 1e-6,
            gradient_clipping = 1.0,
            n_train = .85,
            val_freq = 20,
            weights_loss=(1,0,1),
            augment = False
        )