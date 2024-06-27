from train import train_model, train_model_kfold
from unet.unet_model import UNet
import torch
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--n_folds', type=int, default=None)
    n_folds = parser.parse_args().n_folds
    
    model = UNet(n_channels=1, n_classes=1)
    data_path = "/net/pr2/projects/plgrid/plggsano/Joan/AstraZeneca/catalyst_open_innovation_challenge/"
    if n_folds:
        train_model_kfold(
            data_path, 
            model,
            torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            epochs = 100,
            batch_size = 8,
            learning_rate = 1e-5,
            save_folds = "./validation_results/",
            amp = False,
            weight_decay = 1e-6,
            gradient_clipping = 1.0,
            weights_loss=(1,0.75,0.25),
            augment = False,
            n_folds=n_folds,
        )
    else:
        train_model(
            data_path,
            model,
            torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            epochs = 100,
            batch_size = 8,
            learning_rate = 1e-5,
            save = "./validation_results/",
            amp = False,
            weight_decay = 1e-6,
            gradient_clipping = 1.0,
            n_train = .9,
            val_freq = 50,
            weights_loss=(1,0.75,0.25),
            augment = False
        )