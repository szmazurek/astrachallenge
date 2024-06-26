from train import train_model
from unet.unet_model import UNet
import torch

if __name__ == '__main__':
    model = UNet(n_channels=1, n_classes=1)
    train_model(
        model,
        torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        epochs = 100,
        batch_size = 6,
        learning_rate = 1e-5,
        save_checkpoint = True,
        amp = False,
        weight_decay = 1e-8,
        gradient_clipping = 1.0,
        n_train = .85,
        val_freq = 20,
        weights_loss=(1,0,1),
        augment = False
    )