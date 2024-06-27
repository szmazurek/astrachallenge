import argparse
import yaml
import os
from unet.unet_model import UNet
from train import train_model, train_model_kfold
import torch

# Define argparse
parser = argparse.ArgumentParser("\nAutomatic detection of tumorigenic ares\n")
sub_parser = parser.add_subparsers()
# Compute from a trained model
compute = sub_parser.add_parser('compute', help="Use this argument to segment tumorigenic areas with a trained AI model")
compute.add_argument('test_folder', type=str, help="Provide the directory storing the MRI(s)")
compute.add_argument('model_path', type=str, help="Provide the direcory storing the trained model")
compute.add_argument('to_save', type=str, help="Indicate the directory to save the predicted tumorigenic regions")
compute.add_argument('--model_name', type=str, default=None, help="Provide the name of the model file")
compute.add_argument('--mode', default='compute')
# Train the model
train = sub_parser.add_parser('train', help="Use this argument to train an AI model to detect tumorigenic areas in MRI(s)")
train.add_argument('--configuration', type=str, default="./", help="Provide the path of the 'config.yaml' file with the training specifications")
train.add_argument('--mode', default='train')
# Parse arguments
args = parser.parse_args()
mode = args.mode

if __name__ == '__main__':
    if mode=="train":
        # AI model
        model = UNet(n_channels=1, n_classes=1)

        # Training configuration
        config_file = os.path.join(args.configuration,"config.yaml")
        with open(config_file, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        # Launch training
        if config["n_folds"] is not None: # With cross validation
            train_model_kfold(
                data_path = config["data_path"], 
                model = model,
                device = torch.device(config["device"] if torch.cuda.is_available() else 'cpu'),
                epochs = int(config["epochs"]),
                batch_size = int(config["batch_size"]),
                learning_rate = float(config["learning_rate"]),
                save_folds = config["save"],
                amp = config["grad_scaling_amp"],
                weight_decay = float(config["weight_decay"]),
                gradient_clipping = float(config["gradient_clip"]),
                weights_loss=(config["alpha"],config["beta"],config["gamma"]),
                augment = config["augment"],
                n_folds = config["n_folds"],
                threshold = config["threshold"]
            )
        else: # Without cross validation
            train_model(
                data_path = config["data_path"], 
                model = model,
                device = torch.device(config["device"] if torch.cuda.is_available() else 'cpu'),
                epochs = int(config["epochs"]),
                batch_size = int(config["batch_size"]),
                learning_rate = float(config["learning_rate"]),
                save = config["save"],
                amp = config["grad_scaling_amp"],
                weight_decay = float(config["weight_decay"]),
                gradient_clipping = float(config["gradient_clip"]),
                weights_loss=(config["alpha"],config["beta"],config["gamma"]),
                augment = config["augment"],
                threshold = config["threshold"],
                n_train = float(config["n_train"]),
                val_freq = int(config["validation_frequency"])
        )
    elif mode=="compute":
        print('compute')
    else:
        raise ValueError("No mode provided through command line interface; try python tumorigenic.py --help")