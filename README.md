# Automatic delineation of tumorigenic areas
The following instructions are for UBUNTU-based systems. Start by creating a python environment and install the dependencies. Recomended is to use the following: 

```shell
user@user:~/current_directory$ cd astrachallege/Code
user@user:~/astrachallnege/Code$ conda create -n env-name python=3.10
user@user:~/astrachallnege/Code$ conda activate env-name
user@user:~/astrachallnege/Code$ pip install -r requirements.txt --no-cache
```

### To  display needed arguments the main arguments
```shell
user@user:~/astrachallnege/Code$ python tumorigenesis.py --help
Automatic detection of tumorigenic ares
 [-h] {compute,train} ...

positional arguments:
  {compute,train}
    compute        Use this argument to segment tumorigenic areas with a trained AI model
    train          Use this argument to train an AI model to detect tumorigenic areas in MRI(s)

options:
  -h, --help       show this help message and exit
```

## Training mode
In a bash terminal, in the correct directory type the following to get information about the training mode:
```shell
user@user:~/astrachallnege/Code$ python tumorigenesis.py train --help
usage: Automatic detection of tumorigenic areas train [-h] [--configuration CONFIGURATION] [--mode MODE]

options:
  -h, --help            show this help message and exit
  --configuration CONFIGURATION
                        Provide the path of the 'config.yaml' file with the training specifications
```
### To train a model from scratch
In the configuration file, you will find information about the parameters to train a new model from scratch. Have in mind that is better to have access to GPUs, otherwise the training will take significantly longer to converge. If the *config.yaml* file is located in the same directory as the rest of the code, to train a new model it is enough to type the following command in the terminal. Note that by doing this, you will use the same arguments as the submitted model.
```shell
user@user:~/astrachallnege/Code$ python tumorigenesis.py train
```

## Computing mode	
In a bash terminal, in the correct directory type the following to get information about the computing mode:
```shell
user@user:~/astrachallnege/Code$ python tumorigenesis.py compute --help
usage: Automatic detection of tumorigenic ares compute [-h] [--threshold THRESHOLD] [--device {cpu,cuda}] [--mode MODE] test_folder model_path model_name to_save

positional arguments:
  test_folder           Provide the directory storing the MRI(s)
  model_path            Provide the direcory storing the trained model
  model_name            Provide aproxy for the name of the model file
  to_save               Indicate the directory to save the predicted tumorigenic regions

options:
  -h, --help            show this help message and exit
  --threshold THRESHOLD
                        Indicate the threshold to binarize the probability maps
  --device {cpu,cuda}   Run inference on CPU or GPU
```
### To compute segmentation masks using a trained model	
```shell
python tumorigenesis.py compute testing-data-directory trained-model-location UNET results-directory
```
