import nibabel as nib
import numpy as np
import sklearn.model_selection
import monai.transforms.spatial.functional as Fun
import glob
import torch
import sklearn

def augmentation(train_set):
    # train_set is a list:  train[0] --> [image, segmentation]
    # For now we rotate every pair of images only 1 time --> 50% more of data
    final_train = train_set
    for i in range(len(train_set)):
        k = np.random.randint(1,4)
        tr, sg = train_set[i][0], train_set[i][1]
        tr[0] = Fun.rotate90(tr[0], axes=(0,1), k=k, lazy=False, transform_info={})
        sg[0] = Fun.rotate90(sg[0], axes=(0,1), k=k, lazy=False, transform_info={})
        final_train.append([tr, sg]) 
    return final_train

def load_data(data_path, train_size=0.8, test_size=0.2):
    data_dir = data_path
    masks_b = sorted(glob.glob(data_dir+"_labels/*baseline*"))
    imgs_b = sorted(glob.glob(data_dir+"/*baseline*"))
    masks_15 = sorted(glob.glob(data_dir+"_labels/*15*"))
    imgs_15 = sorted(glob.glob(data_dir+"/*15*"))
    masks_24 = sorted(glob.glob(data_dir+"_labels/*24*"))
    imgs_24 = sorted(glob.glob(data_dir+"/*24*"))

    data_baseline, data_15, data_24 = [], [], []
    for i,j in zip(imgs_b, masks_b): 
        data_baseline.append([
            torch.tensor(np.expand_dims(nib.load(i).get_fdata(), axis=0)),
            torch.tensor(np.expand_dims(nib.load(j).get_fdata(), axis=0))
        ])
    for i,j in zip(imgs_15, masks_15): 
        data_15.append([
            torch.tensor(np.expand_dims(nib.load(i).get_fdata(), axis=0)),
            torch.tensor(np.expand_dims(nib.load(j).get_fdata(), axis=0))
        ])

    for i,j in zip(imgs_24, masks_24): 
        data_24.append([
            torch.tensor(np.expand_dims(nib.load(i).get_fdata(), axis=0)),
            torch.tensor(np.expand_dims(nib.load(j).get_fdata(), axis=0))
        ])
    data = data_15 + data_24
    train, val = sklearn.model_selection.train_test_split(data, train_size=train_size, test_size=test_size, shuffle=True)
    return train, val 

# train is a list:  train[0] --> [image, segmentation]


def load_data_no_split(data_path):
    data_dir = data_path 
    masks_b = sorted(glob.glob(data_dir+"_labels/*baseline*"))
    imgs_b = sorted(glob.glob(data_dir+"/*baseline*"))
    masks_15 = sorted(glob.glob(data_dir+"_labels/*15*"))
    imgs_15 = sorted(glob.glob(data_dir+"/*15*"))
    masks_24 = sorted(glob.glob(data_dir+"_labels/*24*"))
    imgs_24 = sorted(glob.glob(data_dir+"/*24*"))

    data_baseline, data_15, data_24 = [], [], []
    for i,j in zip(imgs_b, masks_b): 
        data_baseline.append([
            torch.tensor(np.expand_dims(nib.load(i).get_fdata(), axis=0)),
            torch.tensor(np.expand_dims(nib.load(j).get_fdata(), axis=0))
        ])
    for i,j in zip(imgs_15, masks_15): 
        data_15.append([
            torch.tensor(np.expand_dims(nib.load(i).get_fdata(), axis=0)),
            torch.tensor(np.expand_dims(nib.load(j).get_fdata(), axis=0))
        ])

    for i,j in zip(imgs_24, masks_24): 
        data_24.append([
            torch.tensor(np.expand_dims(nib.load(i).get_fdata(), axis=0)),
            torch.tensor(np.expand_dims(nib.load(j).get_fdata(), axis=0))
        ])
    data = data_15 + data_24
    return data