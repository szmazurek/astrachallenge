
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import nibabel as nib
import numpy as np
import matplotlib.pylab as plt

from utils.dice_score import multiclass_dice_coeff, dice_coeff


@torch.inference_mode()
def evaluate(net, dataloader, device, amp, dice, epoch):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    if not os.path.exists(f"./Validation-ep={epoch}"):
        os.mkdir(f"./Validation-ep={epoch}")
    affine = np.eye(4)

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for i, batch in enumerate(tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False)):
            image, mask_true = batch[0], batch[1]

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32)
            mask_true = mask_true.to(device=device, dtype=torch.long).int()

            # predict the mask
            mask_pred = net(image)

            assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
            mask_pred = (F.sigmoid(mask_pred) > 0.5).int()
            # compute the Dice score
            dice_score += dice(mask_pred, mask_true)
            #dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)

            N_imgs = image.shape[0]
            for img in range(N_imgs):
                nib.save(nib.Nifti1Image(mask_pred[img].squeeze().cpu().numpy(), affine), f"./Validation-ep={epoch}/mask-pred_i={i}_b={img}.nii.gz")
                nib.save(nib.Nifti1Image(mask_true[img].squeeze().cpu().numpy(), affine), f"./Validation-ep={epoch}/mask-true_i={i}_b={img}.nii.gz")
                nib.save(nib.Nifti1Image(image[img].squeeze().cpu().numpy(), affine), f"./Validation-ep={epoch}/image_i={i}_b={img}.nii.gz")
                fig, ax = plt.subplots(3,3,figsize=(15,15))
                ax[0,0].set_title("Image")
                ax[0,1].set_title("Predicted")
                ax[0,2].set_title("True")
                for j in range(3):
                    ax[j,0].imshow(image[img].squeeze().cpu()[:,:,3+j*2])
                    ax[j,1].imshow(mask_pred[img].squeeze().cpu()[:,:,3+j*2])
                    ax[j,2].imshow(mask_true[img].squeeze().cpu()[:,:,3+j*2])
                fig.tight_layout()
                plt.savefig(f"./Validation-ep={epoch}/image_i={i}_b={img}.png")
                plt.close()

    net.train()
    return dice_score / max(num_val_batches, 1)