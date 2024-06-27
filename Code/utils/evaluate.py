
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os, glob
import nibabel as nib
import numpy as np
import matplotlib.pylab as plt
import logging

@torch.inference_mode()
def compute_segmentation(data_path, model, results_path, device, threshold):
    logging.basicConfig(level=logging.INFO)
    logging.info(" Creating results directory")
    # Generate the directory to store the results
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # Find images
    logging.info(" Loading images")
    files = glob.glob(os.path.join(data_path,"*.nii.gz"))

    # Run inference
    logging.info(" Segmenting images")
    for i, f in enumerate(tqdm(files)):
        name = f.split("/")[-1].split(".")[0]
        # Load data
        img = nib.load(f)
        data, affine = img.get_fdata(), img.affine, 
        try:
            header = img.header
        except:
            header = None
        MRI = torch.tensor(
                np.expand_dims(
                    np.expand_dims(
                        data, axis=0 # [1,1,320,320,10]
                    ), axis=0
                )
            ).to(device=device, dtype=torch.float32)
        output = model(MRI)
        probs = F.sigmoid(output)
        mask = (probs > threshold).int()
        # Generate preliminary image
        fig, ax = plt.subplots(3,3,figsize=(15,15))
        ax[0,0].set_title("Image", fontsize=15, fontweight='bold')
        ax[0,1].set_title("Probability map", fontsize=15, fontweight='bold')
        ax[0,2].set_title("Mask", fontsize=15, fontweight='bold')
        for j in range(3):
            ax[j,0].imshow(MRI[0].squeeze().cpu()[:,:,3+j*2])
            ax[j,1].imshow(probs[0].squeeze().cpu()[:,:,3+j*2])
            ax[j,2].imshow(mask[0].squeeze().cpu()[:,:,3+j*2])
            ax[j,0].spines[["top","right","bottom","left"]].set_visible(False)
            ax[j,1].spines[["top","right","bottom","left"]].set_visible(False)
            ax[j,2].spines[["top","right","bottom","left"]].set_visible(False)
            ax[j,0].set_xticks([]), ax[j,0].set_yticks([])
            ax[j,1].set_xticks([]), ax[j,1].set_yticks([])
            ax[j,2].set_xticks([]), ax[j,2].set_yticks([])
        fig.tight_layout()
        plt.savefig(f"{results_path}/{name}.png")
        plt.close()
        # Save as nifti
        nib.save(nib.Nifti1Image(mask[0,0].cpu().numpy(), affine), f"{results_path}/{name}_segmentation.nii.gz")
        nib.save(nib.Nifti1Image(probs[0,0].cpu().numpy(), affine), f"{results_path}/{name}_probability.nii.gz")
    logging.info(" Done!")

@torch.inference_mode()
def evaluate(net, dataloader, device, amp, dice, epoch, fold=None, th=0.5):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    if fold is not None:
        save_path = f"./validation_results/fold_{fold}/"
    else:
        save_path = f"./validation_results/Validation-ep={epoch}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    affine = np.eye(4)

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for i, batch in enumerate(dataloader):
            image, mask_true = batch[0], batch[1]

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32)
            mask_true = mask_true.to(device=device, dtype=torch.long).int()

            # predict the mask
            mask_pred = net(image)

            assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
            mask_pred = (F.sigmoid(mask_pred) > th).int()
            # compute the Dice score
            dice_score += dice(mask_pred, mask_true)

            N_imgs = image.shape[0]
            for img in range(N_imgs):
                nib.save(nib.Nifti1Image(mask_pred[img].squeeze().cpu().numpy(), affine), f"{save_path}/mask-pred_i={i}_b={img}.nii.gz")
                nib.save(nib.Nifti1Image(mask_true[img].squeeze().cpu().numpy(), affine), f"{save_path}/mask-true_i={i}_b={img}.nii.gz")
                nib.save(nib.Nifti1Image(image[img].squeeze().cpu().numpy(), affine), f"{save_path}/image_i={i}_b={img}.nii.gz")
                fig, ax = plt.subplots(3,3,figsize=(15,15))
                ax[0,0].set_title("Image")
                ax[0,1].set_title("Predicted")
                ax[0,2].set_title("True")
                for j in range(3):
                    ax[j,0].imshow(image[img].squeeze().cpu()[:,:,3+j*2])
                    ax[j,1].imshow(mask_pred[img].squeeze().cpu()[:,:,3+j*2])
                    ax[j,2].imshow(mask_true[img].squeeze().cpu()[:,:,3+j*2])
                fig.tight_layout()
                plt.savefig(f"{save_path}/image_i={i}_b={img}.png")
                plt.close()

    net.train()
    return dice_score / max(num_val_batches, 1)
