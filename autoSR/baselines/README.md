This folder contains source code for the baselines adapted for use with the tf-ds datasets created in ../datasets/new-tf-datasets

Also, some deprecated functions have been removed.
Changes have been made so that all images are normalized (0,1) after input, (0,255) on output
PSNR & SSIM calculated directly from RGB

train.py can be used to train both baselines with their respective settings.