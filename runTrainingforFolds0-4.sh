#!/bin/bash
#   ?? lr wieder auf Standard gesetzt im nnUNetTrainer.py ??
for i in {0..4}
do
  nnUNetv2_train 009 3d_fullres $i -tr nnUNetTrainer_100epochs --val_best --npz

  #nnUNetv2_train 004 2d $i -tr nnUNetTrainer_100epochs --npz
  nnUNetv2_train 009 2d $i -tr nnUNetTrainer_100epochs --val_best --npz

   #-pretrained_weights /home/daniel/ResearchData/Prostate/nnUnet/nnUNet_results/Dataset004_prostateCTV/nnUNetTrainer_100epochs__nnUNetPlans__2d/2023-09-13/fold_2/checkpoint_best.pth
done


#nnUNetv2_predict -i /home/daniel/ResearchData/Prostate/nnUnet/nnUNet_raw/Dataset004_prostateCTV/imagesTs/ -o /home/daniel/ResearchData/Prostate/nnUnet/nnUNet_results/Dataset004_prostateCTV/nnUNetTrainer_100epochs__nnUNetPlans__2d/fold_2/inferenceOutput/ -d 004 -c
#nnUNetv2_predict -i /home/daniel/ResearchData/Prostate/nnUnet/nnUNet_raw/Dataset004_prostateCTV/imagesTs/ \
 # -o /home/daniel/ResearchData/Prostate/nnUnet/nnUNet_results/Dataset004_prostateCTV/nnUNetTrainer_100epochs__nnUNetPlans__2d/fold_2/inferenceOutput/ \
  #-d 004  \
  #-tr nnUNetTrainer_100epochs \
  #-c 2d \
  #-p nnUNetPlans  \
  #-f 2  \
  #-chk /home/daniel/ResearchData/Prostate/nnUnet/nnUNet_results/Dataset004_prostateCTV/nnUNetTrainer_100epochs__nnUNetPlans__2d/fold_2/checkpoint_best.pth
