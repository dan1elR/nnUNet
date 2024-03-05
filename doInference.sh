#!/bin/bash
#   ?? lr wieder auf Standard gesetzt im nnUNetTrainer.py ??
dataset_id=012
datasetDescription='PERYTON100CTV'
results_root="/home/daniel/ResearchData/Prostate/nnUnet/nnUNet_results/Dataset${dataset_id}_${datasetDescription}/"
input_folder="/home/daniel/ResearchData/Prostate/nnUnet/nnUNet_raw/Dataset${dataset_id}_${datasetDescription}/imagesTS/CT/"
output_3d="${results_root}output3d_fullres/"
output_2d="${results_root}output2d/"
output_ensemble="${results_root}outputEnsemble/"
output_pp="${results_root}output_pp/"

#nnUNetv2_find_best_configuration ${dataset_id} -tr nnUNetTrainer_250epochs -c 3d_fullres 2d

#get predictions form each config: 2d and 3d_fullres for all folds
#nnUNetv2_predict -d Dataset${dataset_id}_${datasetDescription} -i ${input_folder} -o ${output_3d} -f  0 1 2 3 4 -tr nnUNetTrainer_250epochs -c 3d_fullres -p nnUNetPlans --save_probabilities
nnUNetv2_predict -d Dataset${dataset_id}_${datasetDescription} -i ${input_folder} -o ${output_2d} -f  0 1 2 3 4 -tr nnUNetTrainer_250epochs -c 2d -p nnUNetPlans --save_probabilities
#ensemble the predictions
nnUNetv2_ensemble -i ${output_3d} ${output_2d} -o ${output_ensemble} -np 8
#apply postprocessing on esembled predictions
nnUNetv2_apply_postprocessing -i ${output_ensemble} -o ${output_pp} -pp_pkl_file /home/daniel/ResearchData/Prostate/nnUnet/nnUNet_results/Dataset${dataset_id}_${datasetDescription}/ensembles/ensemble___nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres___nnUNetTrainer_250epochs__nnUNetPlans__2d___0_1_2_3_4/postprocessing.pkl -np 8 -plans_json /home/daniel/ResearchData/Prostate/nnUnet/nnUNet_results/Dataset${dataset_id}_${datasetDescription}/ensembles/ensemble___nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres___nnUNetTrainer_250epochs__nnUNetPlans__2d___0_1_2_3_4/plans.json

