#!/bin/bash
for i in {1..4}
do
  nnUNetv2_train 004 2d $i --val --npz --val_best -tr nnUNetTrainer_100epochs
done
