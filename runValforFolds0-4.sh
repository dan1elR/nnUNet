#!/bin/bash
for i in {0..4}
do
  nnUNetv2_train 004 2d $i -tr nnUNetTrainer_100epochs --npz
done
