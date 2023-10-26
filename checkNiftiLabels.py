import numpy as np
import nibabel as nib
import glob
import os

root = '/home/daniel/ResearchData/Prostate/nnUnet/nnUNet_raw/Dataset006_prostateCTV/labelsTr/'
files = glob.glob(os.path.join(root, '*.nii.gz'))

for file in files:
    label = nib.load(file)
    data = label.get_fdata()
    intArr = data.round().astype(int)
    newFile = nib.Nifti1Image(intArr, label.affine, label.header)
    nib.save(newFile, file)
