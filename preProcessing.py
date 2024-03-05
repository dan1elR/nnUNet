import nibabel as nib
import SimpleITK as sitk
import os
import glob
import numpy
import  torchio as tio
import torch

keepLargest = tio.KeepLargestComponent(p=1, )
isotropResampling = tio.Resample(target=1)

#rootCT = '/home/daniel/ResearchData/Prostate/nnUnet/CT-images/'
rootLabel = '/home/daniel/ResearchData/Prostate/nnUnet/nnUNet_raw/Dataset010_PerytonCTV/labelsTr'
#ctOUT = '/home/daniel/ResearchData/Prostate/nnUnet/CT-isotropic/'
labelOut = '/home/daniel/ResearchData/Prostate/nnUnet/label-Isotropic-keepLargest/Peryton'

#CTs = glob.glob(rootCT + '/*.nii.gz')
labels = glob.glob(rootLabel + '/*nii.gz')

# for ct in CTs:
#     print('working on CT ', os.path.basename(ct))
#     image = tio.ScalarImage(ct)
#     iso = isotropResampling(image)
#     iso.save(os.path.join(ctOUT,os.path.basename(ct)))
# print('Done with CTs')

for label in labels:
    print('working on Label ', os.path.basename(label))
    image = tio.LabelMap(label)
    CTV = keepLargest(image)
    if not torch.equal(image.data, CTV.data):
        print('There was a second CTV in label ', os.path.basename(label))
    #iso = isotropResampling(CTV)
    CTV.save(os.path.join(labelOut,os.path.basename(label)))