import SimpleITK as sitk
import glob
import os

CTs = '/home/daniel/ResearchData/Prostate/nnUnet/nnUNet_raw/Dataset007_prostateCTV/labelsTr/'
isotropic3CTs = '/home/daniel/ResearchData/Prostate/nnUnet/nnUNet_raw/Dataset008_prostateCTV/labelsTr/'
files = glob.glob(CTs + '*.nii.gz')
def isotropic3z(CT):
    img = sitk.ReadImage(CT)

    rs2 = sitk.ResampleImageFilter()

    new_x = round(img.GetSize()[0] * img.GetSpacing()[0] / 1)
    new_y = round(img.GetSize()[1] * img.GetSpacing()[1] / 1)
    new_z = round(img.GetSize()[2] * img.GetSpacing()[2] / 3)

    rs2.SetOutputSpacing([1, 1, 3])
    rs2.SetSize([new_x, new_y, new_z])
    rs2.SetOutputDirection(img.GetDirection())
    rs2.SetOutputOrigin(img.GetOrigin())
    rs2.SetTransform(sitk.Transform())
    rs2.SetInterpolator(sitk.sitkLinear)

    img3 = rs2.Execute(img)

    return img3


for file in files:
    isoIMG =isotropic3z(file)
    sitk.WriteImage(isoIMG, os.path.join(isotropic3CTs, os.path.basename(file)))