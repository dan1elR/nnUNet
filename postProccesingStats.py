import nibabel as nib
import os
import glob
import pandas as pd
import json
from get95HD import get95HD, isotropic
import SimpleITK as sitk
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import scipy.spatial


DatasetID = '009'
dir =   f'/home/daniel/ResearchData/Prostate/nnUnet/nnUNet_results/Dataset{DatasetID}_prostateCTV/output_pp/'
files = glob.glob(dir+'*.nii.gz')
#columnNames = ['Dice','FN','FP','IoU','TN','TP','n_pred','n_ref', 'HD', 'HD95']#,
#               'Current_lr','train_loss','val_loss','Pseudo Dice','EMA pseudo Dice','epoch']
labelDir = '/home/daniel/ResearchData/Prostate/nnUnet/nnUNet_raw/Dataset008_prostateCTV/labelsTr/'
hDorfs = []
h95Dorfs = []
Dices = []
IDs = []

# def isotropic3(CT):
#     img = sitk.ReadImage(CT)
#
#     rs2 = sitk.ResampleImageFilter()
#
#     new_x = round(img.GetSize()[0] * img.GetSpacing()[0] / 1)
#     new_y = round(img.GetSize()[1] * img.GetSpacing()[1] / 1)
#     new_z = round(img.GetSize()[2] * img.GetSpacing()[2] / 3)
#
#     rs2.SetOutputSpacing([1, 1, 3])
#     rs2.SetSize([new_x, new_y, new_z])
#     rs2.SetOutputDirection(img.GetDirection())
#     rs2.SetOutputOrigin(img.GetOrigin())
#     rs2.SetTransform(sitk.Transform())
#     rs2.SetInterpolator(sitk.sitkLinear)
#
#     img3 = rs2.Execute(img)
#
#     return img3
# def compute_dice_coefficient(ground_truth_path, prediction_path):
#     # Load the images using SimpleITK
#     ground_truth = sitk.ReadImage(ground_truth_path)
#     prediction = sitk.ReadImage(prediction_path)
#
#     if not ground_truth.GetSpacing() == prediction.GetSpacing():
#         prediction= isotropic3(prediction_path)
#     # Compute the Dice coefficient
#     dice_filter = sitk.LabelOverlapMeasuresImageFilter()
#     dice_filter.Execute(ground_truth, prediction)
#
#     dice_coefficient = dice_filter.GetDiceCoefficient()
#
#     return dice_coefficient


def compute_dice_score(label_map_gt, label_map_pred):
    # Load NIfTI label maps
    label_gt = sitk.GetArrayFromImage(label_map_gt)
    label_pred = sitk.GetArrayFromImage(label_map_pred)

    # Ensure both label maps are binary
    if np.max(label_gt) > 1 or np.max(label_pred) > 1:
        print("Label maps should be binary (contain only 0s and 1s).")
        np.place(label_pred, label_pred== 2, 0) # we are only interested in CTV = 1, set rectum=2 to background=0
    # Calculate True Positives, False Positives, and False Negatives
    true_positives = np.sum(label_gt * label_pred)
    false_positives = np.sum(label_pred) - true_positives
    false_negatives = np.sum(label_gt) - true_positives

    # Calculate Dice score
    dice_score = (2.0 * true_positives) / (2.0 * true_positives + false_positives + false_negatives)

    return dice_score

def get95HD(predDir, gtDir, id):
    predIsotropic_sitk = isotropic(predDir)
    gTIsotropic_sitk = isotropic(gtDir)

    # Load the two NIfTI labelmaps
    #predIsotropic_sitk = sitk.ReadImage(os.path.join(valDir, id))
    #gTIsotropic_sitk = sitk.ReadImage(os.path.join(labelDir, id))

    # Convert the SimpleITK images to NumPy arrays
    labelmap1_array = sitk.GetArrayFromImage(predIsotropic_sitk)
    labelmap2_array = sitk.GetArrayFromImage(gTIsotropic_sitk)

    # Find the coordinates of non-zero pixels in predIsotropic_sitk
    labelmap1_coords = np.argwhere(labelmap1_array == 1)

    # Find the coordinates of non-zero pixels in gTIsotropic_sitk
    labelmap2_coords = np.argwhere(labelmap2_array == 1)

    # Calculate the directed Hausdorff distance in both directions (predIsotropic_sitk to gTIsotropic_sitk and vice versa)
    hausdorff_distance1 = directed_hausdorff(labelmap1_coords, labelmap2_coords)[0]
    hausdorff_distance2 = directed_hausdorff(labelmap2_coords, labelmap1_coords)[0]

    # The Hausdorff distance is the maximum of the two distances
    hausdorff_distance = max(hausdorff_distance1, hausdorff_distance2)

    # Print the computed Hausdorff distance
    #print("Hausdorff Distance:", hausdorff_distance)

    def getDistancesFromAtoB(a, b):
        kdTree = scipy.spatial.KDTree(a, leafsize=100)
        return kdTree.query(b, k=1, eps=0, p=2)[0]

    # Compute distances from test to result and vice versa.
    dTestToResult = getDistancesFromAtoB(labelmap1_coords, labelmap2_coords)
    dResultToTest = getDistancesFromAtoB(labelmap1_coords, labelmap2_coords)
    hd95 = max(np.percentile(dTestToResult, 95), np.percentile(dResultToTest, 95))

    return [id, hausdorff_distance, hd95, gTIsotropic_sitk ,predIsotropic_sitk]


for prediction in files:
    tumID = os.path.basename(prediction).replace('.nii.gz', '')
    print(f'working on {tumID}')
    gTlabel = os.path.join(labelDir, os.path.basename(prediction))

    id, hDorf, hd95, gTIso_sitk, predIso_sitk = get95HD(prediction, gTlabel, tumID)
    dice = compute_dice_score(gTIso_sitk, predIso_sitk)
    Dices.append(dice)
    IDs.append(tumID)
    hDorfs.append(hDorf)
    h95Dorfs.append(hd95)

Values = [IDs, hDorfs, h95Dorfs, Dices ]
df = pd.DataFrame(Values).transpose()
df.columns = ['TUMid', 'hausDorfs', 'HausPercents', 'Dice']
df.to_excel(f'{dir}stats.xlsx')











