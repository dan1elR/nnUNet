import pandas as pd
import os.path
import glob
import SimpleITK as sitk
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import scipy.spatial


labelDir  = '/home/daniel/ResearchData/Prostate/nnUnet/nnUNet_raw/Dataset005_prostateCTV/labelsTr/'

def isotropic(CT):
    img = sitk.ReadImage(CT)

    rs2 = sitk.ResampleImageFilter()

    new_x = round(img.GetSize()[0] * img.GetSpacing()[0] / 1)
    new_y = round(img.GetSize()[1] * img.GetSpacing()[1] / 1)
    new_z = round(img.GetSize()[2] * img.GetSpacing()[2] / 1)

    rs2.SetOutputSpacing([1, 1, 1])
    rs2.SetSize([new_x, new_y, new_z])
    rs2.SetOutputDirection(img.GetDirection())
    rs2.SetOutputOrigin(img.GetOrigin())
    rs2.SetTransform(sitk.Transform())
    rs2.SetInterpolator(sitk.sitkNearestNeighbor)

    img3 = rs2.Execute(img)

    return img3
def get95HD(valDir, labelDir, id):
    labelmap1 = isotropic(os.path.join(valDir, id))
    labelmap2 = isotropic(os.path.join(labelDir, id))

    # Load the two NIfTI labelmaps
    #labelmap1 = sitk.ReadImage(os.path.join(valDir, id))
    #labelmap2 = sitk.ReadImage(os.path.join(labelDir, id))

    # Convert the SimpleITK images to NumPy arrays
    labelmap1_array = sitk.GetArrayFromImage(labelmap1)
    labelmap2_array = sitk.GetArrayFromImage(labelmap2)

    # Find the coordinates of non-zero pixels in labelmap1
    labelmap1_coords = np.argwhere(labelmap1_array > 0)

    # Find the coordinates of non-zero pixels in labelmap2
    labelmap2_coords = np.argwhere(labelmap2_array > 0)

    # Calculate the directed Hausdorff distance in both directions (labelmap1 to labelmap2 and vice versa)
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

    return [id, hausdorff_distance, hd95]

#%% do the work
# for fold in range(0,5):
#     valDir = f'/home/daniel/ResearchData/Prostate/nnUnet/nnUNet_results/Dataset005_prostateCTV/nnUNetTrainer_100epochs__nnUNetPlans__3d_fullres/fold_{fold}/validation/'
#     print('fold: ', fold)
#     files = glob.glob(valDir + '*.nii.gz')
#     IDS = []
#     for prediction in files:
#         id = os.path.basename(prediction)
#         IDS.append(id)
#
#     hDorfs = []
#     for id in IDS:
#         hausDorf = get95HD(valDir, labelDir, id)
#         hDorfs.append(hausDorf)
#
#     df = pd.DataFrame(hDorfs, columns = ['TUM-ID', 'Hausdorff', '95HD'])
#     df.set_index('TUM-ID', inplace=True)
#     mean = df.mean()
#     std = df.std()
#     df = df.T
#     df['Mean'] = mean
#     df['std'] = std
#     print(df.Mean)
#     df.to_excel(valDir + 'Hausdorff.xlsx')