import os
import glob
import pandas as pd
import json
from get95HD import get95HD, labelDir, isotropic

DatasetID = '009'
resolution = '2d'
dir =   f'/home/daniel/ResearchData/Prostate/nnUnet/nnUNet_results/Dataset{DatasetID}_prostateCTV/nnUNetTrainer_100epochs__nnUNetPlans__{resolution}/'
labelDir = labelDir.replace('005', DatasetID)
columnNames = ['Dice','FN','FP','IoU','TN','TP','n_pred','n_ref', 'HD', 'HD95']#,
#               'Current_lr','train_loss','val_loss','Pseudo Dice','EMA pseudo Dice','epoch']
#df = pd.DataFrame({'Fold': firstColumn})

stats = []
for fold in [0,1,2,3,4]:
    foldDIR = os.path.join(dir, f'fold_{fold}')
    # log = glob.glob(foldDIR + '/training_log*')
    # logFile = []
    # with open(log[0],'r') as f:
    #     for line in f:
    #         logFile.append(line.rstrip('\n'))
    # bestEMAidx = max(i for i, s in enumerate(logFile) if 'Yayy' in s)
    # bestEMA = float(logFile[bestEMAidx].split(':')[-1])
    # bestEpoch = int(logFile[bestEMAidx - 6].split(' ')[-2])
    # lrAtBest = float(logFile[bestEMAidx - 5].split(' ')[-2])
    # trainLoss = float(logFile[bestEMAidx - 4].split(' ')[-2])
    # valLoss = float(logFile[bestEMAidx - 3].split(' ')[-2])
    # pseudoDice = float(logFile[bestEMAidx - 2].split('[')[-1].replace(']',''))
    valJson = glob.glob(foldDIR + '/*/*summary.json')[0]
    #jsonDF = pd.DataFrame(data['foreground_mean'], index=[0]).T

    json_data = []  # your list with json objects (dicts)
    with open(valJson) as json_file:
        json_data = json.load(json_file)
    valStats = list(json_data['mean']['1'].values())
    # valStats.extend([lrAtBest, trainLoss, valLoss, pseudoDice, bestEMA, bestEpoch])
    if not os.path.isfile(os.path.join(foldDIR, 'validation', 'Hausdorff.xlsx')):
        print('computing Hausdorffs')
        valDir = os.path.join(foldDIR, 'validation')
        print('fold: ', fold)
        files = glob.glob(valDir + '/*.nii.gz')
        IDS = []
        for file in files:
            id = os.path.basename(file)
            IDS.append(id)

        hDorfs = []
        for id in IDS:
            hausDorf = get95HD(valDir, labelDir, id)
            hDorfs.append(hausDorf)

        df = pd.DataFrame(hDorfs, columns=['TUM-ID', 'Hausdorff', '95HD'])
        df.set_index('TUM-ID', inplace=True)
        mean = df.mean()
        std = df.std()
        df = df.T
        df['Mean'] = mean
        df['std'] = std
        print(df.Mean)
        df.to_excel(os.path.join(valDir, 'Hausdorff.xlsx'))
    hausDF = pd.read_excel(os.path.join(foldDIR, 'validation', 'Hausdorff.xlsx'), index_col= 0)
    valStats.extend(list(hausDF.Mean.values))
    stats.append(valStats)

df = pd.DataFrame(stats, columns=columnNames)
mean = df.mean()
std = df.std()
df = df.T
df['Mean'] = mean
df['std'] = std
df.to_excel(dir +  'stats.xlsx')











    # Add new row to specifig index name
#    foldDF = jsonDF._append(pd.DataFrame([lrAtBest], index=['lr'], columns=jsonDF.columns))
#    foldDF = foldDF._append(pd.DataFrame([trainLoss], index=['train_loss'], columns=foldDF.columns))
#    foldDF = foldDF._append(pd.DataFrame([valLoss], index=['val_loss'], columns=foldDF.columns))
#    foldDF = foldDF._append(pd.DataFrame([pseudoDice], index=['Pseudo_Dice'], columns=foldDF.columns))
#    foldDF = foldDF._append(pd.DataFrame([bestEMA], index=['EMA_Dice'], columns=foldDF.columns))
#    foldDF = foldDF._append(pd.DataFrame([bestEpoch], index=['epoch'], columns=foldDF.columns))


