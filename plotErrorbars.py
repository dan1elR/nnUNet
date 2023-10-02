import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import ScaledTranslation
import matplotlib
from matplotlib.pyplot import figure
import pandas as pd
from matplotlib.transforms import Affine2D
import seaborn as sb
from datetime import date

today = date.today()
fig, ax = plt.subplots(figsize = (6.4,4.8), dpi = 200)
#markers = ['s', 'o', 'v', '<', '^', '>']
#colors = ['r', 'g', 'b', 'y']
pngname = f'{today}_nnUNet-valDices.png'

df = pd.read_excel('/home/daniel/Desktop/LRZ Sync+Share/ResearchDaniel/Prostata/TrainingResults/val_Dice-comparison.ods')
#df = df.truncate(after = 5)
x_axisname = 'Run'
xval = np.asarray(df['Unnamed: 0'])
y = np.asarray(df['val Dice mean'])
df['negFehler'] = df['val Dice mean'] - df['val Dice std']
df['posFehler'] = df['val Dice mean'] + df['val Dice std']
yerr= np.transpose(np.asarray(df[['val Dice std', 'val Dice std']]))
label ='valDice'

er1 = ax.errorbar(xval, y, yerr, linestyle="none",
                  label=label, capsize=5, marker = 'o')
#matplotlib.pyplot.errorbar(x = x, y= y,
 #                          yerr=np.transpose(np.asarray(df[['negativer Fehler', 'positiver Fehler']])),
  #                          fmt='None', color= 'b', label = 'Kappa')
ax.set_xlabel("nnUNet setting, Dataset")
ax.set_ylabel("")
#ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.legend(loc='center', bbox_to_anchor=(0.5, 1.08),
              ncol=2, fancybox=True)
ax.grid()
plt.savefig(pngname)
plt.show()