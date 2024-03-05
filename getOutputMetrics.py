import seg_metrics.seg_metrics as sg
import pandas as pd
import os

labels = [0,1]
gdth_path= r"/home/daniel/ResearchData/Prostate/nnUnet/nnUNet_raw/Dataset011_ESTRO100CTV/imagesTS/labels/"# ground truth image full path
pred_path = r"/home/daniel/ResearchData/Prostate/nnUnet/nnUNet_results/Dataset011_ESTRO100CTV/output_pp/" # prediction image full path
csv_file = os.path.join(pred_path, 'metrics-postProcessed.csv')

metrics = sg.write_metrics(labels=labels[1:],  # exclude background
                  gdth_path=gdth_path,
                  pred_path=pred_path,
                  csv_file=csv_file)


# Read the CSV file into a pandas DataFrame  # Replace with your CSV file path
df = pd.read_csv(csv_file)

# Export the DataFrame to an Excel file
excel_file_path = os.path.join(pred_path,'metrics-postProcessed.xlsx')  # Replace with your desired Excel file path
df.to_excel(excel_file_path, index=False)

print(f"CSV file successfully exported to Excel: {excel_file_path}")
