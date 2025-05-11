import os
import re
import numpy as np
import pandas as pd

pattern = r'test of best model'

import argparse

parser = argparse.ArgumentParser(description='Summarize test results')
parser.add_argument('--dataset', type=str, required=True, help='Dataset name to process')
args = parser.parse_args()
dataset_name = args.dataset


columns = ['Folder', 'Loss_mean', 'Loss_std', 'MIoU_mean', 'MIoU_std', 
           'F1_or_DSC_mean', 'F1_or_DSC_std', 'Accuracy_mean', 'Accuracy_std',
           'Specificity_mean', 'Specificity_std', 'Sensitivity_mean', 'Sensitivity_std',
           'Confusion_Matrix']
results_df = pd.DataFrame(columns=columns)

results_dir = './results'
for folder in os.listdir(results_dir):
    folder_path = os.path.join(results_dir, folder)
    if os.path.isdir(folder_path) and dataset_name in folder:
        print(f'Processing folder: {folder_path}')

        losses, mious, f1s, accuracies, specificities, sensitivities = [], [], [], [], [], []
        confusion_matrices = []
        

        for run in range(1, 6):
            log_path = os.path.join(folder_path, 'log', f'fold_{run}')
            if not os.path.exists(log_path):
                log_path = os.path.join(folder_path, 'log', f'run_{run}')
            if os.path.exists(log_path):
                print(f'Processing log folder: {log_path}')
                for file in os.listdir(log_path):
                    if file.endswith('.log'):
                        file_path = os.path.join(log_path, file)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            print(f'Processing {file_path}')
                            lines = f.readlines()
                            for i in range(len(lines)-1, -1, -1):
                                line = lines[i]
                                if re.search(pattern, line):
                                    try:
                                        next_line = lines[i+1] if i+1 < len(lines) else ''
                                        full_line = line.strip() + ' ' + next_line.strip()
                                        data = {}
                                        parts = full_line.split(',')
                                        for part in parts:
                                            key_value = part.strip().split(':')
                                            if len(key_value) == 2:
                                                key = key_value[0].strip()
                                                value = key_value[1].strip()
                                                data[key] = value
                                        losses.append(float(data['loss']))
                                        mious.append(float(data['miou']) * 100)
                                        f1s.append(float(data['f1_or_dsc']) * 100)
                                        accuracies.append(float(data['accuracy']) * 100)
                                        specificities.append(float(data['specificity']) * 100)
                                        sensitivities.append(float(data['sensitivity']) * 100)
                                        cm = data['confusion_matrix'].replace('\n', ' ').strip()
                                        confusion_matrices.append(cm)
                                        break
                                    except Exception as e:
                                        print(f'Error parsing line: {line.strip()}')
                                        print(f'Error: {str(e)}')
                                else:
                                    print(f'No match found in line: {line.strip()}')
        
        if losses:
            row = {
                'Folder': folder,
                'Loss_mean': np.mean(losses),
                'Loss_std': np.std(losses),
                'MIoU_mean': np.mean(mious),
                'MIoU_std': np.std(mious),
                'F1_or_DSC_mean': np.mean(f1s),
                'F1_or_DSC_std': np.std(f1s),
                'Accuracy_mean': np.mean(accuracies),
                'Accuracy_std': np.std(accuracies),
                'Specificity_mean': np.mean(specificities),
                'Specificity_std': np.std(specificities),
                'Sensitivity_mean': np.mean(sensitivities),
                'Sensitivity_std': np.std(sensitivities),
                'Confusion_Matrix': confusion_matrices
            }
            results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)

output_path = os.path.join(results_dir, args.dataset+'_results.xlsx')
results_df.to_excel(output_path, index=False)
