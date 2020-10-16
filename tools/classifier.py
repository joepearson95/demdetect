import glob
import cv2
import os
import pandas as pd

filename = cv2.imread('#')
csv_file = pd.read_csv("#")

num = 0
total = 0

data = []
for row in csv_file.iterrows():
    if csv_file.iloc[num][1] == 1:
        data.append([csv_file.iloc[num][0], 'reading'])
    if csv_file.iloc[num][2] == 1:
        data.append([csv_file.iloc[num][0], 'drinking'])
    if csv_file.iloc[num][3] == 1:
        data.append([csv_file.iloc[num][0], 'eating'])
    if csv_file.iloc[num][4] == 1:
        data.append([csv_file.iloc[num][0], 'phoning'])
    num += 1

label_df = pd.DataFrame(data)
label_df.columns = ['filename', 'class']
label_df.to_csv('labels.csv')
