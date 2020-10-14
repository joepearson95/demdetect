# quick collect all relevant files
import os
import glob
import pandas as pd

# Quickly read the images that held correct data. Then copy them over to the right folder
df = pd.read_csv('#')
file_names = df.iloc[:,-1]

for name in file_names:
    os.system("cp %s training_images" % name)