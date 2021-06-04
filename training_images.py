# quick collect all relevant files
import os
import glob
import pandas as pd

# Quickly read the images that held correct data. Then copy them over to the right folder
df = pd.read_csv('/home/joe/Documents/tester.csv')
file_names = df.iloc[:,-1]

print(len(file_names))
counter = 0
for name in file_names:
    counter += 1
    # print(name[33:].replace("/","_")+".jpeg")
    print("%s/89849" % counter)
    nname = name+".jpeg"
    os.system("cp {0} /home/joe/Documents/trainingimages/{1}".format(nname, name[33:].replace("/","_")+".jpeg"))