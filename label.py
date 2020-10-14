import matplotlib.pyplot as plt
import glob
import cv2
import pandas as pd

# Get images
allImages = list()
names = list()
for filename in glob.iglob('*.jpeg', recursive=True):
    names.append(filename)
    allImages.append(cv2.imread(filename))

category=[]
plt.ion()

# Show the images ready to be labelled
for i,image in enumerate(allImages):
    # plt.imshow(image)
    # plt.pause(0.05)
    category.append((input('category: '), str(names[i])))

# Send this to a CSV file
df = pd.DataFrame(category, columns=['class', 'file'])
df.to_csv('labels.csv')