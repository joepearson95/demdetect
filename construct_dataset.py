import pandas as pd
import pdb
import numpy as np

csv_file='demcare1_ingest_dataset.csv'
df = pd.read_csv(csv_file)

name=df['file_name']
classes =df[df.columns[-3:]]
df =df.iloc[:, :-3]
no_score = df[df.columns.drop(list(df.filter(regex='score')))]
del no_score['file_name']
l=len(name)
dataset=[]
length=10
labels=[]
for i in range(0,l):
	sample=[]
	if l-i-1-9>=0:
		sample1=no_score.iloc[l-i-1]
		label1=classes.iloc[l-i-1]
		name1=name[l-i-1]
		x = name1.split("/")
		x1=int(x[-1][-8:-5])
		sample.append(sample1)
		for i1 in range(1,length):
			name_t=name[l-i-1-i1]
			label1_t=classes.iloc[l-i-1-i1]
			x_t = name_t.split("/")
			x1_t=int(x_t[-1][-8:-5])
			if (x1_t-x1)!=1 or label1_t[0]!=label1[0] or label1_t[1]!=label1[1] or label1_t[2]!=label1[2]:
				sample=[]
				break
			
			sample1_t=no_score.iloc[l-i-1-i1,:]
			sample.append(sample1_t)
			x1=x1_t
			label1=label1_t
		if sample!=[]:
			sample=np.array(sample)
			sample=sample.reshape((34,10))
			dataset.append(sample)
			label=classes.iloc[l-i-1]
			label=np.array(label)
			labels.append(label)
np.save('dataset.npy',dataset)
np.save('labels.npy',labels)
pdb.set_trace()
			
			


