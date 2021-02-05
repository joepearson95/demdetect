import numpy as np
import pdb
import os

def hindawi():
	dataset=np.load('dataset.npy')
	classes=np.load('labels.npy')
	normalized_data = []
	dataset_len=len(dataset)
   
	for x in range(dataset_len):
		neck_joint=[(dataset[x][5*2]+dataset[x][6*2])/2,(dataset[x][5*2+1]+dataset[x][6*2+1])/2]
		hip_middle =[(dataset[x][11*2]+dataset[x][12*2])/2,(dataset[x][11*2+1]+dataset[x][12*2+1])/2]
		torso_middle =[(neck_joint[0] + hip_middle[0]) / 2, (neck_joint[1] + hip_middle[1]) / 2]
		test = []
		for i in range(0,17):
			for j in range(0,10):
				n=np.linalg.norm(np.asarray((neck_joint[0][j], neck_joint[1][j])) - np.asarray((hip_middle[0][j], hip_middle[1][j])))
				test.append(n)
				dataset[x][2*i][j]=(dataset[x][2*i][j]-torso_middle[0][j])/n
				dataset[x][2*i+1][j]=(dataset[x][2*i+1][j]-torso_middle[1][j])/n
		normalized_data.append(dataset[x])
     
	return normalized_data

def hal():
	dataset=np.load('dataset.npy')
	classes=np.load('labels.npy')
	normalized_data = []
	dataset_len = len(dataset)
	for x in range(dataset_len):
		hip_middle =[(dataset[x][11*2]+dataset[x][12*2])/2,(dataset[x][11*2+1]+dataset[x][12*2+1])/2]
		kf = []
		for i in range(0,17):
			for j in range(0,10):
				dataset[x][2*i][j] = dataset[x][2*i][j]-hip_middle[0][j]
				dataset[x][2*i+1][j] = dataset[x][2*i+1][j]-hip_middle[1][j]
				
				part1 = dataset[x][11:]
				part2 = dataset[x][:11]
				reordered = np.concatenate((part1, part2))
				if (2*i+1)+1 <= 33 and (2*i+1)-1 >= 0 and (2*i+1)+2 <= 33 and (2*i+1)-2 >= 0:
					n = np.linalg.norm(reordered[2*i][j] - reordered[2*i+1][j])
					velocity = (
						np.linalg.norm(reordered[2*i+1][j] - reordered[(2*i+1)+1][j]) -
						np.linalg.norm(reordered[2*i-1][j] - reordered[(2*i+1)-1][j])
					)
					acceleration = (
						np.linalg.norm(reordered[2*i+2][j] - reordered[(2*i+1)+2][j]) +
						np.linalg.norm(reordered[2*i-2][j] - reordered[(2*i+1)-2][j]) - 2 * n
					)
					kf.append(n)
					kf.append(velocity)
					kf.append(acceleration)
		kf = np.reshape(np.asarray(kf), (45,10))
		normalized_data.append(kf)
	return normalized_data

n_data=hal()
if os.path.exists("normalized_data.py"):
	os.remove("normalized_data.py")

np.save('normalized_data.npy',n_data)
# data = np.load('normalized_data.npy')
# pdb.set_trace()


