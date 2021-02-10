import numpy as np
import pdb
import os

def hindawi1():
	dataset=np.load('dataset.npy')
	classes=np.load('labels.npy')
	normalized_data = []
	dataset_len = len(dataset)
	for x in range(dataset_len):
		neck_joint=[(dataset[x][5*2]+dataset[x][6*2])/2,(dataset[x][5*2+1]+dataset[x][6*2+1])/2]
		hip_middle =[(dataset[x][11*2]+dataset[x][12*2])/2,(dataset[x][11*2+1]+dataset[x][12*2+1])/2]
		torso_middle =[(neck_joint[0] + hip_middle[0]) / 2, (neck_joint[1] + hip_middle[1]) / 2]
		velocities = []
		accelerations = []
		for i in range(0,17):
			for j in range(0,10):
				n=np.linalg.norm(np.asarray((neck_joint[0][j], neck_joint[1][j])) - np.asarray((hip_middle[0][j], hip_middle[1][j])))
				dataset[x][2*i][j]=(dataset[x][2*i][j]-torso_middle[0][j])/n
				dataset[x][2*i+1][j]=(dataset[x][2*i+1][j]-torso_middle[1][j])/n
				positions=dataset[x]
				l=positions.shape[1]
			for i in range(0,10):
				if i-1>=0 and i+1<=l-1:
					velocity=(positions[:,i+1]-positions[:,i-1])/2
					velocities.append(velocity)
				else:
					continue
			for i in range(0,10):
				if i-2>=0 and i+2<=l-1:
					acceleration=(positions[:,i+2]+positions[:,i-2]-2*positions[:,i])/4
					accelerations.append(acceleration)
				else:
					continue
			
			positions=np.array(positions)
			# below is commented out, it threw an error stating np.array cannot use append. Quick hack to 'solve' it
			# velocities=np.asarray(velocities)
			# accelerations=np.array(accelerations)
			# velocities=np.transpose(velocities)
			# accelerations=np.transpose(accelerations)
			combined_feature=np.vstack((positions[:,2:l-2],np.asarray(np.transpose(velocities))[:,1:l-3]))
			# print(np.asarray(np.transpose(accelerations).shape))
			if np.asarray(np.transpose(accelerations)).shape == (34,6):
				combined_feature=np.vstack((combined_feature,np.asarray(np.transpose(accelerations))))
				# print(np.asarray(np.transpose(accelerations)).shape)
			else:			
				combined_feature=np.vstack((combined_feature,np.asarray(np.transpose(accelerations))[:,1:l-3]))
				# print(np.asarray(np.transpose(accelerations))[:,1:l-3].shape)
			# The below commented out code appeared to throw a mismatch size error when vstacking
			# combined_feature=np.concatenate((combined_feature, np.asarray(np.transpose)))
			# combined_feature=np.vstack((combined_feature,np.asarray(np.transpose(accelerations))[:,1:l-3]))
		normalized_data.append(combined_feature)
	return normalized_data

n_data=hindawi1()
if os.path.exists("normalized_data.py"):
	os.remove("normalized_data.py")

np.save('normalized_data.npy',n_data)
data = np.load('normalized_data.npy')
print(data.shape)
# pdb.set_trace()


# def hindawi():
# 	dataset=np.load('dataset.npy')
# 	classes=np.load('labels.npy')
# 	normalized_data = []
# 	dataset_len=len(dataset)
   
# 	for x in range(dataset_len):
# 		neck_joint=[(dataset[x][5*2]+dataset[x][6*2])/2,(dataset[x][5*2+1]+dataset[x][6*2+1])/2]
# 		hip_middle =[(dataset[x][11*2]+dataset[x][12*2])/2,(dataset[x][11*2+1]+dataset[x][12*2+1])/2]
# 		torso_middle =[(neck_joint[0] + hip_middle[0]) / 2, (neck_joint[1] + hip_middle[1]) / 2]
# 		test = []
# 		for i in range(0,17):
# 			for j in range(0,10):
# 				n=np.linalg.norm(np.asarray((neck_joint[0][j], neck_joint[1][j])) - np.asarray((hip_middle[0][j], hip_middle[1][j])))
# 				test.append(n)
# 				dataset[x][2*i][j]=(dataset[x][2*i][j]-torso_middle[0][j])/n
# 				dataset[x][2*i+1][j]=(dataset[x][2*i+1][j]-torso_middle[1][j])/n
# 		normalized_data.append(dataset[x])
     
# 	return normalized_data

# def hal():
# 	dataset=np.load('dataset.npy')
# 	classes=np.load('labels.npy')
# 	normalized_data = []
# 	dataset_len = len(dataset)
# 	for x in range(dataset_len):
# 		hip_middle =[(dataset[x][11*2]+dataset[x][12*2])/2,(dataset[x][11*2+1]+dataset[x][12*2+1])/2]
# 		kf = []
# 		for i in range(0,17):
# 			for j in range(0,10):
# 				dataset[x][2*i][j] = dataset[x][2*i][j]-hip_middle[0][j]
# 				dataset[x][2*i+1][j] = dataset[x][2*i+1][j]-hip_middle[1][j]
				
# 				part1 = dataset[x][11:]
# 				part2 = dataset[x][:11]
# 				reordered = np.concatenate((part1, part2))
# 				if (2*i+1)+1 <= 33 and (2*i+1)-1 >= 0 and (2*i+1)+2 <= 33 and (2*i+1)-2 >= 0:
# 					n = np.linalg.norm(reordered[2*i][j] - reordered[2*i+1][j])
# 					velocity = (
# 						np.linalg.norm(reordered[2*i+1][j] - reordered[(2*i+1)+1][j]) -
# 						np.linalg.norm(reordered[2*i-1][j] - reordered[(2*i+1)-1][j])
# 					)
# 					acceleration = (
# 						np.linalg.norm(reordered[2*i+2][j] - reordered[(2*i+1)+2][j]) +
# 						np.linalg.norm(reordered[2*i-2][j] - reordered[(2*i+1)-2][j]) - 2 * n
# 					)
# 					kf.append((n, velocity, acceleration))
# 		kf = np.reshape(np.asarray(kf), (3*15,10))
# 		normalized_data.append(kf)
# 	return normalized_data

# n_data=hal()
# if os.path.exists("normalized_data.py"):
# 	os.remove("normalized_data.py")

# np.save('normalized_data.npy',n_data)
# data = np.load('normalized_data.npy')
# print(data.shape)
# pdb.set_trace()


