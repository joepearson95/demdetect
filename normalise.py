import pandas as pd
import numpy as np

class Normalise:
    """
    A class to normalise the skeleton data given.
    """

    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        del self.df['file_name']
        self.classes = self.df[self.df.columns[-3:]]
        self.df = self.df.iloc[:, :-3]
        self.no_score = self.df[self.df.columns.drop(
            list(self.df.filter(regex='score')))]

    # The Hindawi papers algorithm for normalising data
    def hindawi(self):
        # Create feature vectors for each point in the dataset
        feature_vector = []
        posture_vector = []
        dataset_len = len(self.no_score)
        point_len = 17
        for i in range(dataset_len):
            current_arr = []
            for j in range(point_len):
                current_arr.append((
                    self.no_score["point_" + str(j) + "_x"].values[i],
                    self.no_score["point_" + str(j) + "_y"].values[i]
                ))
            feature_vector.append(current_arr)
        # Loop the dataset, calculating the three missing joints.
        # Next, the two distance vectors for this iteration,
        # before normalising the result and appending to []
        for x in range(dataset_len):
            neck_joint = [
                (feature_vector[x][5][0] + feature_vector[x][6][0]) / 2,
                (feature_vector[x][5][1] + feature_vector[x][6][1]) / 2
            ]
            hip_middle = [
                (feature_vector[x][11][0] + feature_vector[x][11][1]) / 2,
                (feature_vector[x][12][0] + feature_vector[x][12][1]) / 2,
            ]
            torso_middle = [
                (neck_joint[0] + neck_joint[1]) / 2,
                (hip_middle[0] + hip_middle[1]) / 2,
            ]
            curr_pos = []
            for y in range(point_len):
                dv1 = np.linalg.norm(np.asarray((feature_vector[x][y][0], feature_vector[x][y][1])) - np.asarray((torso_middle[0], torso_middle[1])))
                dv2 = np.linalg.norm(np.asarray((neck_joint[0], neck_joint[1])) - np.asarray((torso_middle[0], torso_middle[1])))
                curr_pos.append(dv1 / dv2)
            posture_vector.append(curr_pos)
        # Create the dataframe to pass to model
        col_names = []
        for x in range(17):
            col_names.append("point_" + str(x))
        hindawiDF = pd.DataFrame(posture_vector, columns=col_names)
        hind_con = pd.concat([hindawiDF, self.classes], axis=1)
        return hind_con

    def start_with(self, arr, start_index):
        for i in range(start_index, len(arr)):
            yield arr[i]
        for i in range(start_index):
            yield arr[i]

    # The Hal paper for normalisation
    def hal(self):
        # Create a feature vector of points and find the hip joint
        feature_vector = []
        hip_sub = []
        dataset_len = 2#len(self.no_score)
        point_len = 17
        for i in range(dataset_len):
            current_arr = []
            for j in range(point_len):
                current_arr.append((
                    self.no_score["point_" + str(j) + "_x"].values[i],
                    self.no_score["point_" + str(j) + "_y"].values[i]
                ))
            feature_vector.append(current_arr)
        for x in range(dataset_len):
            hip_middle = [
                (feature_vector[x][11][0] + feature_vector[x][11][1]) / 2,
                (feature_vector[x][12][0] + feature_vector[x][12][1]) / 2,
            ]
            # Based on bio-mechanics, the hip is subtracted from each point
            for y in range(point_len):
                hip_sub.append(np.subtract(feature_vector[x][y], hip_middle))
            # Starting with a section of the hip joint and moving to the others,
            # find the normalised euclidean distance. Adaptinge each subsequent
            # vector for its specific usage

            reordered = []
            norm_pos = []
            velocity = []
            acceleration = []
            # loop the array, on out of bounds, specfically ask for the array item
            for j in self.start_with(hip_sub, 11):
                reordered.append(j)
            
            # Due to out of bounds index, these are done before the iterator
            velocity.append((reordered[1]/np.linalg.norm(reordered[1]) - (reordered[16]/np.linalg.norm(reordered[16]))))
            acceleration.append((reordered[2]/np.linalg.norm(reordered[2])) + (reordered[15]/np.linalg.norm(reordered[15]))-2*(reordered[0]/np.linalg.norm(reordered[0])))
            acceleration.append((reordered[3]/np.linalg.norm(reordered[3])) + (reordered[16]/np.linalg.norm(reordered[16]))-2*(reordered[1]/np.linalg.norm(reordered[1])))
            for idx, val in enumerate(reordered):
                norm = np.linalg.norm(reordered[idx])
                norm_pos.append(reordered[idx]/norm)
                if idx + 1 <= len(reordered) - 1 and idx - 1 >= 0:
                    norm_add = np.linalg.norm(reordered[idx + 1])
                    norm_subtract = np.linalg.norm(reordered[idx - 1])
                    velocity.append((reordered[idx + 1]/norm_add) - (reordered[idx - 1]/norm_subtract))
                if idx + 2 <= len(reordered) - 1 and idx - 2 >= 0:
                    norm_add_acc = np.linalg.norm(reordered[idx + 2])
                    norm_subtract_acc =  np.linalg.norm(reordered[idx - 2])
                    acceleration.append((reordered[idx + 2]/norm_add_acc) + (reordered[idx - 2]/norm_subtract_acc)-2*(reordered[idx]/norm))
            velocity.append((reordered[0]/np.linalg.norm(reordered[0])) - (reordered[15]/np.linalg.norm(reordered[15])))
            acceleration.append((reordered[0]/np.linalg.norm(reordered[0])) + (reordered[13]/np.linalg.norm(reordered[13]))-2*(reordered[15]/np.linalg.norm(reordered[15])))
            acceleration.append((reordered[1]/np.linalg.norm(reordered[1])) + (reordered[14]/np.linalg.norm(reordered[14]))-2*(reordered[16]/np.linalg.norm(reordered[16])))

            # Create Kinematic Features vector based on joint position,
            # joint velocity and joint acceleration
        col_names = []
        for x in range(17):
            col_names.append("point_" + str(x))
            # col_names.append("point_" + str(x) + "_y")

        kf = []
        kf.append(norm_pos)
        kf.append(velocity)
        kf.append(acceleration)
        
        # df = np.array(kf).flatten()
        print(norm_pos) #(pd.DataFrame(df).transpose())
        # print(pd.DataFrame(kf, columns=col_names))

        # # euc_df = pd.DataFrame(euc_norm)
        # df = pd.DataFrame(euc_norm)
        # hindawiDF = pd.DataFrame(df.T.values)
        # # hind_con = pd.concat([hindawiDF, self.classes], axis=1)
        # print(euc_norm)


file = '#'
normalise.hal()
