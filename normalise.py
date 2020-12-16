import pandas as pd
import numpy as np

class Normalise:
    """
    A class to normalise the skeleton data given. This is achieved by two algorithms for normalisation.
    One utilises kinematic features (HAL), whilst the other computes the features based on a distance vector
    normalised to a seperate distance vector.
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

    # A function to imitate the loop from hip joint
    # to connected points. Going down first,
    # before going to the head
    def start_with(self, arr, start_index):
        for i in range(start_index, len(arr)):
            yield arr[i]
        for i in range(start_index):
            yield arr[i]

    # The Hal paper for normalisation
    def hal(self):
        # Create a feature vector of points and find the hip joint
        feature_vector = []
        kf = []
        dataset_len = len(self.no_score)
        point_len = 17
        for x in range(dataset_len):
            current_arr = []
            for j in range(point_len):
                current_arr.append((
                    self.no_score["point_" + str(j) + "_x"].values[x],
                    self.no_score["point_" + str(j) + "_y"].values[x]
                ))
            feature_vector.append(current_arr)
        
        for x in range(dataset_len):
            hip_middle = [
                (feature_vector[x][11][0] + feature_vector[x][11][1]) / 2,
                (feature_vector[x][12][0] + feature_vector[x][12][1]) / 2,
            ]
            # Reset on each row iteration
            hip_sub = []
            hip_sub_complete = []

            # Based on bio-mechanics, the hip is subtracted from each point
            for y in range(point_len):
                hip_sub.append(np.subtract(feature_vector[x][y], hip_middle))
            hip_sub_complete.append(hip_sub)

            # Reorder the array so that it simulates moving to each point from
            # hip (root). Done this way due to being tabular
            reordered = []
            curr_pos = []
            for j in self.start_with(hip_sub, 11):
                reordered.append(j)
            
            # Loop the above array. When an 'out of bounds' error
            # happens, explicitly calculate the point before appending
            for idx in range(point_len):
                # Calculate the norm of position
                norm = np.linalg.norm(reordered[idx])
                pos_res = reordered[idx] / norm
                curr_pos.append(pos_res)

                # Calculate the velocity of position
                if idx + 1 <= len(reordered) - 1 and idx - 1 >= 0:
                    norm_add = np.linalg.norm(reordered[idx + 1])
                    norm_subtract = np.linalg.norm(reordered[idx - 1])
                    vel_res = (reordered[idx + 1] / norm_add) - (reordered[idx - 1] / norm_subtract)
                    curr_pos.append(vel_res)
                # Calculate the acceleration of position
                if idx + 2 <= len(reordered) - 1 and idx - 2 >= 0:
                    norm_add_acc = np.linalg.norm(reordered[idx + 2])
                    norm_subtract_acc =  np.linalg.norm(reordered[idx - 2])
                    acc_res = (reordered[idx + 2]/norm_add_acc) + (reordered[idx - 2]/norm_subtract_acc)-2*(reordered[idx]/norm)
                    curr_pos.append(acc_res)
                    
            kf.append(np.array(curr_pos).flatten())
        
        # Add the column names dynamically
        col_names = []
        for x in range(15):
            col_names.append("pos_point_" + str(x) + "_x")
            col_names.append("pos_point_" + str(x) + "_y")
            col_names.append("vel_point_" + str(x) + "_x")
            col_names.append("vel_point_" + str(x) + "_y")
            col_names.append("acc_point_" + str(x) + "_x")
            col_names.append("acc_point_" + str(x) + "_y")

        # Add the classes columns onto the dataframe and return
        df = pd.DataFrame(kf, columns=col_names)
        hal_con = pd.concat([df, self.classes], axis=1)
        return hal_con