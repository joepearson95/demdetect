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

    # Loop a vector starting with specific index
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
        for i in range(17):
            feature_vector.append([
                self.no_score["point_" + str(i) + "_x"],
                self.no_score["point_" + str(i) + "_y"],
            ])
        hip_middle = [
            (feature_vector[11][0] + feature_vector[11][1]) / 2,
            (feature_vector[12][0] + feature_vector[12][1]) / 2,
        ]
        # Based on bio-mechanics, the hip is subtracted from each point
        for x in feature_vector:
            hip_sub.append(np.subtract(x, hip_middle))

        # Starting with a section of the hip joint and moving to the others,
        # find the normalised euclidean distance. Adaptinge each subsequent
        # vector for its specific usage
        num = 1
        euc_norm = []
        vel_norm = []
        acc_norm = []
        kf = []
        for val in self.start_with(feature_vector, 11):
            euc_norm.append(np.linalg.norm(val))
            vel_norm.append(np.linalg.norm(val + 1) - np.linalg.norm(val - 1))
            acc_norm.append(np.linalg.norm(val + 2) + np.linalg.norm(val - 2) - 2 * np.linalg.norm(val))
            num += 1

        # Create Kinematic Features vector based on joint position,
        # joint velocity and joint acceleration
        kf.append(euc_norm)
        kf.append(vel_norm)
        kf.append(acc_norm)

        col_names = []
        for x in range(17):
            col_names.append("point_" + str(x) + "_x")
            col_names.append("point_" + str(x) + "_y")

        # euc_df = pd.DataFrame(euc_norm)
        df = pd.DataFrame(euc_norm)
        hindawiDF = pd.DataFrame(df.T.values)
        # hind_con = pd.concat([hindawiDF, self.classes], axis=1)
        print(euc_norm)


file = '#'
normalise = Normalise(file)
normalise.hindawi()
