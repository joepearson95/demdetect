import pandas as pd
import numpy as np


class Normalise:
    """
    A class to normalise the skeleton data given.
    """

    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        del self.df['file_name']
        self.df = self.df.iloc[:, :-3]
        self.no_score = self.df[self.df.columns.drop(list(self.df.filter(regex='score')))]

    # The Hindawi papers algorithm for normalising data
    def hindawi(self):
        # Create feature vectors for each point in the dataset
        feature_vector = []
        posture_vector = []
        for i in range(17):
            feature_vector.append([
                self.no_score["point_" + str(i) + "_x"][0],
                self.no_score["point_" + str(i) + "_y"][0],
            ])
        # Extract the three joints below using the midpoint formula
        neck_joint = [
            (feature_vector[5][0] + feature_vector[5][1]) / 2,
            (feature_vector[6][0] + feature_vector[6][1]) / 2
        ]
        hip_middle = [
            (feature_vector[11][0] + feature_vector[11][1]) / 2,
            (feature_vector[12][0] + feature_vector[12][1]) / 2,
        ]
        torso_middle = [
            (neck_joint[0] + neck_joint[1]) / 2,
            (hip_middle[0] + hip_middle[1]) / 2,
        ]
        # Create the denominator by creating a distance vector and
        # calculating its norm
        distance_vector2 = [
            (neck_joint[0] + neck_joint[1]) / 2,
            (torso_middle[0] + torso_middle[1]) / 2
        ]
        norm = np.linalg.norm(distance_vector2)

        # Loop the feature vectors and normalise each point
        # returning a posture feature vector
        for x in feature_vector:
            distance_vector1 = [
                (x[0] + x[1]) / 2,
                (torso_middle[0] + torso_middle[1] / 2)
            ]
            posture_vector.append(distance_vector1 / norm)

        return posture_vector

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
                self.no_score["point_" + str(i) + "_x"][0],
                self.no_score["point_" + str(i) + "_y"][0],
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
        for val in self.start_with(hip_sub, 11):
            euc_norm.append(np.linalg.norm(val))
            vel_norm.append(np.linalg.norm(val + 1) - np.linalg.norm(val - 1))
            acc_norm.append(np.linalg.norm(val + 2) + np.linalg.norm(val - 2) - 2 * np.linalg.norm(val))
            num += 1

        # Create Kinematic Features vector based on joint position,
        # joint velocity and joint acceleration
        kf.append(euc_norm)
        kf.append(vel_norm)
        kf.append(acc_norm)
        return kf


file = '#'
normalise = Normalise(file)
normalise.hal()
