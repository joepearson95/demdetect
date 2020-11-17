import pandas as pd
import numpy as np


class Normalise:
    """
    A class to normalise the skeleton data given based on specific papers equations.
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


file = '#'
normalise = Normalise(file)
normalise.paper1()
