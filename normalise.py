import pandas as pd


class Normalise:
    """
    A class to normalise the skeleton data given,
    depending on specific algorithm.
    """

    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        del self.df['file_name']
        self.df = self.df.iloc[:, :-3]
        self.no_score = self.df[self.df.columns.drop(list(self.df.filter(regex='score')))]

    def paper1(self):
        feature_vector = []
        for i in range(17):  # Use this when wanting to do the whole skeleton
            feature_vector.append([
                self.no_score["point_" + str(i) + "_x"][0],
                self.no_score["point_" + str(i) + "_y"][0],
            ])
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
        # The below are for inserting into original array
        # feature_vector.insert(6, neck_joint)
        # feature_vector.insert(12, hip_middle)
        # feature_vector.insert(8, torso_middle)

# print(self.df[self.df.columns['point_'+str(i)+'_y']][0])
# print(self.df[self.df.columns['point_'+str(i)+'_score']][0])
# col_names.append("point_" + str(x) + "_x")
#     col_names.append("point_" + str(x) + "_y")
#     col_names.append("point_" + str(x) + "_score")


file = '#'
normalise = Normalise(file)
normalise.paper1()
