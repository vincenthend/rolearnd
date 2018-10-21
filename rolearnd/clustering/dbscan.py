from pandas import DataFrame
import numpy as np
import math
from classifier import Classifier
import pandas as pd
from sklearn import datasets

class DBSCAN(Classifier):
    '''
    DBSCAN class for clustering
    Link Types available = ["single", "complete", "average", "average-group"]
    '''

    def __init__(self, eps=0.5, min_pts=5, distance="euclidean", **kwargs):
        self.eps = eps
        self.min_pts = min_pts
        distance_type = ["euclidean", "manhattan"]
        if (distance in distance_type):
            self.distance = distance
        else:
            raise ValueError("Distance type %s not supported" % distance)

    def fit(self, X: DataFrame):
        # Get data's information
        self.__data = X.copy()
        n_row, n_col = X.shape
        self.__n_elmt = n_row

        # Create distance matrix
        self.__createDistanceMatrix(self.__data)

        self.labels = [-1 for i in range(self.__n_elmt)]
        for i in range (self.__n_elmt):
            # Count neighbors with distance <= eps, including itself
            n_neighbor = 1 # Count itself
            neighbor_labels = []
            for j in range (self.__n_elmt):
                if (i != j):
                    if (self.distance_matrix[i, j] <= self.eps):
                        n_neighbor += 1
                        # Check for density reachable data's label
                        if ((self.labels[j] != -1) and not(self.labels[j] in neighbor_labels)):
                            neighbor_labels.append(self.labels[j])

            # Data have at leas min_pts neighbors, data is core point
            if (n_neighbor >= self.min_pts):
                # Initiate label for current data
                if (neighbor_labels):
                    self.labels[i] = min(neighbor_labels)
                    # Change density reachable data's label to match this data's label
                    self.__replace_label(neighbor_labels, self.labels[i])
                else:
                    self.labels[i] = self.__generate_new_label()
                # Change neighbor's label
                for j in range(self.__n_elmt):
                    if (i != j):
                        if (self.distance_matrix[i, j] <= self.eps):
                            self.labels[j] = self.labels[i]

    def fit_predict(self, X: DataFrame):
        self.fit (X)
        return self.labels

    def __createDistanceMatrix(self, X: DataFrame):
        self.distance_matrix = np.ndarray(shape=(self.__n_elmt, self.__n_elmt))
        self.cluster_tree = [n for n in range(self.__n_elmt)]
        for index, row in X.iterrows():
            for index_2, row_2 in X.iterrows():
                if (index_2 == index):
                    distance = float('inf')
                else:
                    if (self.distance == "manhattan"):
                        distance = math.fabs(row_2.subtract(row, fill_value=0).abs().sum())
                    elif (self.distance == "euclidean"):
                        distance = (row_2.subtract(row, fill_value=0).pow(2).sum()) ** 0.5
                self.distance_matrix.itemset((index, index_2), distance)

    def __generate_new_label(self):
        label = 0
        while (label in self.labels):
            label += 1
        return label

    def __replace_label(self, old_labels, new_label):
        for i in range (self.__n_elmt):
            if (self.labels[i] in old_labels):
                self.labels[i] = new_label

    def count_purity(self):



'''
TEST DRIVE
def test():
    df = pd.DataFrame(datasets.load_iris().data)
    print ("Testing with euclidean distance...")
    print ("Data: ")
    print (df)
    clustering = DBSCAN(0.8, 2, "euclidean")
    prediction = clustering.fit_predict(df)
    print ("Distance Matrix: ")
    print (clustering.distance_matrix)
    print (prediction)

test()
'''