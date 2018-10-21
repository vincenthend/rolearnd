from classifier import Classifier
from pandas import DataFrame
import numpy as np
import math
import random
import sys

# for test
from sklearn import datasets

class KMeans(Classifier):
    '''
    Parameters
    ----------
    
    num_clusters : int, number of centroid/means/cluster
    init : {'random', 'np.ndarray or list(user defined)'}
           initial centroids
    max_iteration : maximum iteration limit

    Attributes
    ----------
    means : cluster centers/centroids
    labels : labels of each instance
    num_clusters : number of cluster
    init : initial means
    max_iteration : maximum iteration limit
    
    '''
    def __init__(self, num_clusters=3, init='random', max_iteration=300,  **kwargs):
        self.num_clusters = num_clusters
        self.init = init
        self.max_iteration = max_iteration

    def fit(self, X : DataFrame):
        '''
        '''
        num_of_instances = X.shape[0]
        
        if self.num_clusters > num_of_instances:
            raise ValueError(
                "num_of_instances = %d must be larger than num_clusters = %d" % (num_of_instances, self.num_clusters))
        else:
            if (self.num_clusters == 1):
                self.labels = [0 for i in range(0, num_of_instances)]
                self.means = self.update_means(labels, X)
            else:
                init_type = type(self.init)
                if (init_type == str):
                    if (self.init == 'random'):
                        means = [X.iloc[random.randrange(0, num_of_instances)] for i in range(0, self.num_clusters)]
                        self.k_means(means, X)
                    else:
                        raise ValueError("Init type 'random' or ndarray expected, found %s" % (self.init))
                else:
                    if (len(self.init) == self.num_clusters):
                        self.k_means(self.init, X)
                    else:
                        raise ValueError(
                            "number of initial means = %d must be equal to num_clusters = %d" % (len(self.init), self.num_clusters))

    def predict(self, X):
        distances = self.count_distances(self.means, X)
        return self.assign_labels(distances)

    def fit_predict(self, X):
        y = 0 # dummy ... may be deleted soon (?)
        self.fit(X, y)
        return self.labels
    
    def k_means(self, initial_means, data):
        means = initial_means
        prev_means = [[0 for j in range(0, len(means[0]))] for i in range(0, len(means))]
        iteration = 0
        while ((not self.is_means_equal(means, prev_means)) and (iteration < self.max_iteration)):
            distances = self.count_distances(means, data)
            # print(means)
            labels = self.assign_labels(distances)
            # print(labels)
            prev_means = means
            means = self.update_means(labels, data)
            iteration += 1
        
        self.labels = labels
        self.means = means

    def count_distances(self, means, data):
        num_of_instances = data.shape[0]
        distances = [[-1 for j in range(0, len(means))] for i in range(0, num_of_instances)]
        #distances = [[-1 for j in range(0, num_of_instances)] for i in range(0, len(means))]
        
        for instance_idx in range(0, num_of_instances):
            instance_data = [x for x in data.iloc[instance_idx]]
            # print(instance_data)
            for means_idx in range(0, len(means)):
                distances[instance_idx][means_idx] = self.calculate_euclidean_dist(means[means_idx], instance_data)
        
        return distances

    def calculate_euclidean_dist(self, attribute1, attribute2):
        squared_distance = 0

        if (len(attribute1) == len(attribute2)):
            for i in range(0, len(attribute1)):
                squared_distance += pow((attribute1[i] - attribute2[i]), 2)
        else:
            raise ValueError(
                "number of attributes must be equal, attribute1 = %d, attribute2 = %d" % (len(attribute1), len(attribute2)))
        
        return math.sqrt(squared_distance)
    
    def is_means_equal(self, means, prev_means):
        is_equal = True
        num_of_attributes = len(means[0])

        for i in range(0, len(means)):
            for j in range(0, num_of_attributes):
                if(not (means[i][j] == prev_means[i][j])):
                    is_equal = False
                    break
            if (not is_equal):
                break
        
        return is_equal
    
    def assign_labels(self, distances):
        labels = [-1 for i in range(0, len(distances))]
        for i in range(0, len(distances)):
            idx, val = self.min_val(distances[i])
            labels[i] = idx
        
        return labels
        
    def min_val(self, list_elements):
        index = -1
        value = sys.maxsize
        for i in range(0, len(list_elements)):
            if (list_elements[i] < value):
                index = i
                value = list_elements[i]

        return index, value
    
    def update_means(self, labels, data):
        means = [[0 for j in range(0, data.shape[1])] for i in range(0, self.num_clusters)]
        sums = [[0 for j in range(0, data.shape[1])] for i in range(0, self.num_clusters)]
        n_cluster_elmt = [0 for i in range(0, self.num_clusters)]

        for j in range(0, data.shape[1]): # access by column
            for i in range(0, len(labels)):
                for k in range(0, self.num_clusters):
                    if (k == labels[i]):
                        sums[k][j] += data[j][i]
                        if (j == 0):
                            n_cluster_elmt[k] += 1
        
        for cluster_idx in range(0, self.num_clusters):
            for attr_idx in range(0, data.shape[1]):
                means[cluster_idx][attr_idx] = sums[cluster_idx][attr_idx] / n_cluster_elmt[cluster_idx]

        return means
    
def ndarray_to_dataframe(ndarray):
    if (type(ndarray) == list):
        data_list = ndarray
    elif (type(ndarray) == np.ndarray):
        data_list = ndarray.tolist()
    else:
        raise TypeError("ndarray type = 'list' or 'numpy.ndarray' expected, found %s" % type(ndarray))
    col_data = [[ndarray[j][i] for j in range(0, len(ndarray))] for i in range(0, len(ndarray[0]))]
    dataframe = DataFrame({i:col_data[i] for i in range(0, len(ndarray[0]))})
    return dataframe

'''
TEST DRIVE
'''
def test():
    y = 0
    iris = datasets.load_iris()
    kmeans = KMeans(num_clusters=3, init='random')
    test_data = np.array([[5.8,  3.1 ,  6.1,  0.4], [0.8,  1.1 ,  2.1,  0.1]])
    # test_data = [[5.8,  3.1 ,  6.1,  0.4], [0.8,  1.1 ,  2.1,  0.1]]
    # kmeans = KMeans(num_clusters=3, init=[[5.1, 3.5, 1.4, 0.2], [5.0, 2.0, 3.5, 1.0], [5.9, 3.0, 5.1, 1.8]])
    dataframe_iris = ndarray_to_dataframe(iris['data'])
    kmeans.fit(dataframe_iris)

    prediction = kmeans.predict(ndarray_to_dataframe(test_data))
    print(prediction)
    print(kmeans.means)

test()
#'''