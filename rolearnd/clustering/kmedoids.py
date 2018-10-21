from .classifier import Classifier
from pandas import DataFrame
import numpy as np
import math
import random
import sys

# for test
from sklearn import datasets

class KMedoids(Classifier):
    '''
    Parameters
    ----------
    
    num_clusters : int, number of centroid/means/cluster
    init : {'random', 'np.ndarray or list(user defined)'}
           initial medoids
    max_iteration : maximum iteration limit
    swap_medoid : {'optimized', 'random'}
                  swap medoid method, optimized = search for the minimum possible swap, 
                  random = random swap

    Attributes
    ----------
    medoids : cluster centers
    labels : labels of each instance
    num_clusters : number of cluster
    init : initial means
    max_iteration : maximum iteration limit
    
    '''
    def __init__(self, num_clusters=3, init='random', max_iteration=300, swap_medoid='optimized', **kwargs):
        self.num_clusters = num_clusters
        self.init = init
        self.max_iteration = max_iteration
        self.swap_medoid = swap_medoid

    def fit(self, X : DataFrame):
        self.train_data = X
        num_of_instances = X.shape[0]
        
        if self.num_clusters > num_of_instances:
            raise ValueError(
                "num_of_instances = %d must be larger than num_clusters = %d" % (num_of_instances, self.num_clusters))
        else:
            if (self.num_clusters == 1):
                self.labels = [0 for i in range(0, num_of_instances)]
                # self.medoids = self.update_medoids(labels, X)
            else:
                init_type = type(self.init)
                if (init_type == str):
                    if (self.init == 'random'):
                        medoids = [random.randrange(0, num_of_instances) for i in range(0, self.num_clusters)]
                        self.k_medoids(medoids, X)
                    else:
                        raise ValueError("Init type 'random' or ndarray expected, found %s" % (self.init))
                else:
                    if (len(self.init) == self.num_clusters):
                        # init : list of index of selected initial medoids
                        self.k_medoids(self.init, X)
                    else:
                        raise ValueError(
                            "number of initial means = %d must be equal to num_clusters = %d" % (len(self.init), self.num_clusters))

    def predict(self, X):
        distances = self.count_distances(self.medoids, X)
        return self.assign_labels(distances)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels
    
    def k_medoids(self, initial_medoids, data):
        medoids = initial_medoids
        iteration = 0
        while (iteration < self.max_iteration):
            # print(medoids)
            distances = self.count_distances(medoids, data)
            labels = self.assign_labels(distances)
            error = self.calculate_error(distances)
            # print(error)
            
            new_distances = []
            new_medoids = []
            new_error = 0
            if (self.swap_medoid == 'optimized'):
                # daftar error tiap swap buat tiap instance [swap cluster, error]
                swap_errors = [[0, 0] for i in range(0, data.shape[0])]
                # swap buat tiap cluster i
                for i in range(0, self.num_clusters):
                    swap_candidate_idxs = []
                    for j in range(0, len(labels)):
                        if (labels[j] == i):
                            swap_candidate_idxs.append(j)
                    
                    for j in range(0, len(swap_candidate_idxs)):
                        new_medoids = []
                        new_medoids = [x for x in medoids]
                        new_medoids[i] = swap_candidate_idxs[j]
                        new_distances = self.count_distances(new_medoids, data)
                        new_error = self.calculate_error(new_distances)
                        swap_errors[swap_candidate_idxs[j]] = [i, new_error]
                
                min_error = sys.maxsize
                min_swap_idx = -1
                medoid_to_swap = -1
                for i in range(0, len(swap_errors)):
                    if (swap_errors[i][1] < min_error):
                        min_error = swap_errors[i][1]
                        min_swap_idx = i
                        medoid_to_swap = swap_errors[i][0]
                # print(error, min_error)
                if(min_error - error < 0):
                    medoids[medoid_to_swap] = min_swap_idx
                    # print(self.calculate_error(self.count_distances(medoids, data)))
                else:
                    break
            elif (self.swap_medoid == 'random'):
                temp_medoid_to_swap = random.randrange(0, self.num_clusters)
                swap_candidate_idxs = []
                for i in range(0, len(labels)):
                    if (labels[i] == temp_medoid_to_swap):
                        swap_candidate_idxs.append(i)
                new_medoids = medoids
                new_medoids[temp_medoid_to_swap] = random.choice(swap_candidate_idxs)
                new_distances = self.count_distances(new_medoids, data)
                new_error = self.calculate_error(new_distances)
                if (new_error - error < 0):
                    medoids = new_medoids
                else:
                    break
            else:
                raise ValueError(
                    "swap_medoid must be equal to 'optimized' or 'random', found %d" % (self.swap_medoid))

            iteration += 1
        
        self.labels = labels
        self.medoids = medoids

    def count_distances(self, medoids, data):
        num_of_instances = data.shape[0]
        distances = [[-1 for j in range(0, len(medoids))] for i in range(0, num_of_instances)]
        
        for instance_idx in range(0, num_of_instances):
            instance_data = [x for x in data.iloc[instance_idx]]
            for medoid_idx in range(0, len(medoids)):
                medoid_instance = [x for x in self.train_data.iloc[medoids[medoid_idx]]]
                distances[instance_idx][medoid_idx] = self.calculate_absolute_dist(medoid_instance, instance_data)
        
        return distances
    
    def calculate_absolute_dist(self, attribute1, attribute2):
        absolute_distance = 0

        if (len(attribute1) == len(attribute2)):
            for i in range(0, len(attribute1)):
                absolute_distance += abs(attribute1[i] - attribute2[i])
        else:
            raise ValueError(
                "number of attributes must be equal, attribute1 = %d, attribute2 = %d" % (len(attribute1), len(attribute2)))
        
        return absolute_distance
    
    def assign_labels(self, distances):
        labels = [-1 for i in range(0, len(distances))]
        for i in range(0, len(distances)):
            idx, val = self.min_val(distances[i])
            labels[i] = idx
        
        return labels
    
    def calculate_error(self, distances):
        total_error = 0
        for i in range(0, len(distances)):
            idx, val = self.min_val(distances[i])
            total_error += val
        
        return total_error

    def min_val(self, list_elements):
        index = -1
        value = sys.maxsize
        for i in range(0, len(list_elements)):
            if (list_elements[i] < value):
                index = i
                value = list_elements[i]

        return index, value
    
    def ndarray_to_dataframe(self, ndarray):
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

def test():
    y = 0
    iris = datasets.load_iris()
    kmedoids = KMedoids(num_clusters=3, init='random')
    # kmedoids = KMedoids(num_clusters=3, init='random', swap_medoid='random')
    test_data = np.array([[5.8,  3.1 ,  6.1,  0.4], [0.8,  1.1 ,  2.1,  0.1]])
    # test_data = [[5.8,  3.1 ,  6.1,  0.4], [0.8,  1.1 ,  2.1,  0.1]]
    # kmeans = KMeans(num_clusters=3, init=[[5.1, 3.5, 1.4, 0.2], [5.0, 2.0, 3.5, 1.0], [5.9, 3.0, 5.1, 1.8]])
    dataframe_iris = ndarray_to_dataframe(iris['data'])
    kmedoids.fit(dataframe_iris)
    print('finish fit')
    # print(kmedoids.labels)
    print(kmedoids.medoids)
    prediction = kmedoids.predict(ndarray_to_dataframe(test_data))
    print(prediction)

test()
'''