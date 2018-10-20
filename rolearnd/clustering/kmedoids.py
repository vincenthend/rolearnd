from classifier import Classifier
from pandas import DataFrame
import numpy as np
import math
import random

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

    Attributes
    ----------
    medoids : cluster centers
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
        pass
        # '''
        # '''
        # num_of_instances = X.shape[0]
        
        # if self.num_clusters > num_of_instances:
        #     raise ValueError(
        #         "num_of_instances = %d must be larger than num_clusters = %d" % (num_of_instances, self.num_clusters))
        # else:
        #     if (self.num_clusters == 1):
        #         self.labels = [0 for i in range(0, len(num_of_instances))]
        #         self.means = self.update_means(labels, X)
        #     else:
        #         init_type = type(self.init)
        #         if (init_type == str):
        #             if (self.init == 'random'):
        #                 means = [X.iloc[random.randrange(0,150)] for i in range(0, self.num_clusters)]
        #                 self.k_means(means, X)
        #             else:
        #                 raise ValueError("Init type 'random' or ndarray expected, found %s" % (self.init))
        #         else:
        #             if (len(self.init) == self.num_clusters):
        #                 self.k_means(self.init, X)
        #             else:
        #                 raise ValueError(
        #                     "number of initial means = %d must be equal to num_clusters = %d" % (len(self.init), self.num_clusters))

    def predict(self, X):
        pass
        # distances = self.count_distances(self.means, X)
        # return self.assign_labels(distances)

    def fit_predict(self, X):
        pass
        # y = 0 # dummy ... may be deleted soon (?)
        # self.fit(X, y)
        # return self.labels