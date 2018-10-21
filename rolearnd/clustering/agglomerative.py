from pandas import DataFrame, Series
import numpy as np
import math
import collections
from .classifier import Classifier

class Agglomerative(Classifier):
    '''
    Agglomerative Clustering class for clustering

    Parameters
    ----------
    link = ["single", "complete", "average", "average-group"]
        Link type for distance between cluster
    distance = ["manhattan", "euclidean"]
        Distance for distance matrix

    Examples
    --------
    
    Fit and predict in separate process
    >>> classifier = Agglomerative()
    >>> classifier.fit(X)
    >>> classifier.predict()

    Using fit_predict
    >>> class
    >>> classifier = Agglomerative()
    >>> classifier.fit_predict(X)
    
    Take note that X is a dataframe
    '''
    
    def __init__(self, link="single", n_cluster=2, distance="manhattan", **kwargs):
        link_type = ["single", "complete", "average", "average-group"]
        dist_type = ["euclidean", "manhattan"]
        if(link in link_type):
            self.link_type = link
        else:
            raise ValueError("Link type %s not supported" % link)
        if(distance in dist_type):
            self.distance = distance
        else:
            raise ValueError("Link type %s not supported" % link)
        if(n_cluster > 0):
            self.n_cluster = n_cluster
        else:
            raise ValueError("n_cluster must be > 0")

    def fit(self, X : DataFrame):
        self.__data = X.copy()
        n_row, n_col = X.shape
        self.__n_elmt = n_row
        
        self.__createDistMatrix(self.__data)
        for i in range(self.__n_elmt-self.n_cluster):
            # Select the smallest distance index
            min_idx = np.argmin(self.dist_matrix)
            min_row, min_col = self.__idxToRowCol(min_idx, n_row, n_row)
            
            # Generate new distance matrix
            combined_node = self.__generateNode(min_row, min_col)
            self.__removeNode(min_row, min_col)
            self.__mergeNode(combined_node)
            n_row, n_col = self.dist_matrix.shape

    def predict(self):       
        flatten_cluster = self.__flattenCluster()

        result_array = np.array([-1 for n in range(self.__n_elmt)])
        
        for cluster in range(len(flatten_cluster)):
            for i in flatten_cluster[cluster]:
                result_array[i] = cluster

        return result_array

    def fit_predict(self, X : DataFrame):
        self.fit(X)
        return self.predict() 

    def __flattenCluster(self):
        return [self.__flatten(f) for f in self.cluster_tree]

    def __generateNode(self, a, b):
        if(self.link_type == "single"):
            row_a = self.dist_matrix[a,]
            row_b = self.dist_matrix[b,]
            new_row = np.minimum(row_a, row_b)
            new_row = np.delete(new_row, a)
            # Handle row shift
            if(a<b):
                new_row = np.delete(new_row, b-1)
            else:
                new_row = np.delete(new_row, b)
            new_row = np.append(new_row, float('inf'))
        elif(self.link_type == "complete"):
            row_a = self.dist_matrix[a,]
            row_b = self.dist_matrix[b,]
            new_row = np.maximum(row_a, row_b)
            new_row = np.delete(new_row, a)
            # Handle row shift
            if(a<b):
                new_row = np.delete(new_row, b-1)
            else:
                new_row = np.delete(new_row, b)
            new_row = np.append(new_row, float('inf'))
        elif(self.link_type == "average"):
            # Get list of element for cluster and non cluster
            cluster_el = [self.cluster_tree[a], self.cluster_tree[b]]
            opposite_cluster = []
            
            for idx in range(len(self.cluster_tree)):
                if(idx != a and idx != b):
                    opposite_cluster.append(self.cluster_tree[idx])
            
            cluster_el = self.__flatten(cluster_el)

            # Count distance between cluster and other cluster
            new_row = np.array([])
            for opposite_cluster_el in opposite_cluster:
                flat_op_elmt = self.__flatten(opposite_cluster_el)
                for op_el in flat_op_elmt:
                    sum_dist = 0
                    for el in cluster_el:
                        sum_dist += self.dist_matrix_orig[el,op_el]
                        
                new_row = np.append(new_row, [sum_dist/len(cluster_el)])
            new_row = np.append(new_row, float('inf'))

        elif(self.link_type == "average-group"):
            # Get list of element for cluster and non cluster
            cluster_el = [self.cluster_tree[a], self.cluster_tree[b]]
            opposite_cluster = []
            
            for idx in range(len(self.cluster_tree)):
                if(idx != a and idx != b):
                    opposite_cluster.append(self.cluster_tree[idx])
            
            cluster_el = self.__flatten(cluster_el)
            # Get cluster center
            cluster_center = self.__data.iloc[cluster_el].mean()            

            new_row = np.array([])            
            # Count distance between cluster center and other cluster center
            for opposite_cluster_el in opposite_cluster:
                flat_op_elmt = self.__flatten(opposite_cluster_el)
                op_cluster_center = self.__data.iloc[flat_op_elmt].mean()
                dist = math.fabs(op_cluster_center.subtract(cluster_center,fill_value=0).abs().sum()) 
                new_row = np.append(new_row, dist)
            new_row = np.append(new_row, float('inf'))
        return new_row

    def __removeNode(self, a, b):
        a_val = self.cluster_tree.pop(a)
        self.dist_matrix = np.delete(self.dist_matrix,a,0)
        self.dist_matrix = np.delete(self.dist_matrix,a,1)
        if(a<b):
            self.dist_matrix = np.delete(self.dist_matrix,b-1,0)
            self.dist_matrix = np.delete(self.dist_matrix,b-1,1)
            b_val = self.cluster_tree.pop(b-1)
        else:
            self.dist_matrix = np.delete(self.dist_matrix,b,0)
            self.dist_matrix = np.delete(self.dist_matrix,b,1)
            b_val = self.cluster_tree.pop(b)
        self.cluster_tree.append([a_val,b_val])

    def __mergeNode(self, node):
        n_row, n_col = self.dist_matrix.shape        
        
        b = np.zeros((n_row+1, n_col+1))
        b[:-1,:-1] = self.dist_matrix
        self.dist_matrix = b

        for i in range(n_row+1):
            self.dist_matrix[i,n_col] = node[i]
            self.dist_matrix[n_col,i] = node[i]

    def __createDistMatrix(self, X : DataFrame):
        self.dist_matrix = np.ndarray(shape=(self.__n_elmt, self.__n_elmt))
        self.cluster_tree = [n for n in range(self.__n_elmt)]
        for index, row in X.iterrows():
            for index_2, row_2 in X.iterrows():
                if(index_2 == index):
                    distance = float('inf')
                else:
                    if(self.distance == "manhattan"):
                        distance = math.fabs(row_2.subtract(row,fill_value=0).abs().sum())
                    elif(self.distance=="euclidean"):
                        distance = (row_2.subtract(row,fill_value=0).pow(2).sum()) ** 0.5
                self.dist_matrix.itemset((index, index_2), distance)
        
        # Create copy for average - averagegroup
        self.dist_matrix_orig = self.dist_matrix.copy()

    def __idxToRowCol(self, idx, n_row, n_col):
        return (int(idx/n_row),idx%n_col)

    def __flatten(self, x):
        if isinstance(x, collections.Iterable):
            return [a for i in x for a in self.__flatten(i)]
        else:
            return [x]
'''
TEST DRIVE
'''
'''
from sklearn import datasets
def test():
    d = DataFrame(datasets.load_iris().data)
    c = Agglomerative(n_cluster=3, link="single", distance="euclidean")
    c.fit(d)
    print(c.dist_matrix)
    print(c.predict())
    print(c.cluster_tree)

test()
'''