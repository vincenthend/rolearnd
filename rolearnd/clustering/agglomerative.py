from pandas import DataFrame, Series
import numpy as np
import math
import collections
from classifier import Classifier

class Agglomerative(Classifier):
    '''
    Agglomerative Clustering class for clustering

    Link Types available = ["single", "complete", "average", "average-group"]

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
        
        self.__createAdjMatrix(self.__data)
        for i in range(self.__n_elmt-self.n_cluster):
            # Select the smallest distance index
            min_idx = np.argmin(self.adj_matrix)
            min_row, min_col = self.__idxToRowCol(min_idx, n_row, n_row)
            print(self.adj_matrix)
            
            # Generate new distance matrix
            combined_node = self.__generateNode(min_row, min_col)
            self.__removeNode(min_row, min_col)
            self.__mergeNode(combined_node)
            
        print(self.adj_matrix_tracker)

    def fit_predict(self):
        pass

    def __generateNode(self, a, b):
        if(self.link_type == "single"):
            row_a = self.adj_matrix[a,]
            row_b = self.adj_matrix[b,]
            new_row = np.minimum(row_a, row_b)
            new_row = np.delete(new_row, a)
            # Handle row shift
            if(a<b):
                new_row = np.delete(new_row, b-1)
            else:
                new_row = np.delete(new_row, b)
            new_row = np.append(new_row, float('inf'))
        elif(self.link_type == "complete"):
            row_a = self.adj_matrix[a,]
            row_b = self.adj_matrix[b,]
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
            cluster_el = [self.adj_matrix_tracker[a], self.adj_matrix_tracker[b]]
            opposite_cluster = []
            
            for idx in range(len(self.adj_matrix_tracker)):
                if(idx != a and idx != b):
                    opposite_cluster.append(self.adj_matrix_tracker[idx])
            
            cluster_el = self.__flatten(cluster_el)

            # Count distance between cluster and other cluster
            new_row = np.array([])
            for opposite_cluster_el in opposite_cluster:
                flat_op_elmt = self.__flatten(opposite_cluster_el)
                for op_el in flat_op_elmt:
                    sum_dist = 0
                    for el in cluster_el:
                        sum_dist += self.adj_matrix_orig[el,op_el]
                        
                new_row = np.append(new_row, [sum_dist/len(cluster_el)])
            new_row = np.append(new_row, float('inf'))

        elif(self.link_type == "average-group"):
            # Get list of element for cluster and non cluster
             # Get list of element for cluster and non cluster
            cluster_el = [self.adj_matrix_tracker[a], self.adj_matrix_tracker[b]]
            opposite_cluster = []
            
            for idx in range(len(self.adj_matrix_tracker)):
                if(idx != a and idx != b):
                    opposite_cluster.append(self.adj_matrix_tracker[idx])
            
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
        a_val = self.adj_matrix_tracker.pop(a)
        self.adj_matrix = np.delete(self.adj_matrix,a,0)
        self.adj_matrix = np.delete(self.adj_matrix,a,1)
        if(a<b):
            self.adj_matrix = np.delete(self.adj_matrix,b-1,0)
            self.adj_matrix = np.delete(self.adj_matrix,b-1,1)
            b_val = self.adj_matrix_tracker.pop(b-1)
        else:
            self.adj_matrix = np.delete(self.adj_matrix,b,0)
            self.adj_matrix = np.delete(self.adj_matrix,b,1)
            b_val = self.adj_matrix_tracker.pop(b)
        self.adj_matrix_tracker.append([a_val,b_val])

    def __mergeNode(self, node):
        n_row, n_col = self.adj_matrix.shape        
        
        b = np.zeros((n_row+1, n_col+1))
        b[:-1,:-1] = self.adj_matrix
        self.adj_matrix = b

        for i in range(n_row+1):
            self.adj_matrix[i,n_col] = node[i]
            self.adj_matrix[n_col,i] = node[i]

    def __createAdjMatrix(self, X : DataFrame):
        self.adj_matrix = np.ndarray(shape=(self.__n_elmt, self.__n_elmt))
        self.adj_matrix_tracker = [n for n in range(self.__n_elmt)]
        for index, row in X.iterrows():
            for index_2, row_2 in X.iterrows():
                if(index_2 == index):
                    distance = float('inf')
                else:
                    distance = math.fabs(row_2.subtract(row,fill_value=0).abs().sum())
                self.adj_matrix.itemset((index, index_2), distance)
        
        # Create copy for average - averagegroup
        self.adj_matrix_orig = self.adj_matrix.copy()

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
def test():
    d = DataFrame(data=[[9,5,3],[2,3,3],[3,4,3],[3,4,-2],[-4,-5,-5],[-2,-1,-2]])
    c = Agglomerative(n_cluster=3, link="average-group")
    c.fit(d)
    # print(c.adj_matrix)
    # print(np.argmin(c.adj_matrix))

test()
