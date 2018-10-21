from pandas import DataFrame, Series
import numpy as np
import math
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

    def fit(self, X : DataFrame):
        self.__data = X.copy()
        n_row, n_col = X.shape
        self.__n_elmt = n_row
        
        self.__createAdjMatrix(self.__data)
        for i in range(self.__n_elmt-1):
            # Select the smallest distance index
            min_idx = np.argmin(self.adj_matrix)
            min_row, min_col = self.__idxToRowCol(min_idx, n_row, n_row)
            print(self.adj_matrix)
            
            # Generate new distance matrix
            combined_node = self.__generateNode(min_row, min_col)
            self.__removeNode(min_row, min_col)
            self.__mergeNode(combined_node)            

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
        return new_row

    def __removeNode(self, a, b):
        self.adj_matrix = np.delete(self.adj_matrix,a,0)
        self.adj_matrix = np.delete(self.adj_matrix,a,1)
        if(a<b):
            self.adj_matrix = np.delete(self.adj_matrix,b-1,0)
            self.adj_matrix = np.delete(self.adj_matrix,b-1,1)
        else:
            self.adj_matrix = np.delete(self.adj_matrix,b,0)
            self.adj_matrix = np.delete(self.adj_matrix,b,1)

    def __mergeNode(self, node):
        n_row, n_col = self.adj_matrix.shape
        self.adj_matrix.resize((n_row+1, n_col+1))
        for i in range(n_row+1):
            self.adj_matrix[i,n_col] = node[i]
            self.adj_matrix[n_col,i] = node[i]

    def __createAdjMatrix(self, X : DataFrame):
        self.adj_matrix = np.ndarray(shape=(self.__n_elmt, self.__n_elmt))
        for index, row in X.iterrows():
            for index_2, row_2 in X.iterrows():
                if(index_2 == index):
                    distance = float('inf')
                else:
                    distance = math.fabs(row_2.subtract(row,fill_value=0).sum())
                self.adj_matrix.itemset((index, index_2), distance)

    def __idxToRowCol(self, idx, n_row, n_col):
        return (int(idx/n_row),idx%n_col)
'''
TEST DRIVE
'''
def test():
    d = DataFrame(data=[[9,5],[2,3],[3,4]])
    c = Agglomerative()
    c.fit(d)
    # print(c.adj_matrix)
    # print(np.argmin(c.adj_matrix))

test()
