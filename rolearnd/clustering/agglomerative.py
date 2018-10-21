from pandas import DataFrame, Series
import numpy as np
from rolearnd.clustering.classifier import Classifier

class Agglomerative(Classifier):
    '''
    Agglomerative Clustering class for clustering

    Link Types available = ["single", "complete", "average", "average-group"]

    '''
    
    def __init__(self, link="single", distance="manhattan" **kwargs):
        link_type = ["single", "complete", "average", "average-group"]
        dist_type = ["euclidean", "manhattan"]
        if(link in link_type):
            self.link = link
        else:
            raise ValueError("Link type %s not supported" % link)
        if(link in link_type):
            self.distance = distance
        else:
            raise ValueError("Link type %s not supported" % link)

    def fit(self, X : DataFrame):
        self.__data = X.copy()
        self.__createAdjMatrix(X)
        pass

    def predict(self, X):
        pass

    def fit_predict(self):
        pass

    # Create adjacency with manhattan distance
    def __createAdjMatrix(self, X : DataFrame):
        n_row, n_col = X.shape
        self.adj_matrix = np.ndarray(shape=n_row, d_type=float)
        for index, row in X.iterrows():
            for index_2, row_2 in X.iterrows():
                distance = row_2.sub(row, fill_values=0).sum()
                self.adj_matrix.itemset((index, index_2),distance)

'''
TEST DRIVE
'''
def test():
    d = DataFrame(data=[[1,2],[2,3],[3,4]])
    c = Agglomerative()
    c.fit(d)

test()
