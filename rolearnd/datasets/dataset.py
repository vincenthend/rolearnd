from pandas import DataFrame

class Dataset():
    def __init__(self, data : DataFrame, target : DataFrame, feature_names : list, target_name : list):
        self.data = data
        self.target = target
        self.feature_names = feature_names
        self.target_name = target_name