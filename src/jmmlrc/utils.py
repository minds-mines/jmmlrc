import numpy as np

class JointRC: 
    """A simple joint regression/classification object. This object helps us use utilities such as GridSearchCV"""
    def __init__(self, reg_part, class_part):
        self.r = reg_part
        self.c = class_part
        self.rc = np.concatenate((self.r, self.c), axis=1)
        self.shape = self.rc.shape
    
    def __getitem__(self, ix):
        return self.r[ix], self.c[ix]

class GroupedX:
    """A wrapper for X that explicitly groups data"""
    def __init__(self, data, groups):
        self.data = data
        self.groups = groups
        self.shape = self.data.shape

    def __getitem__(self, ix):
        return self.data[ix], self.groups
