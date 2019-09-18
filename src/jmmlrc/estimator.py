import numpy as np

from sklearn.svm import LinearSVC
from sklearn.metrics import hinge_loss

from jmmlrc.norms import l21_norm
from jmmlrc.norms import group_norm
from jmmlrc.norms import trace_norm

from jmmlrc.utils import GroupedX
from jmmlrc.utils import JointRC

class JMMLRC:
    def __init__(self, gamma1, gamma2, gamma3, score_func=None):
        """Initialize the JointLongitudinalRegressionClassifcation object
    
        Initialize the provided tuning parameters. These will deterimine 
        the impact of various regularizations
        
        Args:        
            gamma1: l_{2,1}-norm
            gamma2: l_1 group-norm
            gamma3: trace-norm
        """
        self.g1 = gamma1
        self.g2 = gamma2
        self.g3 = gamma3

        self.score_func = score_func
        
    def fit(self, X, Y, max_iter=20, verbose=False):
        """Learns a joint regression and classification model
        
        Args:
            X: longitudinal input training data, contains muliple modalities
                X.shape = (num_patients, features, time)

            Yr: target longitudinal regression values (cognitive scores)
                Yr.shape = (num_patients, regression_values, time)

            Yc: target longitudinal classification values (diagnoses)
                Yc.shape = (num_patients, class_labels, time)
        """
        self.X, self.groups, self.Yr, self.Yc = self._read_data(X, Y)

        NUM_CLASSES = len(np.unique(self.Yc))

        self.Wr = [np.ones([self.X[0].shape[1], self.Yr[0].shape[1]]) for i in range(0, len(self.X))]
        self.Wc = [np.ones([self.X[0].shape[1], NUM_CLASSES]) for i in range(0, len(self.X))]
        self.rho = [np.ones([NUM_CLASSES]) for i in range(0, len(self.X))]

        for i in range(0, max_iter):
            if verbose: print("cost: " + str(self.get_cost(self.X, self.Yr, self.Yc)))

            W = [np.append(r, c, axis=1) for r, c in zip(self.Wr, self.Wc)]
            D1, D2, D3 = self.get_Ds(W)

            self.update_Wr(D1, D2, D3)
            self.update_Wc(D1, D2, D3)

    def _read_data(self, X, Y):
        """Reads in X and Y Correctly."""
        if type(X) is tuple:
            data = X[0]
            groups = X[1]
        else:
            assert type(X) is GroupedX 
            data = X.data
            groups = X.groups

        if type(Y) is tuple:
            Yr = Y[0]
            Yc = Y[1]
        else:
            assert type(Y) is JointRC
            Yr = Y.r
            Yc = Y.c

        _X = [np.squeeze(x) for x in np.split(data, data.shape[2], axis=2)]
        _Yr = [np.squeeze(y) for y in np.split(Yr, Yr.shape[2], axis=2)]
        _Yc = [np.squeeze(y) for y in np.split(Yc, Yc.shape[2], axis=2)]

        return _X, groups, _Yr, _Yc 

    def predict(self, _X):
        """Predict the regression value and class values for the given data

        Args:
            X: longitudinal input data
        """
        if type(_X) is tuple:
            data = _X[0]
            groups = _X[1]
        else:
            assert type(_X) is GroupedX
            data = _X.data
            groups = _X.groups

        assert groups == self.groups

        X = [np.squeeze(x) for x in np.split(data, data.shape[2], axis=2)]
        reg_predict = [x @ wr for x, wr in zip(X, self.Wr)]
        raw_class_predict = [x @ wc + rho for x, wc, rho in zip(X, self.Wc, self.rho)]
        class_predict = [np.expand_dims(np.argmax(raw, axis=1), axis=1) for raw in raw_class_predict]
        return {"regression" : np.stack(reg_predict, axis=2), "classification" : np.stack(class_predict, axis=2)}

    def get_Ds(self, W):
        D1 = self.g1 * self.D1(W)
        D2 = self.g2 * self.D2(W)
        D3 = self.g3 * self.D3(W)

        return D1, D2, D3

    def update_Wc(self, D1, D2, D3):
        """Updates the classifcation weight matrix based on the following:

        D1 - Derivative of L21 Norm
        D2 - Derivative of Group Norm
        D3 - Derivative of Trace Norm
        """
        D_vals = np.power(np.diag(D1 + D2 + D3), -1.0/2.0)
        D = np.diag(D_vals)

        X_tilde = [np.matmul(x, D) for x in self.X]
         
        svms = [LinearSVC().fit(x, y) for x, y in zip(X_tilde, self.Yc)]
        
        self.Wc = [D @ svm.coef_.T for svm in svms]
        self.rho = [svm.intercept_ for svm in svms]
        
    def update_Wr(self, D1, D2, D3):
        """Updates the regression weight matrix based on the following:

        W = 2 * (X * X.T + self.g1 * D1 + self.g2 * D2 + self.g3 * D3) ^ (-1) * (X * Y)

        D1 - Derivative of L21 Norm
        D2 - Derivative of Group Norm
        D3 - Derivative of Trace Norm
        """
        sums = [x.T @ x + D1 + D2 + D3 for x in self.X]
        new_Ws = [np.linalg.inv(s) @ x.T @ y for s, x, y in zip(sums, self.X, self.Yr)]

        self.Wr = new_Ws

    def get_cost(self, X, Yr, Yc):
        """Calculates the current cost associated with the objective function"""
        r_cost = self.reg_cost(X, Yr)
        c_cost = self.class_cost(X, Yc)

        W = [np.append(r, c, axis=1) for r, c in zip(self.Wr, self.Wc)]
        unfold_W = np.concatenate(W, axis=0)

        l21_cost = l21_norm(unfold_W)
        group_cost = group_norm(unfold_W, self.groups)
        trace_cost = trace_norm(unfold_W)
        
        return r_cost + c_cost + self.g1 * l21_cost + self.g2 * group_cost + self.g3 * trace_cost

    def score(self, X, Y):
        X, groups, Yr, Yc = self._read_data(X, Y)
        
        if self.score_func == "mean_squared_error":
            loss = self.reg_cost(X, Yr)
        elif self.score_func == "hinge":
            loss = self.class_cost(X, Yc)
        else:
            raise ValueError("The score function has not been defined. \
                              In order to use this function pass score_func=\"hinge\" \
                              or score_func=\"mean_squared_error\" to the constructor.")

        return 1.0 / loss

    def class_cost(self, X, Yc):
        """Calculates the current cost of classification"""
        predictions = [x @ wc + rho for x, wc, rho in zip(X, self.Wc, self.rho)] 

        h_loss = sum([hinge_loss(y, pred) for y, pred in zip(Yc, predictions)])

        return h_loss

    def reg_cost(self, X, Yr):
        """Calculates the current cost of regression"""
        regression = sum([np.linalg.norm(x @ wr - yr) for x, wr, yr in zip(X, self.Wr, Yr)])

        return regression

    def D1(self, W):
        """Calculates the L21-norm of the unfolded matrix W"""
        unfold_W = np.concatenate(W, axis=1)
        D1_norm = np.sum(np.square(unfold_W), axis=1)
        D1 = np.diag(1.0 / (2.0 * np.sqrt(D1_norm)))
        return D1

    def D2(self, W):
        """Calculate the L1-group-norm of the unfolded matrix W"""
        unfold_W = np.concatenate(W, axis=1)
        D2_groups = np.split(unfold_W, self.groups)
        D2_groups = [x for x in D2_groups if x.shape[0] > 0]
        D2_group_norms = [np.sqrt(np.sum(np.square(x))) for x in D2_groups]
        D2_expand = [np.full(x.shape[0], val) for x, val in zip(D2_groups, D2_group_norms)]
        D2 = np.diag(np.concatenate(D2_expand))
        return D2

    def D3(self, W):
        """Calculates the trace-norm of the unfolded matrix W"""
        unfold_W = np.concatenate(W, axis=1)
        mul = np.matmul(unfold_W, unfold_W.T)
        eps = np.full(mul.shape, .000001)
        D3_vals = np.power(np.diag(mul + eps), -1.0/2.0) / 2.0
        D3 = np.diag(D3_vals)
        return D3

    def get_params(self, deep=True):
        """Gets the tuning parameters associated with this JLRC implementation"""
        out = dict()
        out['gamma1'] = self.g1
        out['gamma2'] = self.g2
        out['gamma3'] = self.g3

        out['score_func'] = self.score_func
        return out

    def set_params(self, **params):
        """Sets the tuing parameters associated with this JLRC implementation"""
        self.g1 = params['gamma1']
        self.g2 = params['gamma2']
        self.g3 = params['gamma3']

        self.score_func = params['score_func']
        return self
