#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np

from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from jmmlrc.utils import GroupedX
from jmmlrc.utils import JointRC

from jmmlrc.estimator import JMMLRC

__author__ = "Lou Brand"
__copyright__ = "Lou Brand"
__license__ = "mit"

def generate_dummy_data():
    """Generates a random datset to be used in the JMMLRC model

    Dataset Description
        X:       100 patients, 10 features (three groups), 4 time points
        Y_reg:   100 patients, 3 cognitive scores, 4 time points
        Y_class: 100 patients, 1 diagnosis, 4 time points
    """

    X_data = np.random.rand(100, 10, 4)
    X_groups = [1, 5, 9]
    Y_reg = np.random.rand(100, 3, 4)
    Y_class = np.random.randint(3, size=(100, 1, 4))
    
    X = GroupedX(X_data, X_groups)
    Y = JointRC(Y_reg, Y_class)

    return X, Y

def test_estimator():
    """Tests that the fit and predict methods work with the expected interface"""
    X, Y = generate_dummy_data()

    jmmlrc = JMMLRC(gamma1 = 10, gamma2 = 100, gamma3 = 1)

    jmmlrc.fit(X, Y, verbose=True)
    pred = jmmlrc.predict(X)
    
    assert "regression" in pred
    assert "classification" in pred

def test_kfold():
    """Test that the JMMLRC algorithm can utilize sklearn's KFold experiment"""
    X, Y = generate_dummy_data()

    jmmlrc = JMMLRC(gamma1 = 10, gamma2 = 100, gamma3 = 1)

    kf = KFold(n_splits=5)

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        jmmlrc.fit(X_train, Y_train)
        pred = jmmlrc.predict(X_test)

        assert pred["regression"].shape == Y_test[0].shape
        assert pred["classification"].shape == Y_test[1].shape

def test_grid_search_mse():
    """Test that the JMMLRC algorithm can utlize sklearn's GridSearchCV...

    Model selection is determined by the mean squared error of the 
    regression task.
    """
    X, Y = generate_dummy_data()

    jmmlrc = JMMLRC(gamma1 = 10, gamma2 = 100, gamma3 = 1)

    param_grid = {'gamma1': [.00001, 1, 10000],
                  'gamma2': [.00001, 1, 10000],
                  'gamma3': [.00001, 1, 10000],
                  'score_func': ['mean_squared_error']}

    search = GridSearchCV(jmmlrc, param_grid)
    search.fit(X, Y)

    assert search.best_params_ is not None

def test_grid_search_hinge():
    """Test that the JMMLRC algorithm can utlize sklearn's GridSearchCV...

    Model selection is determined by the hinge loss of the classification
    task..
    """
    X, Y = generate_dummy_data()

    jmmlrc = JMMLRC(gamma1 = 10, gamma2 = 100, gamma3 = 1)

    param_grid = {'gamma1': [.00001, 1, 10000],
                  'gamma2': [.00001, 1, 10000],
                  'gamma3': [.00001, 1, 10000],
                  'score_func': ['hinge']}

    search = GridSearchCV(jmmlrc, param_grid)
    search.fit(X, Y)

    assert search.best_params_ is not None
