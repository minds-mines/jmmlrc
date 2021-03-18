#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np

from sklearn.model_selection import KFold, GridSearchCV

from utils import GroupedX, JointRC
from norms import l21_norm, group_norm, trace_norm

from jmmlrc_estimator import JMMLRC

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

def test_l21_norm():
    """Test that l21_norm() works correctly."""
    X = np.array([[1, 2, 3],     # 14
                  [4, 5, 6],     # 77
                  [7, 8, 9],     # 194
                  [10, 11, 12]]) # 365

    assert l21_norm(X) == 14 ** (1./2) + 77 ** (1./2) + 194 ** (1./2) + 365 ** (1./2)

def test_group_norm():
    """Test that group_norm() works correctly."""
    Y = np.array([[1, 2],   # Group 1
                  [3, 4],   # ---1---
                  [5, 6],   # ---1---
                  [7, 8],   # Group 2
                  [9, 10]]) # ---2---

    groups = [3, 5]

    assert group_norm(Y, groups) == 91 ** (1./2) + 294 ** (1./2)

def test_trace_norm():
    """Test that trace_norm() works correctly."""
    Z = np.array([[1, 2, 3],
                  [4, 5, 6]])

    assert trace_norm(Z) == 91 ** (1./2)

def test_build_groupedx():
    """Test that group_norm() works correctly."""
    data =  np.array([[.2, .4, .6, .2, .4, .6],
                      [.3, .1, .8, .3, .1, .8],
                      [.1, .9, .3, .1, .9, .3],
                      [.5, .3, .7, .5, .3, .7],
                      [.3, .7, .6, .3, .7, .6]])
    # Group Number :   #1  #2  #2  #2  #3  #3
                  
    groups = [1, 4, 6]

    X = GroupedX(data, groups)

    # Verify that the shape is correct
    assert X.shape == (5, 6)

    # Make sure that the indexing is working as intended
    data = X[3][0]
    grouping = X[3][1]
    np.testing.assert_array_equal(data, np.array([.5, .3, .7, .5, .3, .7]))
    np.testing.assert_array_equal(grouping, np.array([1, 4, 6]))

    # Be sure that splitting by the proivded grouping is correct
    X_groups = np.split(X.data, X.groups, axis=1)
    assert X_groups[0].shape == (5, 1)
    assert X_groups[1].shape == (5, 3)
    assert X_groups[2].shape == (5, 2)

def test_build_jointrc():
    """Test that building a joint regression/classification object (JointRC) works correctly."""
    regression_values = np.array([[.2, .4, .6],
                                  [.3, .1, .8],
                                  [.1, .9, .3],
                                  [.5, .3, .7],
                                  [.3, .7, .6]])

    class_labels = np.array([[1],
                             [2],
                             [3],
                             [1],
                             [3]])

    Y = JointRC(regression_values, class_labels)

    # See if the shape of the combined regression/classification object is correct
    assert Y.shape == (5, 4)
    np.testing.assert_array_equal(Y.rc[0], np.array([.2, .4, .6, 1]))

    # Determine whether the indexing is appropriate
    Y_reg, Y_class = Y[3]
    np.testing.assert_array_equal(Y_reg, np.array([.5, .3, .7]))
    np.testing.assert_array_equal(Y_class, np.array([1]))

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
