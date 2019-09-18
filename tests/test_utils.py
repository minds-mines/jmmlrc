#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np

from jmmlrc.utils import GroupedX
from jmmlrc.utils import JointRC

__author__ = "Lou Brand"
__copyright__ = "Lou Brand"
__license__ = "mit"


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
