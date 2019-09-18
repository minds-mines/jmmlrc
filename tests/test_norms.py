#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import numpy as np

from jmmlrc.norms import l21_norm
from jmmlrc.norms import group_norm
from jmmlrc.norms import trace_norm

__author__ = "Lou Brand"
__copyright__ = "Lou Brand"
__license__ = "mit"

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
