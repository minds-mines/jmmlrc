import numpy as np

def l21_norm(X):
    """Define the l21 norm for the matrix X."""
    row_norms = np.linalg.norm(X, axis=1)
    l21_norm = np.sum(row_norms)
    return l21_norm

def group_norm(X, groups):
    """Define the group norm for the matrix X and the provided groups."""
    groups = np.split(X, groups)
    norms = [np.linalg.norm(group) for group in groups]
    group_norm = np.sum(norms)
    return group_norm

def trace_norm(X):
    """Define the trace norm for the matrix X."""
    trace_norm = np.trace(X @ X.T) ** (1./2)
    return trace_norm
