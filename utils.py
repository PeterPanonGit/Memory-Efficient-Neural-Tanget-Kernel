import NTK
import numpy as np

def dot_prod_pre_allocated(x1:np.array, x2:np.array, m:NTK.MatrixF):
    """
    Compute the dot product between two matrices,
    and save output to a pre-allocated NTK.MatrixF.
    The reason of having such a function instead of using np.dot()
    is that we want to save the result directly to NTK.MatrixF
    instead of creating an intermediate matrix.
    This could save a lot of memory when n1 and n2 are large.

    Args:
        x1: matrix of shape (n1, k)
        x2: matrix of shape (n2, k)

    Returns:
        NTK.MatrixF of shape (n1, n2)
    """
    assert(len(x1.shape) == 2)
    assert(len(x2.shape) == 2)
    assert(x1.shape[1] == x2.shape[1])
    assert(m.nrow() == x1.shape[0])
    assert(m.ncol() == x2.shape[0])
    for i in range(x1.shape[0]):
        for j in range(x2.shape[0]):
            m.set_val(i, j, np.dot(x1[i, :], x2[j, :]))
