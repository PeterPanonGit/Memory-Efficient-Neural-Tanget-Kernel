from NtkInternal import VectorF, MatrixF, Ntk
import numpy as np

def dot_prod_pre_allocated(x1:np.array, x2:np.array, m:MatrixF):
    """
    Compute the dot product between two matrices,
    and save output to a pre-allocated MatrixF.
    The reason of having such a function instead of using np.dot()
    is that we want to save the result directly to MatrixF
    instead of creating an intermediate matrix.
    This could save a lot of memory when n1 and n2 are large.

    Args:
        x1: matrix of shape (n1, k)
        x2: matrix of shape (n2, k)

    Returns:
        MatrixF of shape (n1, n2)
    """
    assert(len(x1.shape) == 2)
    assert(len(x2.shape) == 2)
    assert(x1.shape[1] == x2.shape[1])
    assert(m.nrow() == x1.shape[0])
    assert(m.ncol() == x2.shape[0])
    for i in range(x1.shape[0]):
        for j in range(x2.shape[0]):
            m.set_val(i, j, np.dot(x1[i, :], x2[j, :]))

class NtkIterator:
    def __init__(self, x1:np.array, x2:np.array, d_max:int):
        # self.x1 and self.x2 are views of x1 and x2.
        # They do not create additional memory
        self.x1 = x1
        self.x2 = x2
        self.d_max = d_max
        self.p1 = VectorF(np.linalg.norm(x1, axis=1))
        self.p2 = VectorF(np.linalg.norm(x2, axis=1))
        self.S = MatrixF(x1.shape[0], x2.shape[0])
        self.H = MatrixF(self.S)
        # dep points to the next layer to be trained
        self.dep = None
        # First fix_dep layers of the infinit wide neural network is fixed.
        self.fix_dep = None

    def set_fix_dep(self, fix_dep:int):
        self.fix_dep = fix_dep
        dot_prod_pre_allocated(self.x1, self.x2, self.S)
        for dep in range(1, fix_dep + 1):
            Ntk(dep, fix_dep, self.p1, self.p2, self.S, self.H)
        # At the first trainable layer, H==S
        self.H.copy(self.S)
        self.dep = fix_dep + 1

    def has_next(self):
        return self.dep < self.d_max

    def next(self):
        Ntk(self.dep, self.fix_dep, self.p1, self.p2, self.S, self.H)
        self.dep = self.dep + 1
