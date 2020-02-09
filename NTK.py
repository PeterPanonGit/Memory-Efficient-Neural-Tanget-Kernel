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
    """
    Wrapper class for the C++ Ntk() function
    At construction time, norm of input data is computed and save
        so that there is no need to recompute in every iteration,
        since the norm factor of activation covariance matrix
        does not change. This is a performance optimization.
    At construction time, activation covariance matrix S and 
        kernel matrix H memory are allocated and every update
        is applied to S and H in place. This improves speed and
        saves memory
    """
    def __init__(self, x1:np.array, x2:np.array, d_max:int):
        # self.x1 and self.x2 are views of x1 and x2.
        # They do not create additional memory
        self.x1 = x1
        self.x2 = x2
        # d_max is the max layers of neural networks to be trained
        # It follows the python way of specifying range, which means
        #   layer [1, 2, ..., d_max-1] are trained, but not including d_max
        self.d_max = d_max
        # p1 and p2 are clipped at 1e-9 to avoid division by 0
        self.p1 = VectorF(np.clip(np.linalg.norm(x1, axis=1), \
            a_min = 1e-9, a_max = None))
        self.p2 = VectorF(np.clip(np.linalg.norm(x2, axis=1), \
            a_min = 1e-9, a_max = None))
        # Allocate memory for S and H
        self.S = MatrixF(x1.shape[0], x2.shape[0])
        self.H = MatrixF(self.S)
        # dep points to the next layer to be trained
        self.dep = None
        # First fix_dep layers of the infinit wide neural network is fixed.
        self.fix_dep = None

    def set_fix_dep(self, fix_dep:int):
        """
        This method must be called before next(), otherwise dep and fix_dep
            will be None and next() will throw error
        """
        self.fix_dep = fix_dep
        dot_prod_pre_allocated(self.x1, self.x2, self.S)
        # If first fix_dep layers are not trained, kernel matrix H will be 0
        #   and thus it only makes sense to start training at fix_dep + 1
        self.dep = fix_dep + 1
        for dep in range(1, fix_dep + 1):
            Ntk(self.p1, self.p2, self.S, self.H)
        # At the first trainable layer, H==S
        self.H.copy(self.S)

    def has_next(self):
        if self.dep is None:
            raise RuntimeError("Please call set_fix_dep() first")

        return self.dep < self.d_max

    def next(self):
        """Update S and H"""
        if self.dep is None or self.fix_dep is None:
            raise RuntimeError("Please call set_fix_dep() first")

        Ntk(self.p1, self.p2, self.S, self.H)
        self.dep = self.dep + 1

    def __del__(self):
        """
        Explicitly call destructors of the C++ objects to ensure their
            destructors are called and their memories are freed.
        """
        del self.p1; del self.p2
        del self.S; del self.H
