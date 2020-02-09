from NtkInternal import VectorF, MatrixF
import math
import numpy as np
import unittest
from utils import dot_prod_pre_allocated, NtkIterator

def kernel_value_batch(X, d_max): 
    """
    Reference python implementation from:
        https://github.com/LeoYu/neural-tangent-kernel-UCI/blob/master/NTK.py

    Args:
        X: input data, shape: (n, k)
        d_max: max number of layers

    Returns:
        K: neural tanget kernel matrices at different layers and fixed number of layers
    """
    K = np.zeros((d_max, d_max, X.shape[0], X.shape[0]))
    SS = np.zeros((d_max, d_max, X.shape[0], X.shape[0]))
    for fix_dep in range(d_max):
        S = np.matmul(X, X.T)
        H = np.zeros_like(S)
        for dep in range(d_max):
            if fix_dep <= dep:
                H += S
            K[dep][fix_dep] = H
            SS[dep][fix_dep] = S
            L = np.diag(S)
            P = np.clip(np.sqrt(np.outer(L, L)), a_min = 1e-9, a_max = None)
            Sn = np.clip(S / P, a_min = -1, a_max = 1)
            S = (Sn * (math.pi - np.arccos(Sn)) + np.sqrt(1.0 - Sn * Sn)) * P / math.pi
            H = H * (math.pi - np.arccos(Sn)) / math.pi
    return K, SS

class TestMatrix(unittest.TestCase):
    def setUp(self):
        self.epsilon = 1e-30

    def test_zero_init(self):
        m = MatrixF(3, 2)
        a = np.array(m, copy=False)
        self.assertLess(np.sum(np.abs(a)), self.epsilon)
        self.assertEqual(a.dtype, np.float32)

    def test_set_val(self):
        # m and a are expected to share the same memory space
        m = MatrixF(3, 2)
        a = np.array(m, copy=False)

        # Test if m changes, a changes
        m.set_val(0, 0, 1)
        self.assertEqual(a[0, 0], 1)

        # Test if a changes, m changes
        a[0, 1] = 2
        self.assertEqual(m.value(0, 1), 2)

        # Test set_val out of bound
        try:
            m.set_val(3, 0, 5)
        except Exception as e:
            self.assertEqual(str(e), "index (3, 0) is out of bound")

    def test_set_zero(self):
        m = MatrixF(3, 2)
        m.set_val(0, 0, 1)
        m.set_zero()
        self.assertLess(np.sum(np.abs(np.array(m, copy=False))), self.epsilon)

    def test_copy_ctor(self):
        m = MatrixF(3, 2)
        m.set_val(0, 0, 1)

        m1 = MatrixF(m)
        # confirm copy success
        self.assertEqual(m1.value(0, 0), 1)

        # confirm m1 and m use different memory address
        m.set_val(0, 1, 2)
        self.assertEqual(m.value(0, 1), 2)
        self.assertEqual(m1.value(0, 1), 0)

    def test_copy(self):
        m = MatrixF(3, 2)
        m.set_val(0, 0, 1)

        m1 = MatrixF(3, 2)
        m1.copy(m)
        # confirm copy success
        self.assertEqual(m1.value(0, 0), 1)

        # confirm m1 and m use different memory address
        m.set_val(0, 1, 2)
        self.assertEqual(m.value(0, 1), 2)
        self.assertEqual(m1.value(0, 1), 0)

class TestNTK(unittest.TestCase):
    def setUp(self):
        self.epsilon = 1e-3

    def test_NTK_one_x(self):
        # 1 data set, 2 data points, each data point has 3 features
        x = np.array([[0, -1.0, 1.0], [0.2, -1.2, 0.8]], dtype=np.float32)
        d_max = 10

        # NtkIterator instance
        ntk_iter = NtkIterator(x, x, d_max)

        # compute neural tangent kernel with python implementation
        K, Spy = kernel_value_batch(x, d_max)

        for fix_dep in range(d_max - 1):
            ntk_iter.set_fix_dep(fix_dep)
            while ntk_iter.has_next():
                Hpy = K[ntk_iter.dep][fix_dep]
                ntk_iter.next()
                self.assertLess(np.sum(np.abs(ntk_iter.H - Hpy)), self.epsilon)

    def test_NTK_two_x(self):
        # 2 data sets, 3 data point in first one, 2 data point in second one
        # Each data point has 3 features
        x1 = np.array([[0.1, 0.3, 0.05], [0.2, 0.1, 0.3], [0.4, 0.3, 0.4]])
        x2 = np.array([[0, -1.0, 1.0], [0.2, -1.2, 0.8]])
        d_max = 10

        # NtkIterator instance
        ntk_iter = NtkIterator(x1, x2, d_max)

        # compute neural tangent kernel with python implementation
        K, Spy = kernel_value_batch(np.vstack([x1, x2]), d_max)

        for fix_dep in range(d_max - 1):
            ntk_iter.set_fix_dep(fix_dep)
            while ntk_iter.has_next():
                Hpy = K[ntk_iter.dep][fix_dep][:x1.shape[0], x1.shape[0]:]
                ntk_iter.next()
                self.assertLess(np.sum(np.abs(ntk_iter.H - Hpy)), self.epsilon)

if __name__ == '__main__':
    unittest.main()
