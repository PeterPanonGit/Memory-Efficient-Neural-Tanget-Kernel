import NTK
import math
import numpy as np
import unittest
from utils import dot_prod_pre_allocated

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
    for fix_dep in range(d_max):
        S = np.matmul(X, X.T)
        H = np.zeros_like(S)
        for dep in range(d_max):
            if fix_dep <= dep:
                H += S
            K[dep][fix_dep] = H
            L = np.diag(S)
            P = np.clip(np.sqrt(np.outer(L, L)), a_min = 1e-9, a_max = None)
            Sn = np.clip(S / P, a_min = -1, a_max = 1)
            S = (Sn * (math.pi - np.arccos(Sn)) + np.sqrt(1.0 - Sn * Sn)) * P / math.pi
            H = H * (math.pi - np.arccos(Sn)) / math.pi
    return K

class TestMatrix(unittest.TestCase):
    def setUp(self):
        self.epsilon = 1e-30

    def test_zero_init(self):
        m = NTK.MatrixF(3, 2)
        a = np.array(m, copy=False)
        self.assertLess(np.sum(np.abs(a)), self.epsilon)
        self.assertEqual(a.dtype, np.float32)

    def test_set_val(self):
        # m and a are expected to share the same memory space
        m = NTK.MatrixF(3, 2)
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
        m = NTK.MatrixF(3, 2)
        m.set_val(0, 0, 1)
        m.set_zero()
        self.assertLess(np.sum(np.abs(np.array(m, copy=False))), self.epsilon)

    def test_copy_ctor(self):
        m = NTK.MatrixF(3, 2)
        m.set_val(0, 0, 1)

        m1 = NTK.MatrixF(m)
        # confirm copy success
        self.assertEqual(m1.value(0, 0), 1)

        # confirm m1 and m use different memory address
        m.set_val(0, 1, 2)
        self.assertEqual(m.value(0, 1), 2)
        self.assertEqual(m1.value(0, 1), 0)

    def test_copy(self):
        m = NTK.MatrixF(3, 2)
        m.set_val(0, 0, 1)

        m1 = NTK.MatrixF(3, 2)
        m1.copy(m)
        # confirm copy success
        self.assertEqual(m1.value(0, 0), 1)

        # confirm m1 and m use different memory address
        m.set_val(0, 1, 2)
        self.assertEqual(m.value(0, 1), 2)
        self.assertEqual(m1.value(0, 1), 0)

class TestNTK(unittest.TestCase):
    def setUp(self):
        self.epsilon = 1e-2

    def test_NTK(self):
        # 2 data points, each data point has 3 features
        x = np.array([[0, -1.0, 1.0], [0.2, -1.2, 0.8]], dtype=np.float32)
        d_max = 10
        # allocate memory for data
        p = NTK.VectorF(np.linalg.norm(x, axis=1))
        S = NTK.MatrixF(x.shape[0], x.shape[0])
        H = NTK.MatrixF(S)

        # compute neural tangent kernel with python implementation
        K = kernel_value_batch(x, d_max)

        fix_dep = 0
        dot_prod_pre_allocated(x, x, S)
        H.copy(S)
        for dep in range(1, d_max):
            NTK.Ntk(dep, fix_dep, p, p, S, H)
            self.assertLess(np.sum(np.abs(H - K[dep][fix_dep])), self.epsilon)

        fix_dep = 2
        dot_prod_pre_allocated(x, x, S)
        H.set_zero()
        for dep in range(1, d_max):
            if dep == fix_dep: H.copy(S)
            NTK.Ntk(dep, fix_dep, p, p, S, H)
            print(dep, np.array(H, copy=False), K[dep][fix_dep])
            self.assertLess(np.sum(np.abs(H - K[dep][fix_dep])), self.epsilon)

if __name__ == '__main__':
    unittest.main()
7
