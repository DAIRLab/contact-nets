import unittest
import numpy as np
from fastnnls import fnnls

class TestFNNLS(unittest.TestCase):

    def test_typical(self):
        """
        Test a typical input.
        """
        A = np.array([[6, 6, 2], [13, 14, 19], [18, 18, 1]])
        y = np.array([9, 5, 1])
        AtA = A.T.dot(A)
        Aty = A.T.dot(y)

        ans = np.array([0.159, 0.0, 0.191])
        ret = fnnls(AtA, Aty)
        np.testing.assert_array_almost_equal(ans, ret, decimal=3)

    def test_non_square(self):
        """
        Test a case where the input is not square.
        """
        A = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 2, 3])
        AtA = A.T.dot(A)
        Aty = A.T.dot(y)

        ans = np.array([0.0, 0.5])
        ret = fnnls(AtA, Aty)
        np.testing.assert_array_almost_equal(ans, ret, decimal=3)

    def test_neg(self):
        """
        Test a case where the matrix A has negative entries.
        """
        A = np.array([[-1, 2], [3, 4], [5, -6]])
        y = np.array([1, 2, 3])
        AtA = A.T.dot(A)
        Aty = A.T.dot(y)

        ans = np.array([0.615, 0.0769])
        ret = fnnls(AtA, Aty)
        np.testing.assert_array_almost_equal(ans, ret, decimal=3)

    def test_inner(self):
        """
        Test a case where the matrix A does not have full rank.
        This case forces FNNLS to enter the inner while loop.
        """
        A = np.array([[1, 2], [3, 4], [0, 0]])
        y = np.array([1, 2, 3])
        AtA = A.T.dot(A)
        Aty = A.T.dot(y)

        ans = np.array([0, 0.5])
        ret = fnnls(AtA, Aty)
        np.testing.assert_array_almost_equal(ans, ret, decimal=3)

if __name__ == '__main__':
    unittest.main()
