import numpy as np


def fnnls(AtA, Aty, epsilon=None, iter_max=None):
    """
    Given a matrix A and vector y, find x which minimizes the objective function
    f(x) = ||Ax - y||^2.
    This algorithm is similar to the widespread Lawson-Hanson method, but
    implements the optimizations described in the paper
    "A Fast Non-Negativity-Constrained Least Squares Algorithm" by
    Rasmus Bro and Sumen De Jong.

    Note that the inputs are not A and y, but are
    A^T * A and A^T * y

    This is to avoid incurring the overhead of computing these products
    many times in cases where we need to call this routine many times.

    :param AtA:       A^T * A. See above for definitions. If A is an (m x n)
                      matrix, this should be an (n x n) matrix.
    :type AtA:        numpy.ndarray
    :param Aty:       A^T * y. See above for definitions. If A is an (m x n)
                      matrix and y is an m dimensional vector, this should be an
                      n dimensional vector.
    :type Aty:        numpy.ndarray
    :param epsilon:   Anything less than this value is consider 0 in the code.
                      Use this to prevent issues with floating point precision.
                      Defaults to the machine precision for doubles.
    :type epsilon:    float
    :param iter_max:  Maximum number of inner loop iterations. Defaults to
                      30 * [number of cols in A] (the same value that is used
                      in the publication this algorithm comes from).
    :type iter_max:   int, optional
    """
    if epsilon is None:
        epsilon = np.finfo(np.float64).eps

    n = AtA.shape[0]

    if iter_max is None:
        iter_max = 30 * n

    if Aty.ndim != 1 or Aty.shape[0] != n:
        raise ValueError('Invalid dimension; got Aty vector of size {}, ' \
                         'expected {}'.format(Aty.shape, n))

    # Represents passive and active sets.
    # If sets[j] is 0, then index j is in the active set (R in literature).
    # Else, it is in the passive set (P).
    sets = np.zeros(n, dtype=np.bool)
    # The set of all possible indices. Construct P, R by using `sets` as a mask
    ind = np.arange(n, dtype=int)
    P = ind[sets]
    R = ind[~sets]

    x = np.zeros(n, dtype=np.float64)
    w = Aty
    s = np.zeros(n, dtype=np.float64)

    i = 0
    # While R not empty and max_(n \in R) w_n > epsilon
    while not np.all(sets) and np.max(w[R]) > epsilon and i < iter_max:
        # Find index of maximum element of w which is in active set.
        j = np.argmax(w[R])
        # We have the index in MASKED w.
        # The real index is stored in the j-th position of R.
        m = R[j]

        # Move index from active set to passive set.
        sets[m] = True
        P = ind[sets]
        R = ind[~sets]

        # Get the rows, cols in AtA corresponding to P
        AtA_in_p = AtA[P][:, P]
        # Do the same for Aty
        Aty_in_p = Aty[P]

        # Update s. Solve (AtA)^p * s^p = (Aty)^p
        s[P] = np.linalg.lstsq(AtA_in_p, Aty_in_p, rcond=None)[0]
        s[R] = 0.

        while np.any(s[P] <= epsilon):
            i += 1

            mask = (s[P] <= epsilon)
            alpha = np.min(x[P][mask] / (x[P][mask] - s[P][mask]))
            x += alpha * (s - x)

            # Move all indices j in P such that x[j] = 0 to R
            # First get all indices where x == 0 in the MASKED x
            zero_mask = (x[P] < epsilon)
            # These correspond to indices in P
            zeros = P[zero_mask]
            # Finally, update the passive/active sets.
            sets[zeros] = False
            P = ind[sets]
            R = ind[~sets]

            # Get the rows, cols in AtA corresponding to P
            AtA_in_p = AtA[P][:, P]
            # Do the same for Aty
            Aty_in_p = Aty[P]

            # Update s. Solve (AtA)^p * s^p = (Aty)^p
            s[P] = np.linalg.lstsq(AtA_in_p, Aty_in_p, rcond=None)[0]
            s[R] = 0.

        x = s.copy()
        w = Aty - AtA.dot(x)

    return x
