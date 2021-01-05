import pdb  # noqa

from lemkelcp import lemkelcp as lcp
import numpy as np
from osqpth.osqpth import OSQP
import torch
from torch import Tensor


def _to_numpy(x: Tensor) -> np.ndarray: return x.detach().cpu().numpy()


def solve_lcp(lcp_mat: Tensor, lcp_vec: Tensor) -> Tensor:
    """Solve a single Linear Complementarity Problem."""
    sol, exit_code, msg = lcp.lemkelcp(_to_numpy(lcp_mat), _to_numpy(lcp_vec), maxIter = 1000)

    assert exit_code == 0, msg
    return torch.tensor(sol).double().unsqueeze(1)


def solve_lcp_batch(lcp_mats: Tensor, lcp_vecs: Tensor) -> Tensor:
    """Solve a batch of Linear Complementarity Problems.

    Note that this is not a differentiable operation.

    Args:
        lcp_mats: batch_n x n x n.
        lcp_vecs: batch_n x n x 1.

    Returns:
        A tensor of the shape batch_n x n x 1, representing the solutions to each LCP problem.
    """
    sols = [solve_lcp(mat, vec).unsqueeze(0) for (mat, vec) in zip(lcp_mats, lcp_vecs)]
    return torch.cat(sols, dim=0)


def solve_qp_osqpth(A: Tensor, b: Tensor, R: Tensor, h: Tensor) -> Tensor:
    """Solve a quadratic program using osqpth (differentiable).

    Osqpth is a differentiable operating-splitting quadratic program solver inspired by qpth,
    a previous effort by the same authors. We found empirically that osqpth solves problems
    we're interested in around 5x faster than qpth.

    Args:
        A: batch_n x n x n. Quadratic coefficient. n is the number of QP variables.
        b: batch_n x 1 x n. Linear coefficient.
        R: batch_n x r x n. Inequality matrix. r is the number of inequalities.
        h: batch_n x r x 1. Inequality vector.

    Returns:
        A tensor of the shape batch_n x n x 1, representing the solutions to each QP problem.
    """
    assert A.shape[0] == b.shape[0] == R.shape[0] == h.shape[0]
    batch_n = A.shape[0]

    # Just pass all indices to OSQP
    A_ones = _to_numpy(torch.ones_like(A[0]))
    R_ones = _to_numpy(torch.ones_like(R[0]))

    solver = OSQP(A_ones.nonzero(),
                  A_ones.shape,
                  R_ones.nonzero(),
                  R_ones.shape,
                  max_iter=100000,
                  eps_abs=1e-6,
                  eps_rel=1e-6)

    # Handle weird OSQPTH batch sizing
    def preprocess(x): return x.flatten() if batch_n == 1 else x.reshape(batch_n, -1)
    (A, b, R, h) = map(preprocess, (A, b, R, h))

    sol = solver(A, b, R, torch.ones_like(h) * (-float('inf')), h)

    return sol.reshape(batch_n, -1, 1)
