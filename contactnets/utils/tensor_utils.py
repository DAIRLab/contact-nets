import pdb  # noqa

import torch
from torch import Tensor

from contactnets.utils import utils


def set_default_tensor_type(device: torch.device) -> None:
    """Set the appriate default tensor type according to the device."""
    is_cpu = str(device) == 'cpu'
    default_type = torch.DoubleTensor if is_cpu else torch.cuda.DoubleTensor  # type: ignore
    torch.set_default_tensor_type(default_type)


def matrix_diag(diagonal: Tensor) -> Tensor:
    """
    Written by fmassa at: https://github.com/pytorch/pytorch/issues/12160
    """
    N = diagonal.shape[-1]
    shape = diagonal.shape[:-1] + (N, N)
    device, dtype = diagonal.device, diagonal.dtype
    result = torch.zeros(shape, dtype=dtype, device=device)
    indices = torch.arange(result.numel(), device=device).reshape(shape)
    indices = indices.diagonal(dim1=-2, dim2=-1)
    result.view(-1)[indices] = diagonal
    return result


def veceye(n: int, veclen: int) -> Tensor:
    """Compute a block diagonal matrix with column vectors of ones as blocks.

    Example:

    veceye(3, 2) =
        tensor([[1., 0., 0.],
                [1., 0., 0.],
                [0., 1., 0.],
                [0., 1., 0.],
                [0., 0., 1.],
                [0., 0., 1.]])

    Args:
        n: number of columns.
        veclen: number of ones in each matrix diagonal block.

    Returns:
        A (n * veclen) x n matrix.
    """

    return torch.eye(n).repeat(1, veclen).reshape(n * veclen, n)


def pad_right(x: Tensor, elem: float, num: int) -> Tensor:
    """Right pad a batched tensor with an element.

    Args:
        x: batch_n x n x m.
        elem: element to pad with.
        num: how many columns filled with elem to add.

    Returns:
        A batch_n x n x (m + num) tensor. The new elements are all filled with elem.
    """
    pad = torch.ones(x.shape[0], x.shape[1], num) * elem
    return torch.cat((x, pad), dim=2)


def pad_left(x: Tensor, elem: float, num: int) -> Tensor:
    """Left pad a batched tensor with an element.

    Args:
        x: batch_n x n x m.
        elem: element to pad with.
        num: how many columns filled with elem to add.

    Returns:
        A batch_n x n x (num + m) tensor. The new elements are all filled with elem.
    """
    pad = torch.ones(x.shape[0], x.shape[1], num) * elem
    return torch.cat((pad, x), dim=2)


def pad_top(x: Tensor, elem: float, num: int) -> Tensor:
    """Top pad a batched tensor with an element.

    Args:
        x: batch_n x n x m.
        elem: element to pad with.
        num: how many rows filled with elem to add.

    Returns:
        A batch_n x (num + n) x m tensor. The new elements are all filled with elem.
    """
    pad = torch.ones(x.shape[0], num, x.shape[2]) * elem
    return torch.cat((pad, x), dim=1)


def pad_bottom(x: Tensor, elem: float, num: int) -> Tensor:
    """Bottom pad a batched tensor with an element.

    Args:
        x: batch_n x n x m.
        elem: element to pad with.
        num: how many rows filled with elem to add.

    Returns:
        A batch_n x (n + num) x m tensor. The new elements are all filled with elem.
    """
    pad = torch.ones(x.shape[0], num, x.shape[2]) * elem
    return torch.cat((x, pad), dim=1)


def diag_append(x: Tensor, elem: float, num: int) -> Tensor:
    """Diagonally pad a batched tensor with the identity times an element after the orginal.

    For each batched matrix, make the new matrix block diagonal with the original matrix in the
    upper left corner and eye(num) * elem in the bottom right corner.

    Args:
        x: batch_n x n x m.
        elem: element to pad with.
        num: the size of the identity matrix.

    Returns:
        A batch_n x (n + num) x (m + num) tensor. The new elements are all filled with elem.
    """

    batch_n = x.shape[0]

    brblock = torch.eye(num).unsqueeze(0).repeat(batch_n, 1, 1) * elem
    bottom_zeros = torch.zeros(batch_n, num, x.shape[2])
    bottom_block = torch.cat((bottom_zeros, brblock), dim=2)

    x = pad_right(x, 0, num)
    x = torch.cat((x, bottom_block), dim=1)
    return x


def diag_prepend(x: Tensor, elem: float, num: int) -> Tensor:
    """Diagonally pad a batched tensor with the identity times an element before the orginal.

    For each batched matrix, make the new matrix block diagonal with eye(num) * elem in the
    upper left corner and the original matrix in the bottom right corner.

    Args:
        x: batch_n x n x m.
        elem: element to pad with.
        num: the size of the identity matrix.

    Returns:
        A batch_n x (num + n) x (num + m) tensor. The new elements are all filled with elem.
    """
    batch_n = x.shape[0]

    tlblock = torch.eye(num).unsqueeze(0).repeat(batch_n, 1, 1) * elem
    top_zeros = torch.zeros(batch_n, num, x.shape[2])
    top_block = torch.cat((tlblock, top_zeros), dim=2)

    x = pad_left(x, 0, num)
    x = torch.cat((top_block, x), dim=1)
    return x


def robust_sqrt(out_squared: Tensor, eps = 1e-8) -> Tensor:
    # TODO: write description
    out = torch.zeros(out_squared.shape)
    out_big = out_squared >= eps ** 2
    out_small = torch.logical_not(out_big)
    out[out_big] = torch.sqrt(out_squared[out_big])
    out[out_small] = out_squared[out_small] * 0.5 / eps + 0.5 * eps
    return out


"""A part of the pylabyk library: numpytorch.py at https://github.com/yulkang/pylabyk"""


def block_diag(m):
    """
    Make a block diagonal matrix along dim=-3
    EXAMPLE:
    block_diag(torch.ones(4,3,2))
    should give a 12 x 8 matrix with blocks of 3 x 2 ones.
    Prepend batch dimensions if needed.
    You can also give a list of matrices.
    :type m: torch.Tensor, list
    :rtype: torch.Tensor
    """
    if type(m) is list:
        # Remove Nones from list
        m = utils.filter_none(m)

        m = torch.cat([m1.unsqueeze(-3) for m1 in m], -3)

    d = m.dim()
    n = m.shape[-3]
    siz0 = m.shape[:-3]
    siz1 = m.shape[-2:]
    m2 = m.unsqueeze(-2)
    eye = attach_dim(torch.eye(n).unsqueeze(-2), d - 3, 1)
    return (m2 * eye).reshape(siz0 + torch.Size(torch.tensor(siz1) * n))


def attach_dim(v, n_dim_to_prepend=0, n_dim_to_append=0):
    return v.reshape(
        torch.Size([1] * n_dim_to_prepend) + v.shape + torch.Size([1] * n_dim_to_append))
