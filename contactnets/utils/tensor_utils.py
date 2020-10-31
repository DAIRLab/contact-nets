import torch

from contactnets.utils import utils

def matrix_diag(diagonal):
    N = diagonal.shape[-1]
    shape = diagonal.shape[:-1] + (N, N)
    device, dtype = diagonal.device, diagonal.dtype
    result = torch.zeros(shape, dtype=dtype, device=device)
    indices = torch.arange(result.numel(), device=device).reshape(shape)
    indices = indices.diagonal(dim1=-2, dim2=-1)
    result.view(-1)[indices] = diagonal
    return result

def veceye(n, veclen):
    E = torch.eye(n).repeat(1, veclen).reshape(n * veclen, n)
    return E

def pad_mat_right(mat, elem, num):
    # Assume mat is batched
    pad = torch.ones(mat.shape[0], mat.shape[1], num) * elem
    return torch.cat((mat, pad), dim=2)

def pad_mat_left(mat, elem, num):
    # Assume mat is batched
    pad = torch.ones(mat.shape[0], mat.shape[1], num) * elem
    return torch.cat((pad, mat), dim=2)

def pad_mat_bottom(mat, elem, num):
    # Assume mat is batched
    pad = torch.ones(mat.shape[0], num, mat.shape[2]) * elem
    return torch.cat((mat, pad), dim=1)

def pad_mat_top(mat, elem, num):
    # Assume mat is batched
    pad = torch.ones(mat.shape[0], num, mat.shape[2]) * elem
    return torch.cat((pad, mat), dim=1)

def diag_expand_mat(mat, diag_elem, num):
    # Assume mat is batched
    # Will append column of zeros and row of zeros
    # With bottom right element being diag_elem
    # num times
    batch_n = mat.shape[0]
    
    
    brblock = torch.eye(num).unsqueeze(0).repeat(batch_n, 1, 1) * diag_elem
    bottom_zeros = torch.zeros(batch_n, num, mat.shape[2])
    bottom_block = torch.cat((bottom_zeros, brblock), dim=2)

    mat = pad_mat_right(mat, 0, num)
    mat = torch.cat((mat, bottom_block), dim=1)
    return mat

def diag_prepend_mat(mat, diag_elem, num):
    # Assume mat is batched
    # Will append column of zeros and row of zeros
    # With bottom right element being diag_elem
    # num times
    batch_n = mat.shape[0]
    
    
    tlblock = torch.eye(num).unsqueeze(0).repeat(batch_n, 1, 1) * diag_elem
    top_zeros = torch.zeros(batch_n, num, mat.shape[2])
    top_block = torch.cat((tlblock, top_zeros), dim=2)

    mat = pad_mat_left(mat, 0, num)
    mat = torch.cat((top_block, mat), dim=1)
    return mat

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
        torch.Size([1] * n_dim_to_prepend)
        + v.shape
        + torch.Size([1] * n_dim_to_append))
