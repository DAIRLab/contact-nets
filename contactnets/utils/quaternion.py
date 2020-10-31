# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree (QuaterNet).
#

import torch
import numpy as np

import pdb

def inv_mat(batch_n):
    Istar = torch.diag(torch.tensor([1.0, -1, -1, -1]))
    return Istar.unsqueeze(0).repeat(batch_n, 1, 1)

# PyTorch-backed implementations
def qinv(q):
    """
        Form q' from q (negative imaginary part)
    """
    assert q.shape[-1] == 4

    return inv_mat(q.shape[0]).bmm(q.unsqueeze(2)).squeeze(2)

def qmat(q):
    # Returns Q such that q p = Q(q) p
    r1 = torch.cat((q[:, 0:1], -q[:, 1:2], -q[:, 2:3], -q[:, 3:4]), dim = 1)
    r2 = torch.cat((q[:, 1:2], q[:, 0:1], -q[:, 3:4], q[:, 2:3]), dim = 1)
    r3 = torch.cat((q[:, 2:3], q[:, 3:4], q[:, 0:1], -q[:, 1:2]), dim = 1)
    r4 = torch.cat((q[:, 3:4], -q[:, 2:3], q[:, 1:2], q[:, 0:1]), dim = 1)
    Q = torch.cat((r1.unsqueeze(1), r2.unsqueeze(1), r3.unsqueeze(1), r4.unsqueeze(1)), dim=1)
    return Q

def pmat(p):
    # Returns Qhat such that q p = Qhat (p) q
    r1 = torch.cat((p[:, 0:1], -p[:, 1:2], -p[:, 2:3], -p[:, 3:4]), dim = 1)
    r2 = torch.cat((p[:, 1:2], p[:, 0:1], p[:, 3:4], -p[:, 2:3]), dim = 1)
    r3 = torch.cat((p[:, 2:3], -p[:, 3:4], p[:, 0:1], p[:, 1:2]), dim = 1)
    r4 = torch.cat((p[:, 3:4], p[:, 2:3], -p[:, 1:2], p[:, 0:1]), dim = 1)
    Q = torch.cat((r1.unsqueeze(1), r2.unsqueeze(1), r3.unsqueeze(1), r4.unsqueeze(1)), dim=1)
    return Q

def qjac(q, p):
    # Returns the jacobian of the coordinates p w.r.t.
    # the elements of q
    # https://math.stackexchange.com/questions/2713061/jacobian-of-a-quaternion-rotation-wrt-the-quaternion 
    batch_n = q.shape[0]

    quat = qmat(qmul(q, p)).bmm(inv_mat(batch_n)) + \
            pmat(qmul(p, qinv(q)))
    return quat[:, 1:4, :].transpose(1, 2)

def qmul(q, r):
    return qmat(q).bmm(r.unsqueeze(2)).squeeze(2)

def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]
    
    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)
    
    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)

def qdiff(q1, q2):
    # Returns the smallest angle between the rotations expressed by q1 and q2
    quat_diffs = qmul(q1, qinv(q2))
    axis_norms = torch.norm(quat_diffs[:, 1:4], 2, dim=1)
    return 2 * torch.asin(axis_norms)

def qeuler(q, order, epsilon=0):
    """
    Convert quaternion(s) q to Euler angles.
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    
    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = q.view(-1, 4)
    
    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]
    
    if order == 'xyz':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2*(q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q1 * q3 + q0 * q2), -1+epsilon, 1-epsilon))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2*(q2 * q2 + q3 * q3))
    elif order == 'yzx':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2*(q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2*(q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q1 * q2 + q0 * q3), -1+epsilon, 1-epsilon))
    elif order == 'zxy':
        x = torch.asin(torch.clamp(2 * (q0 * q1 + q2 * q3), -1+epsilon, 1-epsilon))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2*(q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2*(q1 * q1 + q3 * q3))
    elif order == 'xzy':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2*(q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2*(q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q0 * q3 - q1 * q2), -1+epsilon, 1-epsilon))
    elif order == 'yxz':
        x = torch.asin(torch.clamp(2 * (q0 * q1 - q2 * q3), -1+epsilon, 1-epsilon))
        y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2*(q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2*(q1 * q1 + q3 * q3))
    elif order == 'zyx':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2*(q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q0 * q2 - q1 * q3), -1+epsilon, 1-epsilon))
        z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2*(q2 * q2 + q3 * q3))
    else:
        raise

    return torch.stack((x, y, z), dim=1).view(original_shape)

# Numpy-backed implementations

def qmul_np(q, r):
    q = torch.from_numpy(q).contiguous()
    r = torch.from_numpy(r).contiguous()
    return qmul(q, r).numpy()

def qrot_np(q, v):
    q = torch.from_numpy(q).contiguous()
    v = torch.from_numpy(v).contiguous()
    return qrot(q, v).numpy()

def qeuler_np(q, order, epsilon=0, use_gpu=False):
    if use_gpu:
        q = torch.from_numpy(q).cuda()
        return qeuler(q, order, epsilon).cpu().numpy()
    else:
        q = torch.from_numpy(q).contiguous()
        return qeuler(q, order, epsilon).numpy()

def qfix(q):
    """
    Enforce quaternion continuity across the time dimension by selecting
    the representation (q or -q) with minimal distance (or, equivalently, maximal dot product)
    between two consecutive frames.
    
    Expects a tensor of shape (L, J, 4), where L is the sequence length and J is the number of joints.
    Returns a tensor of the same shape.
    """
    assert len(q.shape) == 3
    assert q.shape[-1] == 4
    
    result = q.copy()
    dot_products = np.sum(q[1:]*q[:-1], axis=2)
    mask = dot_products < 0
    mask = (np.cumsum(mask, axis=0)%2).astype(bool)
    result[1:][mask] *= -1
    return result

def expmap_to_quaternion(e):
    """
    Convert axis-angle rotations (aka exponential maps) to quaternions.
    Stable formula from "Practical Parameterization of Rotations Using the Exponential Map".
    Expects a tensor of shape (*, 3), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 4).
    """
    assert e.shape[-1] == 3
    
    original_shape = list(e.shape)
    original_shape[-1] = 4
    e = e.reshape(-1, 3)

    theta = np.linalg.norm(e, axis=1).reshape(-1, 1)
    w = np.cos(0.5*theta).reshape(-1, 1)
    xyz = 0.5*np.sinc(0.5*theta/np.pi)*e
    return np.concatenate((w, xyz), axis=1).reshape(original_shape)

def euler_to_quaternion(e, order):
    """
    Convert Euler angles to quaternions.
    """
    assert e.shape[-1] == 3
    
    original_shape = list(e.shape)
    original_shape[-1] = 4
    
    e = e.reshape(-1, 3)
    
    x = e[:, 0]
    y = e[:, 1]
    z = e[:, 2]
    
    rx = np.stack((np.cos(x/2), np.sin(x/2), np.zeros_like(x), np.zeros_like(x)), axis=1)
    ry = np.stack((np.cos(y/2), np.zeros_like(y), np.sin(y/2), np.zeros_like(y)), axis=1)
    rz = np.stack((np.cos(z/2), np.zeros_like(z), np.zeros_like(z), np.sin(z/2)), axis=1)

    result = None
    for coord in order:
        if coord == 'x':
            r = rx
        elif coord == 'y':
            r = ry
        elif coord == 'z':
            r = rz
        else:
            raise
        if result is None:
            result = r
        else:
            result = qmul_np(result, r)
            
    # Reverse antipodal representation to have a non-negative "w"
    if order in ['xyz', 'yzx', 'zxy']:
        result *= -1
    
    return result.reshape(original_shape)

def quaternion_to_rotmat_vec(q):
    """
    Converts batched quaternions of shape (batch, 4) to vectorized rotation matrices of shape (batch, 9)
    """
    qr = q[:, 0:1]
    qi = q[:, 1:2]
    qj = q[:, 2:3]
    qk = q[:, 3:4]
    r1 = torch.cat((1. - 2*(qj ** 2 + qk ** 2), 2*(qi*qj - qk*qr), 2*(qi*qk + qj*qr)), dim=1)
    r2 = torch.cat((2*(qi*qj + qk*qr), 1. - 2*(qi ** 2 + qk ** 2), 2*(qj*qk - qi*qr)), dim=1)
    r3 = torch.cat((2*(qi*qk - qj*qr), 2*(qj*qk + qi*qr), 1. - 2*(qi ** 2 + qj ** 2)), dim=1)
    return torch.cat((r1,r2,r3), dim=1)

def quaternion_to_rotmat_jac(q):
    """
    Converts batched quaternions q of shape (batch, 4) to the jacobian of the
    corresponding rotation matrix w.r.t. q of shape (batch, 9, 4)
    """
    qr = q[:, 0:1]
    qi = q[:, 1:2]
    qj = q[:, 2:3]
    qk = q[:, 3:4]
    z = torch.zeros_like(qk)

    r1 = 2. * torch.cat((z, z, -2. * qj, -2. * qk), dim=1)
    r2 = 2. * torch.cat((-qk, qj, qi, -qr), dim=1)
    r3 = 2. * torch.cat((qj, qk, qr, qi), dim=1)
    r4 = 2. * torch.cat((qk, qj, qi, qr), dim=1)
    r5 = 2. * torch.cat((z, -2 * qi, z, -2*qk), dim=1)
    r6 = 2. * torch.cat((-qi, -qr, qk, qj), dim=1)
    r7 = 2. * torch.cat((-qj, qk, -qr, qi), dim=1)
    r8 = 2. * torch.cat((qi, qr, qk, qj), dim=1)
    r9 = 2. * torch.cat((z, -2*qi, -2*qj, z), dim=1)

    return torch.cat((r1.unsqueeze(1), r2.unsqueeze(1), r3.unsqueeze(1), \
                      r4.unsqueeze(1), r5.unsqueeze(1), r6.unsqueeze(1), \
                      r7.unsqueeze(1), r8.unsqueeze(1), r9.unsqueeze(1)), dim=1)

def quaternion_to_rotmat(q):
    return quaternion_to_rotmat_vec(q).reshape(-1, 3, 3)
