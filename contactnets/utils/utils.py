import torch

import pickle
import re
import math
import os
import psutil

import pdb

import contactnets.interaction
import contactnets.utils.quaternion as quat
from contactnets.utils import tensor_utils

def elements_identical(l):
    if len(l) == 0:
        return True

    return l.count(l[0]) == len(l)

def filter_none(l):
    # Removes None elements from list
    return [i for i in l if (i is not None)]

def list_dict_swap(v):
    # Convert list of dicts to dict of lists
    return {k: [dic[k] for dic in v] for k in v[0]}

def transpose_lists(l):
    # Transpose list of lists
    return list(map(list, zip(*l)))

def process_memory():
    # return process memory in megabytes
    process = psutil.Process(os.getpid())
    return process.memory_info().rss * 1e-6

def compute_quadratic_loss(A, b, c, x):
    return x.transpose(1, 2).bmm(A).bmm(x) + b.bmm(x) + c

def generate_normalizing_layer(data):
    means = data.mean(dim=0)
    stds = data.std(dim=0)

    stds_recip = 1 / stds
    stds_recip[stds_recip == float('Inf')] = 0

    layer = torch.nn.Linear(data.shape[1], data.shape[1], bias=True)
    layer.weight = torch.nn.Parameter(torch.diag(stds_recip), requires_grad=False)
    layer.bias = torch.nn.Parameter(-means * stds_recip, requires_grad=False)
    return layer

def create_geometry2d(vertices):
    # vertices are 2 x n tensor
    angles = -torch.atan2(vertices[0, :], vertices[1, :]) + math.pi / 2
    return contactnets.interaction.PolyGeometry2D(vertices, angles)

def rot2d(theta):
    batch_n = theta.shape[0]
    # Takes in batch theta vector and makes batches of rotation matrices
    c, s = torch.cos(theta), torch.sin(theta)
    c = c.unsqueeze(1).unsqueeze(0)
    s = s.unsqueeze(1).unsqueeze(0)
    r1 = torch.cat((c, -s), dim=2).reshape(batch_n, 1, 2)
    r2 = torch.cat((s, c), dim=2).reshape(batch_n, 1, 2)
    rots = torch.cat((r1, r2), dim=1)

    return rots

def transform_vertices_2d(x, vertices):
    # Transforms vertices by state x, y, rot
    # x is batched
    k = vertices.shape[1]
    batch_n = x.shape[0]

    vertices = vertices.unsqueeze(0).repeat(batch_n, 1, 1)

    if x.shape[1] >= 3:
        # Check if has rot field
        rot = rot2d(x[:, 2])
        vertices = torch.bmm(rot, vertices)

    trans = x[:, 0:2, :].repeat(1, 1, k)
    vertices = vertices + trans
    return vertices

def transform_vertices_3d(x, vertices):
    # Transforms vertices by state x, y, z, rot quaternion
    # state is batched
    # vertices are batch_n x 3 x k matrix

    k = vertices.shape[2]
    batch_n = x.shape[0]
    vert_quat = torch.cat((torch.zeros(batch_n, k, 1), vertices.transpose(1,2)), dim=2)
    rot_quat = x[:, 3:7, :].squeeze(2)

    vert_quat = vert_quat.reshape(k * batch_n, -1)
    rot_quat = rot_quat.repeat(1, k).reshape(k * batch_n, -1)

    vert_rot = quat.qmul(quat.qmul(rot_quat, vert_quat), quat.qinv(rot_quat))

    vert_rot = vert_rot.reshape(batch_n, k, 4)

    vert_rot = vert_rot[:, :, 1:4]

    pos_shift = x[:, 0:3, :].transpose(1,2)

    return vert_rot + pos_shift

def transform_and_project_3d(x, vertices, ed):
    batch_n = x.shape[0]
    if vertices.dim() == 2:
        vertices = vertices.unsqueeze(0).repeat(batch_n, 1, 1)

    vert_transformed = transform_vertices_3d(x, vertices)

    dists = vert_transformed.bmm(ed.transpose(1,2).repeat(batch_n,1,1))

    if ed.nelement() > 3:
        k = vertices.shape[2]
        dists = dists.reshape(batch_n, ed.shape[1] * k, 1)

    return dists.transpose(1,2)

def transform_and_project_3d_jacobian(x, vertices, ed, vertex_jac=False):
    # Jacobian of transformed and projected vertices w.r.t. x
    # If vertex_jac is true, it also stacks the jacobian w.r.t. vertices
    batch_n = x.shape[0]
    k = vertices.shape[1]
    qrot = x[:, 3:7, :].squeeze(2)
    qverts = torch.cat((torch.zeros(k, 1), vertices.t()), dim=1)

    extract_n = ed.shape[1]
    qrot = qrot.repeat(1, k).reshape(k * batch_n, -1)

    qverts = qverts.repeat(batch_n, 1)


    qjac = quat.qjac(qrot, qverts)
    qjac = qjac.reshape(-1, qjac.shape[1], qjac.shape[2])
    ed = ed.repeat(k * batch_n, 1, 1)

    rot_jac_dist = ed.bmm(qjac.transpose(1,2)).reshape(batch_n, k * extract_n, -1)
    pos_jac_dist = ed.reshape(batch_n, k * extract_n, -1)

    jac = torch.cat((pos_jac_dist, rot_jac_dist), dim=2)

    if vertex_jac:
        vertjac = quat.quaternion_to_rotmat(qverts)
        # ONLY WORKS FOR BATCH_N = 1
        #vertjac_dist = ed.bmm(vertjac.transpose(1,2)).reshape(k * extract_n, -1, 1)
        vertjac_dist = ed.bmm(vertjac.transpose(1,2)).transpose(1,2)

        #vertjac_dist = tensor_utils.block_diag(vertjac_dist).reshape(batch_n, k, k * 3)
        vertjac_dist = tensor_utils.block_diag(vertjac_dist).t().unsqueeze(0).repeat(batch_n, 1, 1)
        jac = torch.cat((jac, vertjac_dist), dim=2)

    return jac

def vertices_to_pR_weights(vertices):
    # z components first
    k = vertices.shape[1]
    Jx_weights = torch.cat((torch.ones(k,1),torch.zeros(k,2),vertices.t(),torch.zeros(k,6)), dim=1)
    Jy_weights = torch.cat((torch.zeros(k,1),torch.ones(k,1),torch.zeros(k,1 + 3),vertices.t(),torch.zeros(k,3)), dim=1)
    Jz_weights = torch.cat((torch.zeros(k,2),torch.ones(k,1),torch.zeros(k,6),vertices.t()), dim=1)

    Jt_weights = torch.cat((Jx_weights,Jy_weights),dim=1).reshape(2 * k, 12)
    return (Jz_weights, Jt_weights)

