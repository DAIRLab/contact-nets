# flake8: noqa
# TODO: clean up

import csv
import glob
import math
import os
import pdb  # noqa
import pickle
import random
from random import randrange
import time
from typing import List, Tuple

import click
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline
import torch

from contactnets.experiments.block3d.sim import Block3DParams
from contactnets.utils import dirs, file_utils

BLOCK_HALF_WIDTH = 0.0524


def do_process(inbag: str, downsamp: int, center: bool, perturb: bool, zrot: bool,
               pullback: bool, toss_comp: int, zshift: float, length_scale: float,
               use_python3: bool, plot: bool) -> Tuple[int, int]:
    DOWNSAMP = downsamp

    # static parameters for toss detection
    TOSS_BALL_TOLERANCE = 3e-4
    TOSS_BALL_H = 10
    FALL_BALL_H = TOSS_BALL_H
    FALLING_H = 2
    RELEASE_H = 2
    G = -9.81
    FALL_TOL = 3.
    FALL_TOL_MEAN = 1.
    # FALL_ACC_MIN = G - FALL_TOL
    FALL_ACC_MAX = G + FALL_TOL

    # far edge of board in board coordinates (y-axis)
    BOARD_EDGE_Y_MAX = 0.6
    BOARD_EDGE_Y_MIN = -0.05
    BOARD_TOP_Z_MAX = 0.052
    RELEASE_CLEARANCE = 0.01

    # false if datapoint is over the board
    OFFBOARD_CONDITION = lambda data, i: data[i,1] > BOARD_EDGE_Y_MAX or data[i,1] < BOARD_EDGE_Y_MIN

    # true if datapoint is over the board
    ONBOARD_CONDITION = lambda data, i: not OFFBOARD_CONDITION(data[:,1:4], i)

    # true if datapoint is over the board and has acceleration (-g)
    ONBOARD_FALLING_CONDITION = lambda data, i: (not OFFBOARD_CONDITION(data[:,1:4], i)) \
                                and is_accelerating_g(data[i:,:], FALLING_H) \
                                and get_min_corner_height(data[i:,1:8], RELEASE_H) > RELEASE_CLEARANCE

    # true if block moves significantly over horizon H
    MOVING_CONDITION = lambda data, i: maxDeltaOverHorizon(data[i:,:], TOSS_BALL_H) > TOSS_BALL_TOLERANCE

    # true if block is flat on board
    FLAT_CONDITION = lambda data, i: maxZOverHorizon(data[i:,:], FALL_BALL_H) < BOARD_TOP_Z_MAX

    # false if block moves significantly over horizon H
    STOPPED_CONDITION = lambda data, i: not MOVING_CONDITION(data, i) and FLAT_CONDITION(data,i)

    def is_accelerating_g(data, H):
        accels = data[:H,16]
        all_bound = np.all(accels < FALL_ACC_MAX)

        #return np.all(accels > FALL_ACC_MIN) and np.all(accels < FALL_ACC_MAX)
        mean_bound = np.abs(np.mean(accels) - G) < FALL_TOL_MEAN
        return all_bound and mean_bound

    def get_min_corner_height(data, H):
        return get_min_corner_heights(data[:H,:]).min()

    def get_min_corner_heights(q):
        vertices = BLOCK_HALF_WIDTH * np.array([[-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
                                                [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
                                                [1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0]])
        #pdb.set_trace()
        rots = Rotation.from_quat(q[:,3:])
        phis = np.zeros((q.shape[0],8))
        z = q[:,2]
        for i in range(8):
            p_i = vertices[:,i]
            p_w = rots.apply(p_i)
            phis[:,i] = p_w[:,2] + z
        return phis.min(axis=1)


    # returns maximum ||\delta x|\ over a horizon of H in data
    def maxDeltaOverHorizon(data, H):
        diff_data = data[1:H,:] - data[0:(H-1),:]
        return np.max(np.linalg.norm(diff_data,axis=1))

    def maxZOverHorizon(data, H):
        z_data = data[1:H,2]
        return np.max(z_data)

    def minZOverHorizon(data, H):
        z_data = data[1:H,2]
        return np.min(z_data)

    # find first index of data for which condition() returns true
    def get_first_true(data,condition,H):
        for i in range(0, data.shape[0] - (H-1)):
            if condition(data, i):
                return i
        return -1

    # finds first toss in data, return (-1, 0) if no experiment found
    def get_first_experiment(data):
        removal_time = get_first_true(data[:,1:4],OFFBOARD_CONDITION,1)
        if removal_time < 0:
            return (-1,0)
        start_time = get_first_true(data[removal_time:,:],ONBOARD_FALLING_CONDITION,1) + removal_time
        if start_time < removal_time:
            return (-1,0)
        stopped_time = get_first_true(data[start_time:,1:4],STOPPED_CONDITION,TOSS_BALL_H) + start_time
        if stopped_time < start_time:
            return (-1,0)
        end_time = stopped_time + TOSS_BALL_H
        return (start_time,end_time)

    # permutation of get_first_experiment to start at index start
    def get_first_experiment_after(start, data):
        (s,e) = get_first_experiment(data[start:,:])
        return (s + start, e + start)

    def extract_experiments(data):
        starts = []
        ends = []
        s_last = 0

        # loop through data and extract experiment indices
        while True:
            # find next experiment
            (s,e) = get_first_experiment_after(s_last, data)

            # break if no remaining expeiment found
            if s < s_last:
                break
            s_last = e
            starts = starts + [s]
            ends = ends + [e]

        return (starts,ends)

    def rotvecfix(rv):
        for i in range(rv.shape[0]-1):
            rvi = rv[i,:]
            rvip1 = rv[i+1,:]
            theta = np.linalg.norm(rvip1)
            if theta > 0.0:
                rnew = rvip1*(1 - 2*math.pi/theta)
                if np.linalg.norm(rvi - rnew) < np.linalg.norm(rvi - rvip1):
                    rv[i+1,:] = rnew
        return rv

    # run apriltag_csv.py to convert rosbag to csv file
    datestr = str(int(time.time()))
    csvfn = datestr + '_temp.csv'

    if use_python3:
        cmd = 'python ' + dirs.processing_path('apriltag_csv.py') + ' ' + inbag + ' ' + csvfn
    else:
        cmd = 'export PYTHONPATH=/opt/ros/melodic/lib/python2.7/dist-packages;' + \
            'python2 ' + dirs.processing_path('apriltag_csv.py') + ' ' + inbag + ' ' + csvfn

    print(cmd)
    os.system(cmd)

    # get number of datapoints
    num_t = 0
    with open(csvfn,'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        num_t = sum(1 for row in plots)

    # read CSV trajectory
    t = np.zeros(num_t)
    p_t = np.zeros((3,num_t))
    q_t = np.zeros((4,num_t))
    bp_t = np.zeros((3,num_t))
    bq_t = np.zeros((4,num_t))
    i = 0
    with open(csvfn,'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for (i, row) in enumerate(plots):
            row = np.asarray([float(e) for e in row])
            t[i] = row[0]
            q_t[:,i] = row[1:5]
            p_t[:,i] = row[5:8]
            bq_t[:,i] = row[8:12]
            bp_t[:,i] = row[12:15]
    if DOWNSAMP > 1:
        t = t[::DOWNSAMP]
        q_t = q_t[:,::DOWNSAMP]
        p_t = p_t[:,::DOWNSAMP]
        bq_t = bq_t[:,::DOWNSAMP]
        bp_t = bp_t[:,::DOWNSAMP]


    # clean up temporary CSV
    os.system('rm ./' + csvfn)

    # assume orientation of plate is constant
    Rot_board = Rotation.from_quat(bq_t.T)
    #pdb.set_trace()

    if perturb:
        xrot = Rotation.from_rotvec(math.pi * (randrange(4)) / 2 * np.array([1, 0, 0]))
        yrot = Rotation.from_rotvec(math.pi * (randrange(4)) / 2 * np.array([0, 1, 0]))
        #zrot = Rotation.from_rotvec(math.pi * (randrange(4)) / 2 * np.array([0, 0, 1]))

        #rot_t = Rot_board.inv() * Rotation.from_quat(q_t.T) * xrot * yrot * zrot
        # rot_t = Rot_board.inv() * Rotation.from_quat(q_t.T) * xrot * yrot
        rot_t = Rot_board.inv() * Rotation.from_quat(q_t.T) * xrot
        # rot_t = Rot_board.inv() * Rotation.from_quat(q_t.T)
    else:
        rot_t = Rot_board.inv() * Rotation.from_quat(q_t.T)


    p_t =  Rot_board.inv().apply(p_t.T - bp_t.T).T

    fs = 148.

    # Cut-off frequency of angular velocity filter. < fs/2 (Nyquist)
    fc_p = 60.

    filter_p = False
    if filter_p:
        # filter angular velocity
        w_p = np.clip((fc_p / (fs / 2)), a_min = 0.000001, a_max = 0.999999) # Normalize the frequency
        b, a = signal.butter(1, w_p, 'low')
        for i in range(3):
            #p_t[i,:] = signal.filtfilt(b, a, p_t[i,:],padtype='odd',padlen=100)
            p_t[i,:] = signal.medfilt(p_t[i,:],kernel_size=3)


    if True:
        rvecs = rotvecfix(rot_t.as_rotvec()).T
        #print(rvecs.shape)

        for i in range(3):
            rvecs[i,:] = signal.medfilt(rvecs[i,:],kernel_size=3)
        rot_t = rot_t.from_rotvec(rvecs.T)



    t_start = t[0]
    t = t-t_start
    # calculate derivatives with spline interpolation
    rspline = RotationSpline(t, rot_t)
    pspline = CubicSpline(t, p_t.T)

    #pdb.set_trace()
    #quat_t = rspline(t).as_quat().T
    quat_t = rot_t.as_quat().T
    #quat_t[-1,:] *= -1

    #pdb.set_trace()
    pdiff = p_t[:,1:] - p_t[:,:-1]
    tdiff = np.tile((t[1:] - t[:-1]).reshape([1,-1]), [3,1])
    dp_t = pdiff / tdiff
    # repeat FIRST column so that \delta p = v' \delta t
    dp_t = np.hstack((dp_t[:,[0]],dp_t))
    #dp_t = pspline(t, 1).T


    w_t_old = rspline(t, 1).T
    if True:
        rot_rel = rot_t[:-1].inv() * rot_t[1:]
        rel_vecs = rot_rel.as_rotvec()
        w_t = rel_vecs.T / tdiff
        # repeat FIRST column so that \delta q = w' \delta t
        w_t = np.hstack((w_t[:,[0]],w_t))
    else:
        w_t = w_t_old

    # butterworth filter of order 2 to smooth velocity states

    # sampling frequency
    fs = 148.

    # Cut-off frequency of angular velocity filter. < fs/2 (Nyquist)
    fc_w = 60.

    # Cut-off frequency of linear velocity filter. < fs/2 (Nyquist)
    fc_v = 45.

    # Cut-off frequency of linear accel filter. < fs/2 (Nyquist)
    fc_a = 45.

    filter_avel = True
    if filter_avel:
        # filter angular velocity
        w_w = np.clip((fc_w / (fs / 2)), a_min = 0.000001, a_max = 0.999999) # Normalize the frequency
        b, a = signal.butter(1, w_w, 'low')
        for i in range(3):
            w_t[i,:] = signal.medfilt(w_t[i,:],kernel_size=3)
            #w_t[i,:] = signal.filtfilt(b, a, w_t[i,:],padtype='odd',padlen=100)

    filter_vel = True
    if filter_vel:
        # filter linear velocity
        w_v = np.clip((fc_v / (fs / 2)), a_min = 0.000001, a_max = 0.999999) # Normalize the frequency
        b, a = signal.butter(1, w_v, 'low')
        for i in range(3):
            #dp_t[i,:] = signal.filtfilt(b, a, dp_t[i,:],padtype='odd',padlen=100)
            dp_t[i,:] = signal.medfilt(dp_t[i,:],kernel_size=3)



    if True:
        # delta v = v' - v
        dpdiff = dp_t[:,1:] - dp_t[:,:-1]
        ddp_t = dpdiff / tdiff
        # repeat first column because want to compare \delta v with corner height of q'
        ddp_t = np.hstack((ddp_t[:,[0]],ddp_t))
    else:
        ddp_t = pspline(t, 2).T

    filter_acc = False
    if filter_acc:

        # filter linear velocity
        w_a = np.clip((fc_a / (fs / 2)), a_min = 0.000001, a_max = 0.999999) # Normalize the frequency
        b, a = signal.butter(1, w_a, 'low')
        for i in range(3):
            ddp_t[i,:] = signal.medfilt(ddp_t[i,:],kernel_size=3)
            #ddp_t[i,:] = signal.filtfilt(b, a, ddp_t[i,:],padtype='odd',padlen=100)


    # package data into matrix
    data = np.concatenate((np.expand_dims(t, axis=0), p_t, quat_t, dp_t, w_t, ddp_t), axis=0)
    data = data.T

    # get indices bounding experiment trials
    (starts, ends) = extract_experiments(data)

    # throw out bad tosses
    T_MIN = .3
    N_MIN = T_MIN*148/DOWNSAMP
    CLEARANCE_MIN_MIN = -.007
    CLEARANCE_MIN_MAX = 0.007
    CLEARANCE_AIRFORCE = 0.01
    AIRFORCE_MAX = 9.81/2
    throw_out: List[int] = []
    contact_forces = ddp_t.T - np.array([0,0,G])
    contact_forces = np.sum(np.abs(contact_forces)**2,axis=-1)**(1./2)
    configs = data[:,1:8]
    clearances = get_min_corner_heights(configs)
    NTOSS = len(starts)
    for i in range(NTOSS):
        i = NTOSS - 1 - i
        too_short = ends[i] - starts[i] < N_MIN
        corner_height = clearances[starts[i]:ends[i]]
        lowest = np.amin(corner_height)
        bad_contact = lowest < CLEARANCE_MIN_MIN or lowest > CLEARANCE_MIN_MAX
        toss_forces = contact_forces[starts[i]:ends[i]]
        air_forces = toss_forces[corner_height > CLEARANCE_AIRFORCE]
        max_air_force = np.amax(air_forces)
        #print(max_air_force)
        forces_in_air = max_air_force > AIRFORCE_MAX
        #pdb.set_trace()

        if too_short or bad_contact or forces_in_air:
            throw_out = throw_out + [i]
            ends.pop(i)
            starts.pop(i)
    if len(throw_out) > 0:
        print('discarded tosses:')
        print(throw_out)

    if center:
        pos_init = p_t[0:2, 0:1]
        pos_init = np.concatenate((pos_init, np.zeros((1, 1))), axis=0)
        pos_init = np.tile(pos_init, (1, p_t.shape[1]))

        p_t -= pos_init

        if pullback:
            pullback_mat = np.zeros_like(p_t)
            pullback_mat[1, :] = -0.5
            p_t += pullback_mat


    if zrot:
        # RANDOM ROTATION
        # board_zrot = Rotation.from_rotvec(np.random.normal(0.0, scale=0.3) * np.array([0, 0, 1]))
        board_zrot = Rotation.from_rotvec(np.random.uniform(0, 2 * math.pi) * np.array([0, 0, 1]))
        p_t = board_zrot.apply(p_t.T).T
        dp_t = board_zrot.apply(dp_t.T).T

        quat_t = (board_zrot * Rotation.from_quat(quat_t.T)).as_quat().T

    # ground is higher in data than real life by about 1mm so static z shift
    #static_ground_shift = 0.001
    static_ground_shift = 0.0
    p_t = p_t + (zshift + static_ground_shift) * np.tile(np.array([[0.0, 0, 1]]).T, (1, p_t.shape[1]))

    output_torch_tensors = True
    if output_torch_tensors:
        #quat_shuffle = np.concatenate((quat_t[1:4, :], quat_t[0:1, :]), axis=0)
        quat_shuffle = np.concatenate((quat_t[3:4, :], quat_t[0:3, :]), axis=0)
        learning_data = np.concatenate((p_t, quat_shuffle, dp_t, w_t, np.zeros((6, p_t.shape[1]))), axis=0).T

        file_n = len(glob.glob1(dirs.out_path('data', 'all'), '*.pt'))

        for i, (start, end) in enumerate(zip(starts, ends)):
            #pdb.set_trace()
            run = learning_data[start:end, :]

            inverse_shift = np.zeros((1, 2))

            # if center:
                # pos_init = run[0:1, 0:2]
                # inverse_shift = inverse_shift + pos_init
                # pos_init = np.concatenate((pos_init, np.zeros((1, 17))), axis=1)
                # pos_init = np.tile(pos_init, (run.shape[0], 1))

                # run = run - pos_init

                # # Pull back run center
                # pullback = np.zeros_like(run)
                # pullback[:, 1] = -0.5
                # run = run + pullback

            # if perturb:
                # # RANDOM TRANSLATION
                # rand_trans = np.random.randn(1, 2) / 10
                # inverse_shift = inverse_shift - rand_trans
                # rand_trans = np.concatenate((rand_trans, np.zeros((1, 17))), axis=1)
                # rand_trans = np.tile(rand_trans, (run.shape[0], 1))

                # run = run + rand_trans

            run_tensor = torch.tensor(run).unsqueeze(0)
            run_tensor[:, :, 0:3] *= length_scale
            run_tensor[:, :, 7:10] *= length_scale
            run_tensor[:, :, 13:16] *= length_scale

            torch.save(run_tensor, dirs.out_path('data', 'all', f'{i + file_n}.pt'))
            #torch.save(torch.tensor(t_start + t[start]), dirs.out_path('data', 'all', f'{i + file_n}.pt.time'))
            #torch.save(torch.tensor(inverse_shift), dirs.out_path('data', 'all', f'{i + file_n}.pt.shift'))

    # add indices to data matrix for CSV saving
    start_vect = 0*t
    end_vect = 0*t
    for i in range(len(starts)):
        start_vect[i] = starts[i]
        end_vect[i] = ends[i]

    data = np.concatenate((np.expand_dims(start_vect, axis=0), \
                           np.expand_dims(end_vect, axis=0), \
                           np.expand_dims(t, axis=0), \
                           p_t, quat_t, dp_t, w_t), axis=0)

    if plot:
        if True:
            plt.figure(3)
            plt.plot(t, p_t.T)
            #plt.plot(t, .01*(ddp_t[2,:].T - G))
            plt.plot(t, contact_forces*.01)
            plt.plot(t, clearances)
            plt.plot(t, (clearances > CLEARANCE_AIRFORCE) * 0.1)
            for i in range(3):
                plt.scatter(t[starts],p_t[i,starts].T)
            for i in range(3):
                plt.scatter(t[ends],p_t[i,ends].T)
            #plt.legend(['x','y','z','zddot + g '])
            plt.legend(['x','y','z','F ','phi_min'])
            #plt.show()

        if True:
            plt.figure(4)
            plt.plot(t, dp_t.T)
            for i in range(3):
                plt.scatter(t[starts],dp_t[i,starts].T)
            for i in range(3):
                plt.scatter(t[ends],dp_t[i,ends].T)
            plt.legend(['xdot','ydot','zdot'])
            #plt.show()

        if True:
            plt.figure(5)
            plt.plot(t, w_t.T)
            #plt.plot(t, w_t_old.T)

            for i in range(3):
                plt.scatter(t[starts],w_t[i,starts].T)
            for i in range(3):
                plt.scatter(t[ends],w_t[i,ends].T)
            #plt.legend(['wx','wy','wz','wx_old','wy_old','wz_old'])
            plt.legend(['wx','wy','wz'])#,'wx_old','wy_old','wz_old'])
            #plt.show()

        if True:
            plt.figure(7)
            av = rot_t.as_rotvec().T
            theta = np.sum(np.abs(av.T)**2,axis=-1)**(1./2)
            #av[theta >= math.pi/4, :] = -av[theta >= math.pi/4, :]
            #print(av.shape)
            av = rotvecfix(av.T).T
            plt.plot(t, av.T)
            plt.plot(t,np.sum(np.abs(av.T)**2,axis=-1)**(1./2))
            #plt.plot(t, w_t_old.T)

            for i in range(3):
                plt.scatter(t[starts],av[i,starts].T)
            for i in range(3):
                plt.scatter(t[ends],av[i,ends].T)
            #plt.legend(['wx','wy','wz','wx_old','wy_old','wz_old'])
            plt.legend(['rx','ry','rz','theta'])#,'wx_old','wy_old','wz_old'])
            plt.show()

            # plotting
        if False:
            # get first toss
            #pdb.set_trace()
            TOSS = toss_comp

            t = t[starts[TOSS]:ends[TOSS]]
            t = t - t[0]
            p_t = p_t[:,starts[TOSS]:ends[TOSS]]
            dp_t = dp_t[:,starts[TOSS]:ends[TOSS]]
            #ddp_t = ddp_t[:,starts[TOSS]:ends[TOSS]]
            z = p_t[2,:].T
            dz = dp_t[2,:].T
            #ddz = ddp_t[2,:].T
            #pdb.set_trace()
            z_grav = np.maximum(z[0] - 9.81 * 0.5 * (t ** 2) + dz[0] * t, 0.04)
            #pdb.set_trace()
            v_grav = dz[0] - 9.81 * t
            dt = t[1:] - t[:-1]
            dz_eul = np.cumsum(dt * v_grav[1:])
            z_grav_eul = z_grav + 0.0
            z_grav_eul[1:] = np.maximum(dz_eul + z_grav_eul[0],0.04)


            plt.figure(6)
            plt.plot(t, z)

            #pdb.set_trace()
            plt.plot(t, z_grav)
            plt.plot(t, z_grav_eul)
            #plt.plot(t, ddz - G)
            plt.legend(['z data','gravity exact','gravity implicit euler'])
            plt.figure(7)

            plt.plot(t, dz)
            plt.plot(t, np.maximum(v_grav, -1.))

            #for i in range(3):
            #    plt.scatter(t[starts[0]],p_t[i,starts[0]].T)
            #for i in range(3):
            #    plt.scatter(t[ends[0]],p_t[i,ends[0]].T)

            plt.show()
    # return num accepted, num rejected
    return (len(starts), NTOSS - len(starts))

def setup_directories() -> None:
    file_utils.create_empty_directory(dirs.out_path('data', 'all'))
    file_utils.clear_directory(dirs.out_path('data', 'train'))
    file_utils.clear_directory(dirs.out_path('data', 'valid'))
    file_utils.clear_directory(dirs.out_path('data', 'test'))

def write_experiment(length_scale: float, run_n: int, downsample: int) -> None:
    def block_params():
        # Block measures 4 inches by 4 inches loose construction
        vertices = length_scale * BLOCK_HALF_WIDTH * \
                    torch.tensor([[-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
                                  [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
                                  [1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0]]).t()

        bp = Block3DParams(vertices = vertices,
                           mass = torch.tensor(0.37),
                           inertia = torch.tensor(0.00081) * (length_scale ** 2),
                           mu = torch.tensor(0.18),
                           g = torch.tensor(9.81) * length_scale,
                           dt = torch.tensor(downsample / 148.0),
                           restitution = torch.tensor(0.125))
        bp.run_n = run_n

        return bp

    bp = block_params()

    file_utils.ensure_created_directory(dirs.out_path('params'))

    with open(dirs.out_path('params', 'experiment.pickle'), 'wb') as file:
        pickle.dump(bp, file)

@click.group()
def cli():
    pass

@cli.command('single')
@click.argument('bagnum')
@click.option('--downsample', default=1)
@click.option('--center/--no-center', default=False)
@click.option('--perturb/--no-perturb', default=False)
@click.option('--zrot/--no-zrot', default=False)
@click.option('--pullback/--no-pullback', default=False)
@click.option('--use_python3/--use_python2', default=False)
@click.option('--toss_comp', default=0)
@click.option('--zshift', default=0.0)
@click.option('--length_scale', default=1 / BLOCK_HALF_WIDTH)
def process_cmd(bagnum: int, downsample: int, center: bool, perturb: bool, zrot: bool, pullback: bool, use_python3: bool,
            toss_comp: int, zshift: float, length_scale: float) -> None:

    bag = dirs.data_path('tosses_odom', str(bagnum), 'odom.bag')
    torch.set_default_tensor_type(torch.DoubleTensor)
    setup_directories()
    write_experiment(length_scale, 5, downsample)

    do_process(bag, downsample, center, perturb, zrot, pullback, toss_comp, zshift, length_scale, use_python3, True)

def do_process_multi(num: int, downsample: int, center: bool, perturb: bool, zrot: bool, pullback: bool, use_python3: bool, plot: bool,
                     toss_comp: int, zshift: float, length_scale: float) -> None:
    torch.set_default_tensor_type(torch.DoubleTensor)
    setup_directories()


    accept_n = 0
    reject_n = 0
    while accept_n < num:
        toss_groups = os.listdir(dirs.data_path('tosses_odom'))
        toss_groups = [fold for fold in toss_groups if 'DS' not in fold]
        random.shuffle(toss_groups)

        for toss_group in toss_groups:
            bag = dirs.data_path('tosses_odom', toss_group, 'odom.bag')
            (A,R) = do_process(bag, downsample, center, perturb, zrot, pullback, toss_comp, zshift, length_scale, use_python3, plot)
            accept_n += A
            reject_n += R
            if accept_n > num: break

    for n in range(num, accept_n):
        print('Removing toss: ', n)
        os.remove(dirs.out_path('data', 'all', f'{n}.pt'))

    print('Number of accepted tosses: ', accept_n)
    print('Number of rejected tosses: ', reject_n)
    print('Removed tosses: ', accept_n - num)
    write_experiment(length_scale, num, downsample)

@cli.command('multi')
@click.argument('num', type=int)
@click.option('--downsample', default=1)
@click.option('--center/--no-center', default=False)
@click.option('--perturb/--no-perturb', default=False)
@click.option('--zrot/--no-zrot', default=False)
@click.option('--pullback/--no-pullback', default=False)
@click.option('--use_python3/--use_python2', default=False)
@click.option('--plot/--no_plot', default=False)
@click.option('--toss_comp', default=0)
@click.option('--zshift', default=0.0)
@click.option('--length_scale', default=1 / BLOCK_HALF_WIDTH)
def process_multi_cmd(num: int, downsample: int, center: bool, perturb: bool, zrot: bool, pullback: bool, use_python3: bool, plot: bool,
                      toss_comp: int, zshift: float, length_scale: float) -> None:
    do_process_multi(num, downsample, center, perturb, zrot, pullback, use_python3, plot, toss_comp, zshift, length_scale)

if __name__ == "__main__": cli()
