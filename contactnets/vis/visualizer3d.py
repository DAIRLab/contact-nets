import torch
from torch import Tensor
from torch.nn import Module

import sys
import subprocess
import os

import shutil

import numpy as np
import math
import itertools

from contactnets.utils import file_utils, tensor_utils, utils, dirs
from contactnets.utils import quaternion as quat
from contactnets.system import SystemParams, SimResult
from contactnets.interaction import PolyGeometry3D

import pygame.time
from OpenGL.GL import *
from OpenGL.GLU import *
import glfw
import imageio

from typing import *
import pdb


class Visualizer3D:
    sp: SystemParams
    results: List[SimResult]
    geometries: List[PolyGeometry3D]
    colors: List[Tuple[float, float, float, float]]

    def __init__(self, results: List[SimResult], geometries: List[PolyGeometry3D],
            sp: SystemParams) -> None:
        # Can have multiple results, all share a list of geometries
        self.results = results
        self.geometries = geometries
        self.sp = sp

        # Colors for each sim result, for all geometries
        self.colors = [(1.0, 0, 0, 0.5),
                       (0, 0, 1.0, 0.5),
                       (0, 1.0, 0, 0.5)]

    def setup_window(self, screen_size: Tuple[int, int], headless):
        # Returns window, not sure how to type annotate that
        glfw.init()
        if headless:
            glfw.window_hint(glfw.VISIBLE, False)
        window = glfw.create_window(screen_size[0], screen_size[1], "window", None, None)
        glfw.make_context_current(window)
        return window

    def setup_projection(self, screen_size: Tuple[int, int],
            cam_pos: Tuple[float, float, float], cam_rot: Tuple[float, float, float, float]):
        glEnable(GL_DEPTH_TEST)

        glMatrixMode(GL_PROJECTION)
        gluPerspective(20, (screen_size[0] / screen_size[1]), 0.1, 100.0)

        glMatrixMode(GL_MODELVIEW)
        glRotatef(*cam_rot)
        glTranslatef(*cam_pos)

    def make_light(self):
        light_diffuse = [1.0, 1.0, 1.0, 1.0]  # Red diffuse light
        light_position = [0.0, 0.0, 1.0, 1.0]

        glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
        glLightfv(GL_LIGHT0, GL_POSITION, light_position)
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHTING)

    def draw_cube(self, geometry: PolyGeometry3D, color: Tuple[float, float, float, float]):
        # Normals for the 6 faces of a cube
        normals = [
            (-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0), (0.0, 0.0, -1.0), (0.0, 0.0, 1.0)
        ]
        # Vertex indices for the 6 faces of a cube
        faces = [
            [0, 1, 2, 3], [3, 2, 6, 7], [7, 6, 5, 4],
            [4, 5, 1, 0], [5, 6, 2, 1], [7, 4, 0, 3]
        ]
        vertices = geometry.vertices.t().tolist()

        for i in range(6):
            glBegin(GL_QUADS)

            glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, color)

            glNormal3fv(normals[i])
            glVertex3fv(vertices[faces[i][0]])
            glVertex3fv(vertices[faces[i][1]])
            glVertex3fv(vertices[faces[i][2]])
            glVertex3fv(vertices[faces[i][3]])
            glEnd()

    def draw_ground(self):
        glBegin(GL_QUADS)
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, [0.9, 0.9, 0.9, 0.9])
        glNormal3fv([0.0, 0.0, 1.0])
        glVertex3fv([-5.0, -5.0, 0.0])
        glVertex3fv([5.0, -5.0, 0.0])
        glVertex3fv([5.0, 5.0, 0.0])
        glVertex3fv([-5.0, 5.0, 0.0])
        glEnd()

    def render(self, box_width=2.0, headless=False,
                     save_video=False, video_framerate=30) -> Optional[str]:
        result_n = len(self.results)
        obj_n = len(self.geometries)
        step_n = len(self.results[0].configurations[0])

        screen_size = (900, 900)
        # cam_pos = (0.0, 20.0, -0.1)
        # cam_rot = (-90.0, 1.0, 0.0, 0.0)
        cam_pos = (0.0, 50.0, -10.0)
        cam_rot = (-80.0, 1.0, 0.0, 0.0)

        if save_video: file_utils.create_empty_directory(dirs.out_path('snaps'))

        window = self.setup_window(screen_size, headless)
        self.make_light()
        self.setup_projection(screen_size, cam_pos, cam_rot)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_BLEND)

        clock = pygame.time.Clock()

        for i in range(step_n):
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glClearColor(0.0, 0.0, 0.0, 0.0)

            for result, result_color in zip(self.results, self.colors):
                for j in range(obj_n):
                    glPushMatrix()
                    geometry = self.geometries[j]

                    configuration = result.configurations[j][i]
                    vertices = geometry.vertices * (box_width / 2.0)

                    translation = configuration[:, 0:3, :].clone()
                    translation[:, 0:2, :] *= -1
                    glTranslatef(*translation.flatten().tolist())

                    rot_quat = configuration[:, 3:, 0].clone()
                    # Kind of a hack
                    rot_quat[:, 3] *= -1

                    rot_mat = quat.quaternion_to_rotmat_vec(rot_quat)
                    rot_mat = rot_mat.reshape(-1, 3, 3)
                    rot_mat = tensor_utils.diag_expand_mat(rot_mat, 1.0, 1)
                    glMultMatrixf(rot_mat.tolist())

                    self.draw_cube(geometry, result_color)
                    glPopMatrix()

            self.draw_ground()

            if save_video:
                image_buffer = glReadPixels(0, 0, screen_size[0], screen_size[1],
                        OpenGL.GL.GL_RGB, OpenGL.GL.GL_UNSIGNED_BYTE)
                image = np.frombuffer(image_buffer, dtype=np.uint8)\
                          .reshape(screen_size[0], screen_size[1], 3)
                image = np.flip(image, 0)

                filename = dirs.out_path('snaps', '%04d.png' % i)
                imageio.imwrite(filename, image)

            if not save_video:
                glfw.swap_buffers(window)
                clock.tick(1 / (5 * self.sp.dt))

        glfw.destroy_window(window)
        glfw.terminate()

        if save_video:
            file_utils.ensure_created_directory(dirs.out_path('renders'))
            filename = str(file_utils.num_files(dirs.out_path('renders'))) + '.avi'
            
            os.system("ffmpeg -r {} -f image2 -i {} -y -qscale 0 -s 1280x960 -aspect 4:3 {} >/dev/null 2>&1"\
                    .format(video_framerate, dirs.out_path('snaps', '%04d.png'),
                                             dirs.out_path('renders', filename)))

            shutil.rmtree(dirs.out_path('snaps'))

            return dirs.out_path('renders', filename)
