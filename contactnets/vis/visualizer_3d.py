import os
import pdb  # noqa
import shutil
from typing import Any, List, Optional, Tuple

import OpenGL.GL as GL
import OpenGL.GLU as GLU
import glfw
import imageio
import numpy as np
import pygame.time
from torch import Tensor

from contactnets.interaction import PolyGeometry3D
from contactnets.system import SimResult, SystemParams
from contactnets.utils import dirs, file_utils
from contactnets.utils import quaternion as quat
from contactnets.utils import tensor_utils

Position = Tuple[float, float, float]
Rotation = Tuple[float, float, float, float]

Size = Tuple[int, int]
Color = Tuple[float, float, float, float]


class Visualizer3D:
    """A class handling 3D rendering of polytopes.

    Attributes:
        sp: contains system parameters such as time step for rendering.

        results: the various sim results to render. If this is a singleton list, only one
        set of results is drawn. More sim results generally correspond to comparing different
        method; i.e., render ground truth, e2e prediction, and structured prediction.

        geometries: the polytope geometries to render. The length of this list is
        equal to the length of the configurations or velocities attribute for any of the
        results (the number of entities). All sim results share the same geometries for
        rendering.

        colors: the colors for each sim result, used as required. Default colors are red,
        blue, green. So the first sim result (usually ground truth) would be rendered in red,
        and the second (usually predicted trajectory) would be rendered in blue.
    """
    sp: SystemParams
    results: List[SimResult]
    geometries: List[PolyGeometry3D]
    colors: List[Color]

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

    def render(self, box_width=2.0, headless=False,
               save_video=False, video_framerate=30) -> Optional[str]:
        """Render the scene, returning a string to the saved file if save_video is true."""
        obj_n, step_n = len(self.geometries), len(self.results[0].configurations[0])

        screen_size = (900, 900)
        cam_pos = (0.0, 50.0, -10.0)
        cam_rot = (-80.0, 1.0, 0.0, 0.0)

        if save_video: file_utils.create_empty_directory(dirs.out_path('snaps'))

        window = self.setup_window(screen_size, headless)
        self.make_light()
        self.setup_projection(screen_size, cam_pos, cam_rot)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glEnable(GL.GL_BLEND)

        clock = pygame.time.Clock()

        for step in range(step_n):
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            GL.glClearColor(0.0, 0.0, 0.0, 0.0)

            for result, result_color in zip(self.results, self.colors):
                for j in range(obj_n):
                    geometry = self.geometries[j]
                    configuration = result.configurations[j][step]

                    GL.glPushMatrix()
                    self.do_transformation(configuration)
                    self.draw_cube(geometry, result_color)
                    GL.glPopMatrix()

            self.draw_ground()

            if save_video:
                self.save_snap(step, screen_size)
            else:
                glfw.swap_buffers(window)
                clock.tick(1 / (5 * self.sp.dt))

        glfw.destroy_window(window)
        glfw.terminate()

        if save_video:
            return self.make_video(video_framerate)

        return None

    def setup_window(self, screen_size: Size, headless: bool) -> Any:
        # Returns window, not sure how to type annotate that
        glfw.init()
        if headless: glfw.window_hint(glfw.VISIBLE, False)
        window = glfw.create_window(screen_size[0], screen_size[1], "window", None, None)
        glfw.make_context_current(window)
        return window

    def setup_projection(self, screen_size: Size, cam_pos: Position, cam_rot: Rotation) -> None:
        GL.glEnable(GL.GL_DEPTH_TEST)

        GL.glMatrixMode(GL.GL_PROJECTION)
        GLU.gluPerspective(20, (screen_size[0] / screen_size[1]), 0.1, 100.0)

        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glRotatef(*cam_rot)
        GL.glTranslatef(*cam_pos)

    def make_light(self) -> None:
        light_diffuse = [1.0, 1.0, 1.0, 1.0]
        light_position = [0.0, 0.0, 1.0, 1.0]

        GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, light_diffuse)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, light_position)
        GL.glEnable(GL.GL_LIGHT0)
        GL.glEnable(GL.GL_LIGHTING)

    def do_transformation(self, configuration: Tensor) -> None:
        translation = configuration[:, 0:3, :].clone()
        translation[:, 0:2, :] *= -1
        GL.glTranslatef(*translation.flatten().tolist())

        rot_quat = configuration[:, 3:, 0].clone()
        # Kind of a hack
        rot_quat[:, 3] *= -1

        rot_mat = quat.quaternion_to_rotmat_vec(rot_quat)
        rot_mat = rot_mat.reshape(-1, 3, 3)
        rot_mat = tensor_utils.diag_append(rot_mat, 1.0, 1)
        GL.glMultMatrixf(rot_mat.tolist())

    def draw_cube(self, geometry: PolyGeometry3D, color: Color) -> None:
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
        vertices = geometry.vertices.tolist()

        for i in range(6):
            GL.glBegin(GL.GL_QUADS)

            GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_DIFFUSE, color)

            GL.glNormal3fv(normals[i])
            GL.glVertex3fv(vertices[faces[i][0]])
            GL.glVertex3fv(vertices[faces[i][1]])
            GL.glVertex3fv(vertices[faces[i][2]])
            GL.glVertex3fv(vertices[faces[i][3]])
            GL.glEnd()

    def draw_ground(self) -> None:
        GL.glBegin(GL.GL_QUADS)
        GL.glMaterialfv(GL.GL_FRONT_AND_BACK, GL.GL_DIFFUSE, [0.9, 0.9, 0.9, 0.9])
        GL.glNormal3fv([0.0, 0.0, 1.0])
        GL.glVertex3fv([-5.0, -5.0, 0.0])
        GL.glVertex3fv([5.0, -5.0, 0.0])
        GL.glVertex3fv([5.0, 5.0, 0.0])
        GL.glVertex3fv([-5.0, 5.0, 0.0])
        GL.glEnd()

    def save_snap(self, step: int, screen_size: Size) -> None:
        image_buffer = GL.glReadPixels(0, 0, screen_size[0], screen_size[1],
                                       GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
        image = np.frombuffer(image_buffer, dtype=np.uint8)\
                  .reshape(screen_size[0], screen_size[1], 3)
        image = np.flip(image, 0)

        filename = dirs.out_path('snaps', '%04d.png' % step)
        imageio.imwrite(filename, image)

    def make_video(self, video_framerate: int) -> str:
        file_utils.ensure_created_directory(dirs.out_path('renders'))
        filename = str(file_utils.num_files(dirs.out_path('renders'))) + '.avi'

        snaps = dirs.out_path('snaps', '%04d.png')
        video_out = dirs.out_path('renders', filename)

        os.system(f'ffmpeg -r {video_framerate} -f image2 -i {snaps} -y -qscale 0 '
                  f'-s 1280x960 -aspect 4:3 {video_out} >/dev/null 2>&1')

        shutil.rmtree(dirs.out_path('snaps'))

        return dirs.out_path('renders', filename)
