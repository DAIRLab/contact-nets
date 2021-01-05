from dataclasses import dataclass
from itertools import chain
import os
import pdb  # noqa
import shutil
from typing import Any, List, Optional, Tuple, Union

import torch
from torch import Tensor

from contactnets.entity import Ground2D
from contactnets.interaction import BallGeometry2D, PolyGeometry2D
from contactnets.system import SimResult, SystemParams
from contactnets.utils import dirs, file_utils, utils

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
if 'DISPLAY' not in os.environ:
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
import pygame  # noqa
from pygame import Surface  # noqa
import pygame.camera  # noqa

Geometry2D = Union[PolyGeometry2D, BallGeometry2D]
Color = Tuple[int, int, int]
Size = Tuple[int, int]


@dataclass
class Camera:
    """A utility struct for capturing information about the scene camera."""
    x: float
    y: float
    zoom: float
    screen_size: Size


def pixelate(x): return int(round(x.item()))


class Visualizer2D:
    """A class handling 2D rendering of both polygon and ball geometries.

    Attributes:
        sp: contains system parameters such as time step for rendering.

        camera: the camera information for the scene.

        results: the various sim results to render. If this is a singleton list, only one
        set of results is drawn. More sim results generally correspond to comparing different
        method; i.e., render ground truth, e2e prediction, and structured prediction.

        geometries: the polygon or ball geometries to render. The length of this list is
        equal to the length of the configurations or velocities attribute for any of the
        results (the number of entities). All sim results share the same geometries for
        rendering.

        grounds: the ground objects to render.

        colors: the colors for each sim result, used as required. Default colors are red,
        blue, green. So the first sim result (usually ground truth) would be rendered in red,
        and the second (usually predicted trajectory) would be rendered in blue.
    """
    sp: SystemParams
    camera: Camera

    results: List[SimResult]
    geometries: List[Geometry2D]
    grounds: List[Ground2D]
    colors: List[Color]

    def __init__(self, results: List[SimResult], geometries: List[Geometry2D],
                 grounds: List[Ground2D], sp: SystemParams) -> None:
        self.results = results
        self.geometries = geometries
        self.grounds = grounds
        self.sp = sp

        self.camera = Camera(x=0.0, y=0.0, zoom=80.0, screen_size=(1280, 960))

        self.colors = [(255, 0, 0),
                       (0, 0, 255),
                       (0, 255, 0)]

    def render(self, save_video=False, video_framerate=30) -> Optional[str]:
        """Render the scene, returning a string to the saved file if save_video is true."""
        result_n, obj_n, ground_n = len(self.results), len(self.geometries), len(self.grounds)
        step_n = len(self.results[0].configurations[0])

        if save_video: file_utils.create_empty_directory(dirs.out_path('snaps'))

        screen: Surface = self.pygame_init()

        ground_surfaces = [self.make_surface(220) for _ in range(ground_n)]
        geometry_surfaces = [[self.make_surface(255) for _ in range(obj_n)]
                                                     for _ in range(result_n)]  # noqa
        all_surfaces = list(chain.from_iterable(geometry_surfaces)) + ground_surfaces

        clock = pygame.time.Clock()

        for step in range(step_n):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None

            self.fill_background(screen, all_surfaces)

            self.render_geometries(geometry_surfaces, step)
            self.render_grounds(ground_surfaces)

            for surface in all_surfaces:
                screen.blit(surface, (0, 0))

            screen.blit(pygame.transform.flip(screen, False, True), (0, 0))
            pygame.display.flip()

            if save_video:
                pygame.image.save(screen, dirs.out_path('snaps', '%04d.png' % step))
            else:
                clock.tick(1 / self.sp.dt)

        if save_video:
            return self.make_video(video_framerate)

        return None

    def pygame_init(self) -> Surface:
        pygame.camera.init()
        return pygame.display.set_mode(self.camera.screen_size, 0, 32)

    def make_surface(self, alpha: int) -> Surface:
        surface = pygame.Surface(self.camera.screen_size, pygame.SRCALPHA)
        surface.set_alpha(alpha)
        return surface

    def fill_background(self, screen: Surface, all_surfaces: List[Surface]) -> None:
        screen.fill((220, 220, 220, 0))
        for surface in all_surfaces:
            surface.fill((220, 220, 220, 0))

    def render_geometries(self, geometry_surfaces: List[Surface], step: int) -> None:
        for result, result_color, result_surfaces in \
                zip(self.results, self.colors, geometry_surfaces):
            for j in range(len(self.geometries)):
                configuration = result.configurations[j][step]
                surface = result_surfaces[j]

                geometry = self.geometries[j]
                if isinstance(geometry, PolyGeometry2D):
                    self.render_polygon(geometry, configuration, surface, result_color)
                elif isinstance(geometry, BallGeometry2D):
                    self.render_ball(geometry, configuration, surface, result_color)
                else:
                    raise Exception('Unrecognized geometry.')

    def render_polygon(self, geometry: PolyGeometry2D, configuration: Tensor,
                       surface: Surface, color: Color) -> None:
        vertices = geometry.vertices

        vertices = utils.transform_vertices_2d(configuration, vertices).squeeze(0)
        vertices = self.do_camera_transform(vertices)

        pygame.draw.polygon(surface, color, self.to_points_tuple(vertices))

    def render_ball(self, geometry: BallGeometry2D, configuration: Tensor,
                    surface: Surface, color: Color) -> None:
        com_pos = configuration[0, :2, 0]
        angle = configuration[0, 2, 0]
        radius = geometry.radius

        out_pos = com_pos + radius * torch.tensor([torch.cos(angle), torch.sin(angle)])

        com_pos = self.do_camera_transform(com_pos.unsqueeze(0)).squeeze(0)
        out_pos = self.do_camera_transform(out_pos.unsqueeze(0)).squeeze(0)

        com_pos_pix = (pixelate(com_pos[0]), pixelate(com_pos[1]))
        out_pos_pix = (pixelate(out_pos[0]), pixelate(out_pos[1]))

        radius = pixelate(radius * self.camera.zoom)

        pygame.draw.circle(surface, color, com_pos_pix, radius)
        darker_color = (color[0] // 2, color[1] // 2, color[2] // 2)
        pygame.draw.line(surface, darker_color, com_pos_pix, out_pos_pix, 3)

    def do_camera_transform(self, points: Tensor) -> Tensor:
        cam_shift = torch.cat((torch.ones(1, 1) * self.camera.x,
                               torch.ones(1, 1) * self.camera.y), dim=1)
        center_shift = torch.cat((torch.ones(1, 1) * self.camera.screen_size[0] / 2,
                                  torch.ones(1, 1) * self.camera.screen_size[1] / 2), dim=1)

        points = points - cam_shift
        points = points * self.camera.zoom
        points = points + center_shift

        return points

    def render_grounds(self, ground_surfaces: List[Surface]) -> None:
        screen_width = self.camera.screen_size[0]
        screen_height = self.camera.screen_size[1]

        for ground_surface, wall in zip(ground_surfaces, self.grounds):
            angle_shift = pixelate(torch.sin(wall.angle) * screen_width / 2)
            height_shift = pixelate(
                (torch.cos(wall.angle) * wall.height - self.camera.y) * self.camera.zoom)
            pygame.draw.polygon(ground_surface, (70, 70, 70),
                                [(0, screen_height / 2 - angle_shift + height_shift),
                                 (screen_width, screen_height / 2 + angle_shift + height_shift),
                                 (screen_width, 0), (0, 0)])

    def to_points_tuple(self, vertices: Tensor) -> List[Tuple[Any, Any]]:
        return [(vertex[0].item(), vertex[1].item()) for vertex in vertices]

    def make_video(self, video_framerate: int) -> str:
        file_utils.ensure_created_directory(dirs.out_path('renders'))
        filename = str(file_utils.num_files(dirs.out_path('renders'))) + '.avi'

        snaps = dirs.out_path('snaps', '%04d.png')
        video_out = dirs.out_path('renders', filename)

        os.system(f'ffmpeg -r {video_framerate} -f image2 -i {snaps} -y -qscale 0 '
                  f'-s 1280x960 -aspect 4:3 {video_out} >/dev/null 2>&1')

        shutil.rmtree(dirs.out_path('snaps'))

        return dirs.out_path('renders', filename)
