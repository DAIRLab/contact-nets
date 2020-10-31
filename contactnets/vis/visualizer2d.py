import torch
from torch import Tensor
from torch.nn import Module

import sys
import subprocess
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
if 'DISPLAY' not in os.environ:
    os.environ['SDL_VIDEODRIVER'] = 'dummy'
import pygame
import pygame.camera

import shutil

import math
import itertools

from contactnets.utils import file_utils, utils, dirs
from contactnets.system import SystemParams, SimResult
from contactnets.interaction import PolyGeometry2D

from typing import *
import pdb


class Visualizer2D:
    sp: SystemParams
    results: List[SimResult]
    geometries: List[PolyGeometry2D] 
    colors: List[Tuple[int, int, int]]

    def __init__(self, results: List[SimResult], geometries: List[PolyGeometry2D],
                       sp: SystemParams) -> None:
        # Can have multiple results, all share a list of geometries
        self.results = results
        self.geometries = geometries
        self.sp = sp 
        
        # Colors for each sim, for all geometries
        self.colors = [(255, 0, 0),
                       (0, 0, 255),
                       (0, 255, 0)]
    
    def render(self, save_video=False, video_framerate=30) -> Optional[str]:
        cam_x, cam_y, cam_zoom = 0, 0, 80.0
        
        result_n = len(self.results)
        obj_n = len(self.geometries)
        step_n = len(self.results[0].configurations[0])
        
        #poly_scale = 120 / math.sqrt(torch.max(torch.abs(xs_list[0][:, 0:2])).item())
        screen_size = (1280, 960)

        if save_video:
            file_utils.create_empty_directory(dirs.out_path('snaps'))

        pygame.init()
        pygame.display.init()
        screen = pygame.display.set_mode(screen_size, 0, 32)

        ground_surface = pygame.Surface(screen_size, pygame.SRCALPHA)
        ground_surface.set_alpha(220)
        surfaces = [[pygame.Surface(screen_size, pygame.SRCALPHA) for _ in range(obj_n)] for _ in range(result_n)]
        all_surfaces = [item for sublist in surfaces for item in sublist]
        all_surfaces.append(ground_surface) 
        
        clock = pygame.time.Clock()
        
        for i in range(step_n):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
            
            screen.fill((220, 220, 220, 0))
            for surface in all_surfaces:
                surface.fill((220, 220, 220, 0))
            
            for result, result_color, result_surfaces in \
                        zip(self.results, self.colors, surfaces):
                for j in range(obj_n):
                    configuration = result.configurations[j][i]
                    vertices = self.geometries[j].vertices
                    vertices_trans = utils.transform_vertices_2d(configuration,
                            vertices * 1).squeeze(0)

                    vertex_n = vertices.shape[1]
                    
                    cam_shift = torch.cat((torch.ones(1, vertex_n) * cam_x,
                                           torch.ones(1, vertex_n) * cam_y), dim=0)
                    center_shift = torch.cat((torch.ones(1, vertex_n) * screen_size[0] / 2,
                                              torch.ones(1, vertex_n) * screen_size[1] / 2), dim=0)
                    
                    vertices_trans = vertices_trans - cam_shift
                    vertices_trans = vertices_trans * cam_zoom
                    vertices_trans = vertices_trans + center_shift
                    
                    pygame.draw.polygon(result_surfaces[j], result_color, 
                            self.to_points_tuple(vertices_trans))
            
            pygame.draw.polygon(ground_surface, (70, 70, 70),
                    [(0, 480), (1280, 480), (1280, 0), (0, 0)])
            
            for surface in all_surfaces:
                screen.blit(surface, (0, 0))

            screen.blit(pygame.transform.flip(screen, False, True), (0, 0))
            pygame.display.flip()

            if save_video:
                filename = dirs.out_path('snaps', '%04d.png' % i)
                pygame.image.save(screen, filename)
            
            if not save_video:
                clock.tick(1 / self.sp.dt)

        if save_video:
            file_utils.ensure_created_directory(dirs.out_path('renders'))
            filename = str(file_utils.num_files(dirs.out_path('renders'))) + '.avi'

            os.system("ffmpeg -r {} -f image2 -i {} -y -qscale 0 -s 1280x960 -aspect 4:3 {} >/dev/null 2>&1"\
                    .format(video_framerate, dirs.out_path('snaps', '%04d.png'), 
                                             dirs.out_path('renders', filename)))

            shutil.rmtree(dirs.out_path('snaps'))

            return dirs.out_path('renders', filename) 
        
    def to_points_tuple(self, vertices: Tensor) -> List[Tuple[int, int]]:
        points = []
        for i in range(vertices.shape[1]):
            points.append((vertices[0, i].item(), vertices[1, i].item()))
        return points
