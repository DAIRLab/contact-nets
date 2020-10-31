import torch
from torch import Tensor
from torch.nn import Module

from vpython import *

import os

import math

from contactnets.utils import utils, dirs
from contactnets.utils import quaternion as quat
from contactnets.system import SystemParams, SimResult
from contactnets.interaction import PolyGeometry3D
import time

from typing import *
import pdb

class Visualizer3D:
    result: SimResult
    sp: SystemParams
    geometries: List[PolyGeometry3D] 
    colors: List[color]

    def __init__(self, result: List[SimResult],
                    geometries: List[PolyGeometry3D],
                    sp: SystemParams) -> None:
        self.result = result[0]
        self.sp = sp 
        self.geometries = geometries

        self.colors = [color.red, color.blue, color.green]
    
    def render(self, save_video=False, box_width = 2.0) -> None:
        num_objects = len(self.geometries)
        num_time_steps = len(self.result.configurations[0])

        center = self.vpython_vector(self.result.configurations[0][0][0, 0:3, 0])
        canvas(width=1200,height=1200,center=center,background=color.white)


        #wall=box(pos=vector(0,1,0),size=vector(0.2,3,2),color=color.green)
        floor=box(pos=vector(0,0,-0.005),size=vector(box_width * 20, box_width * 20, 0.01),color=color.black, opacity=0.5)
        
        box_size = vector(box_width, box_width, box_width)
        boxes = []
        corners = []
        # Angular velocity arrows
        arrows = []

        for i in range(num_objects):
            x = self.result.configurations[i][0][0, :, 0]
            v = self.result.velocities[i][0][0, :, 0]

            vertices = self.geometries[i].vertices
            points_pos = self.points_transform(self.result.configurations[i][0], vertices)

            axis, up = self.axis_up(x)
            boxes.append(box(pos=self.vpython_vector(x[0:3]), axis=self.vpython_vector(axis),
                up=self.vpython_vector(up), size=box_size, color=self.colors[i], opacity=0.5))

            corners.append(points(pos=points_pos, radius=8, color=self.colors[i]))
            arrows.append(arrow(pos=self.vpython_vector(x[0:3]), axis=self.vpython_vector(v[3:6]) / 10, shaftwidth=0.1))

        time.sleep(2)

        # Num replays
        for rep in range(15):
            for i in range(num_time_steps):
                sleep(self.sp.dt)# * 400)

                for j, corner_obj in enumerate(corners):
                    x = self.result.configurations[j][i][0, :, 0]
                    v = self.result.velocities[j][i][0, :, 0]
                    
                    vertices = self.geometries[j].vertices

                    axis, up = self.axis_up(x)
                    boxes[j].pos = self.vpython_vector(x[0:3]) 
                    boxes[j].axis = self.vpython_vector(axis)
                    boxes[j].up = self.vpython_vector(up)
                    boxes[j].size = box_size 

                    points_pos = self.points_transform(self.result.configurations[j][i], vertices)
                    corner_obj.clear()
                    corner_obj.append(points_pos)

                    arrows[j].pos = self.vpython_vector(x[0:3])
                    #arrows[j].axis = self.vpython_vector(v[3:6])
                    # arrows[j].axis = self.vpython_vector(v[3:6]) / 10
                    if i > 1:
                        vdiff = v - self.result.velocities[j][i-1][0,:,0]
                    arrows[j].axis = self.vpython_vector(v[0:3])
                    if rep == 0:
                        #pdb.set_trace()
                        print(x)

    def points_transform(self, configuration: Tensor, vertices: Tensor) -> List[vector]:
        vertices = utils.transform_vertices_3d(configuration, vertices.unsqueeze(0))
        points_list = []
        for vert in vertices.squeeze(0):
            points_list.append(vector(vert[0], vert[1], vert[2]))
        return points_list
        
    def vpython_vector(self, x: Tensor) -> vector:
        return vector(x[0], x[1], x[2])

    def axis_up(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        axis = quat.qrot(x[3:7], torch.tensor([1.0, 0, 0]))
        up = quat.qrot(x[3:7], torch.tensor([0.0, 0, 1]))
        return axis, up

