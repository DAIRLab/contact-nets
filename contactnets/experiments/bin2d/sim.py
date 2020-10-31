import sys
sys.path.append('..')

import torch
from torch import Tensor
from torch.nn import Module
import math

from dataclasses import dataclass, field

from contactnets.system import System, SystemParams, SimResult
from contactnets.entity import Dynamic3D, Dynamic3DParams, Ground3D
from contactnets.vis import Visualizer3D
from contactnets.interaction import PolyGround3D, LCP

from typing import *

import pdb

import numpy as np


torch.set_default_tensor_type(torch.DoubleTensor)

NS = 3
NE = 0
NC = 3
@dataclass
class Bin2DParams:
    # Vertices must be in this order for rendering to work
    circles: Tensor = field(default_factory=lambda:
                       torch.tensor([[1.0],
                                     [1.0],
                                     [0.3]]).double().repeat(1, NS))
    capsules: Tensor = field(default_factory=lambda:
                       torch.tensor([[1.0],
                                     [1.0],
                                     [1.0],
                                     [0.3]]).double().repeat(1, NS))
    ellipses: Tensor = field(default_factory=lambda:
                       torch.tensor([[1.0],
                                     [1.0],
                                     [1.0],
                                     [0.3]]).double().repeat(1, NE))
    #vertices: Tensor = torch.tensor([[-1.1, -1.2, -1.0, -.8, 1.2, .9, 1.1, .9],
    #                                 [-1.0, -1.0, .9, 1.0, -.8, -1.0, 1.0, 1.0],
    #                                 [.9, -1.2, -1.0, 1.1, 1.0, -.8, -.8, 1.2]]).double()

    dt: Tensor = field(default_factory=lambda: torch.tensor(0.02))
    g: Tensor = field(default_factory=lambda: torch.tensor(10.0))

def create_empty_system(bp: Block3DParams = Block3DParams()) -> System:
    return create_system(torch.zeros(7), torch.zeros(6), bp=bp)


def create_system(configuration: Tensor,
                  velocity: Tensor,
                  step_n=100, bp: Block3DParams = Block3DParams()) \
                    -> System:
    #pdb.set_trace()
    configuration = configuration.reshape(1, 7, 1)
    velocity = velocity.reshape(1, 6, 1)

    def create_entities() -> Tuple[Dynamic3D, Ground3D]:
        params = Dynamic3DParams(mass = bp.mass,
                                 inertia = bp.inertia)
        
        dynamic = Dynamic3D(configuration, velocity, params)
        
        ground = Ground3D(batch_n=1) 

        return dynamic, ground

    def compute_G_bases(bases_n):
        bases = torch.zeros(bases_n, 2)
        for i, angle in enumerate(np.linspace(0, 2 * math.pi * (1 - 1 / bases_n), bases_n)):
            bases[i, 0] = math.cos(angle)
            bases[i, 1] = math.sin(angle)

        return bases

    def create_interaction(dynamic: Dynamic3D, ground: Ground3D) -> LCP:
        poly_ground_3d = PolyGround3D(dynamic, ground, bp.vertices, bp.mu)

        G_bases = compute_G_bases(16)

        return LCP([poly_ground_3d], G_bases)

    sp = SystemParams(bp.dt, bp.g)

    dynamic, ground = create_entities()
    lcp = create_interaction(dynamic, ground)
    system = System([dynamic, ground], lcp, sp)

    return system


def create_controls(control: Tensor = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), step_n=100) \
        -> List[List[Tensor]]:
    # Repeat control for the Dynamic2D entity, empty vector for grond
    controls = [[control.reshape(1, 6, 1),
                torch.zeros(1, 0, 1)] for _ in range(step_n)]

    return controls


def side_fall() -> System:
    return create_system(Tensor([0.0, 0.0, 2.05, 0.707, 0, 0.707, 0]),
                         Tensor([0.0, 0.0, -1.0, 0, 0, 0]))


def sim() -> SimResult:
    system = side_fall()
    controls = create_controls()
    lcp = system.resolver
    
    result = system.sim(controls)
    
    vis = Visualizer3D([result], [lcp.interactions[0].geometry], system.params)
    vis.render()


def main():
    sim()

if __name__ == "__main__": main()
