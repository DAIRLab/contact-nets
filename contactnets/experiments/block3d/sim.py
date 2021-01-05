from dataclasses import dataclass, field
import math
import pdb  # noqa
from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor

from contactnets.entity import Dynamic3D, Dynamic3DParams, Ground3D
from contactnets.interaction import LCP, ElasticLCP, InteractionResolver, PolyGround3D
from contactnets.system import SimResult, System, SystemParams
from contactnets.vis import Visualizer3D

torch.set_default_tensor_type(torch.DoubleTensor)


def vertices_factory():
    return torch.tensor([[-1, -1, -1, -1, 1, 1, 1, 1],
                         [-1, -1, 1, 1, -1, -1, 1, 1],
                         [1, -1, -1, 1, 1, -1, -1, 1]]).double().t()


@dataclass
class Block3DParams:
    # Vertices must be in this order for rendering to work
    vertices: Tensor = field(default_factory=vertices_factory)
    dt: Tensor = field(default_factory=lambda: torch.tensor(0.0068))
    # dt: Tensor = field(default_factory=lambda: torch.tensor(0.02))
    # g: Tensor = field(default_factory=lambda: torch.tensor(10.0))
    g: Tensor = field(default_factory=lambda: torch.tensor(200.0))
    # mu: Tensor = field(default_factory=lambda: torch.tensor(0.7))
    mu: Tensor = field(default_factory=lambda: torch.tensor(0.2))
    mass: Tensor = field(default_factory=lambda: torch.tensor(0.3))
    inertia: Tensor = field(default_factory=lambda: torch.tensor(0.3))
    restitution: Tensor = field(default_factory=lambda: torch.tensor(.25))

    run_n: int = 0
    step_n: int = 0


def create_empty_system(bp: Block3DParams = Block3DParams(), elastic=False) -> System:
    return create_system(torch.zeros(7), torch.zeros(6), bp=bp, elastic=elastic)


def create_system(configuration: Tensor, velocity: Tensor, step_n=100,
                  bp: Block3DParams = Block3DParams(),
                  elastic: bool = False, restitute_friction: bool = True) -> System:
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

    def create_interaction(dynamic: Dynamic3D, ground: Ground3D,
                           elastic: bool, restitute_friction: bool) -> InteractionResolver:
        poly_ground_3d = PolyGround3D(dynamic, ground, bp.vertices, bp.mu)

        G_bases = compute_G_bases(16)
        if elastic:
            return ElasticLCP([poly_ground_3d], G_bases, bp.restitution, restitute_friction)
        else:
            return LCP([poly_ground_3d], G_bases)

    sp = SystemParams(bp.dt, bp.g)

    dynamic, ground = create_entities()
    resolver = create_interaction(dynamic, ground, elastic, restitute_friction)
    system = System([dynamic, ground], resolver, sp)

    return system


def create_controls(control: Tensor = torch.zeros(6), step_n=100) -> List[List[Tensor]]:
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

    return result


def main():
    sim()


if __name__ == "__main__": main()
