from dataclasses import dataclass, field
import pdb  # noqa
from typing import List, Tuple, cast

import numpy as np
import torch
from torch import Tensor

from contactnets.entity import Dynamic2D, Dynamic2DParams, Entity, Ground2D
from contactnets.interaction import LCP, BallBin2D
from contactnets.system import SimResult, System, SystemParams
from contactnets.vis import Visualizer2D


@dataclass
class BallBin2DParams:
    radii: Tensor = field(default_factory=lambda: torch.ones(5))
    masses: Tensor = field(default_factory=lambda: torch.ones(5))
    inertias: Tensor = field(default_factory=lambda: torch.ones(5) * (1 / 3))
    ball_mus: Tensor = field(default_factory=lambda: torch.ones(5) * 0.3)

    angles: Tensor = field(default_factory=lambda: torch.tensor([np.pi / 8, -np.pi / 8]))
    heights: Tensor = field(default_factory=lambda: torch.zeros(2))
    wall_mus: Tensor = field(default_factory=lambda: torch.ones(2) * 0.3)

    dt: Tensor = field(default_factory=lambda: torch.tensor(0.01))
    g: Tensor = field(default_factory=lambda: torch.tensor(9.8))

    run_n: int = 0
    step_n: int = 0


def create_empty_system(bp: BallBin2DParams = BallBin2DParams()) -> System:
    configs = []
    velocities = []
    for radius in bp.radii:
        configs += [torch.zeros(1, 3, 1)]
        velocities += [torch.zeros(1, 3, 1)]
    return create_system(configs, velocities, bp=bp)


def create_system(configurations: List[Tensor], velocities: List[Tensor],
                  step_n=100, bp: BallBin2DParams = BallBin2DParams()) -> System:

    def create_entities() -> Tuple[List[Dynamic2D], List[Ground2D]]:
        balls = []
        walls = []
        for (mass, inertia, configuration, velocity) in \
                zip(bp.masses, bp.inertias, configurations, velocities):

            params = Dynamic2DParams(mass = mass, inertia = inertia)

            dynamic = Dynamic2D(configuration.reshape(1, 3, 1),
                                velocity.reshape(1, 3, 1), params)
            balls += [dynamic]

        for (angle, height) in zip(bp.angles, bp.heights):
            wall = Ground2D(batch_n = 1, angle = angle, height = height)
            walls += [wall]

        return balls, walls

    def create_interaction(balls: List[Dynamic2D], walls: List[Ground2D]) -> LCP:
        ball_bin_2d = BallBin2D(balls, walls, bp.radii, bp.ball_mus, bp.wall_mus)

        G_bases = Tensor([1.0, -1.0])

        return LCP([ball_bin_2d], G_bases)

    sp = SystemParams(bp.dt, bp.g)

    balls, walls = create_entities()
    lcp = create_interaction(balls, walls)
    entities = cast(List[Entity], balls) + cast(List[Entity], walls)
    system = System(entities, lcp, sp)

    return system


def create_controls(control: Tensor, step_n=100, ball_n=1, wall_n=1) -> List[List[Tensor]]:
    # Repeat control for the Dynamic2D entity, empty vector for grond
    all_control = [control.reshape(1, 3, 1)] * ball_n + [torch.zeros(1, 0, 1)] * wall_n
    controls = [all_control for _ in range(step_n)]

    return controls


def ball_stack() -> Tuple[BallBin2DParams, System]:
    bp = BallBin2DParams()
    ball_n = bp.radii.numel()
    configs = []
    velocities = []
    for i in range(ball_n):
        configs += [(i + 1) * Tensor([1.0, 2.0, 0.0])]
        velocities += [Tensor([0.0, 0.0, 0.0])]
    return (bp, create_system(configs, velocities, bp=bp))


def sim() -> SimResult:
    torch.set_default_tensor_type(torch.DoubleTensor)

    (bp, system) = ball_stack()
    controls = create_controls(
        torch.zeros(3), ball_n=bp.radii.numel(), wall_n=bp.angles.numel())
    lcp = system.resolver

    result = system.sim(controls)

    vis = Visualizer2D([result], lcp.interactions[0].ball_geometries,
                       lcp.interactions[0].walls, system.params)
    vis.camera.y = 4
    vis.camera.zoom = 40.0
    vis.render()

    return result


def main():
    sim()


if __name__ == "__main__": main()
