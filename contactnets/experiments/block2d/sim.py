from dataclasses import dataclass, field
from typing import List, Tuple, cast

import torch
from torch import Tensor

from contactnets.entity import Dynamic2D, Dynamic2DParams, Ground2D
from contactnets.interaction import LCP, PolyGround2D
from contactnets.system import SimResult, System, SystemParams
from contactnets.vis import Visualizer2D

torch.set_default_tensor_type(torch.DoubleTensor)


def vertices_factory():
    return torch.tensor([[1.4, 1.3, -0.8, -1], [-1, 1.2, 1.15, -1.14]]).t()


@dataclass
class Block2DParams:
    vertices: Tensor = field(default_factory=vertices_factory)
    dt: Tensor = field(default_factory=lambda: torch.tensor(0.02))
    g: Tensor = field(default_factory=lambda: torch.tensor(10.0))
    mu: Tensor = field(default_factory=lambda: torch.tensor(1.0))
    mass: Tensor = field(default_factory=lambda: torch.tensor(1.0))
    inertia: Tensor = field(default_factory=lambda: torch.tensor(1.0))
    run_n: int = 0
    step_n: int = 0


def create_empty_system(bp: Block2DParams = Block2DParams()) -> System:
    return create_system(torch.zeros(3), torch.zeros(3), bp=bp)


def create_system(configuration: Tensor, velocity: Tensor, step_n=100,
                  bp: Block2DParams = Block2DParams()) -> System:
    configuration = configuration.reshape(1, 3, 1)
    velocity = velocity.reshape(1, 3, 1)

    def create_entities() -> Tuple[Dynamic2D, Ground2D]:
        params = Dynamic2DParams(mass=bp.mass, inertia=bp.inertia)

        dynamic = Dynamic2D(configuration, velocity, params)

        ground = Ground2D(batch_n=1, angle=torch.tensor(0.0))

        return dynamic, ground

    def create_interaction(dynamic: Dynamic2D, ground: Ground2D) -> LCP:
        poly_ground_2d = PolyGround2D(dynamic, ground, bp.vertices, bp.mu)

        G_bases = Tensor([1.0, -1.0])

        return LCP([poly_ground_2d], G_bases)

    sp = SystemParams(bp.dt, bp.g)

    dynamic, ground = create_entities()
    lcp = create_interaction(dynamic, ground)
    system = System([dynamic, ground], lcp, sp)

    return system


def create_controls(control: Tensor, step_n=100) -> List[List[Tensor]]:
    # Repeat control for the Dynamic2D entity, empty vector for ground
    controls = [[control.reshape(1, 3, 1), torch.zeros(1, 0, 1)] for _ in range(step_n)]

    return controls


def falling_1() -> Tuple[System, List[List[Tensor]]]:
    system = create_system(Tensor([0, 2, 0.5]), Tensor([1, 2.5, 0.5]))
    controls = create_controls(torch.tensor([0.0, 0.0, 0.0]))
    return system, controls


def sim() -> SimResult:
    system, controls = falling_1()
    lcp = system.resolver

    result = system.sim(controls)

    vis = Visualizer2D([result], [lcp.interactions[0].geometry],
                       [cast(Ground2D, system.entities[1])], system.params)
    vis.render()

    return result


def main():
    sim()


if __name__ == "__main__": main()
