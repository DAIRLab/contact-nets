from dataclasses import dataclass
import pdb  # noqa
from typing import List, Tuple, cast

import torch
from torch import Tensor

from contactnets.entity import Dynamic2D, Entity, Ground2D
from contactnets.interaction import Interaction


@dataclass
class BallGeometry2D:
    radius: Tensor


class BallBin2D(Interaction):
    """A frictional 2D interaction between several balls and several walls."""
    balls: List[Dynamic2D]
    walls: List[Ground2D]

    ball_geometries: List[BallGeometry2D]
    ball_mus: Tensor
    wall_mus: Tensor

    def __init__(self, balls: List[Dynamic2D], walls: List[Ground2D],
                 radii: Tensor, ball_mus: Tensor, wall_mus: Tensor) -> None:
        super().__init__(cast(List[Entity], balls) + cast(List[Entity], walls))

        self.balls = balls
        self.walls = walls
        self.ball_geometries = [BallGeometry2D(radius) for radius in radii]
        self.ball_mus = ball_mus
        self.wall_mus = wall_mus

    def ball_n(self) -> int:
        return len(self.balls)

    def wall_n(self) -> int:
        return len(self.walls)

    def compute_phi_ballwall(self, ground: Ground2D, ball_geometry: BallGeometry2D,
                             ball_config: Tensor) -> Tensor:
        surf_trans = self.compute_Jn_ballwall(ground)
        phi = surf_trans.bmm(ball_config).transpose(1, 2)
        return phi - ground.height - ball_geometry.radius

    def compute_Jn_ballwall(self, ground: Ground2D) -> Tensor:
        Jn = torch.cat((-torch.sin(ground.angle).reshape(1, 1, 1),
                        torch.cos(ground.angle).reshape(1, 1, 1),
                        torch.zeros(1, 1, 1)), dim=2)
        return Jn.repeat(self.batch_n(), 1, 1)

    def compute_Jt_tilde_ballwall(self, ground: Ground2D,
                                  ball_geometry: BallGeometry2D, mu: float) -> Tensor:
        Jt_tilde = torch.cat((torch.cos(ground.angle).reshape(1, 1, 1),
                              torch.sin(ground.angle).reshape(1, 1, 1),
                              ball_geometry.radius * torch.ones(1, 1, 1)), dim=2)
        return mu * Jt_tilde.repeat(self.batch_n(), 1, 1)

    def compute_phi_ballball(self, ball1_geometry: BallGeometry2D,
                             ball2_geometry: BallGeometry2D, ball1_config: Tensor,
                             ball2_config: Tensor) -> Tensor:
        d_config = ball1_config - ball2_config
        d_com = d_config[:, :2, :]
        return d_com.norm(dim=1) - ball1_geometry.radius - ball2_geometry.radius

    def compute_ballball_normals(self, ball1_config: Tensor, ball2_config: Tensor) -> Tensor:
        d_config = ball1_config - ball2_config
        d_com = d_config[:, :2, :]
        dists = d_com.norm(dim=1).repeat(1, 2, 1)
        Jnb1 = (d_com / dists).transpose(1, 2)
        return Jnb1

    def compute_Jn_ballball(self, ball1_config: Tensor, ball2_config: Tensor) \
            -> Tuple[Tensor, Tensor]:
        Jnb1 = self.compute_ballball_normals(ball1_config, ball2_config)
        Jnb1 = torch.cat((Jnb1, torch.zeros(Jnb1.shape[0], 1, 1)), dim=2)
        return (Jnb1, -Jnb1)

    def compute_Jt_tilde_ballball(self, ball1_geometry: BallGeometry2D,
                                  ball2_geometry: BallGeometry2D, ball1_config: Tensor,
                                  ball2_config: Tensor, mu: float) -> Tuple[Tensor, Tensor]:
        Jnb1 = self.compute_ballball_normals(ball1_config, ball2_config)
        Jtb1 = mu * torch.cat((Jnb1[:, :, 1:2], -Jnb1[:, :, 0:1],
                               ball1_geometry.radius * torch.ones(Jnb1.shape[0], 1, 1)), dim=2)
        Jtb2 = mu * torch.cat((-Jnb1[:, :, 1:2], Jnb1[:, :, 0:1],
                               ball2_geometry.radius * torch.ones(Jnb1.shape[0], 1, 1)), dim=2)
        return (Jtb1, Jtb2)

    def compute_phi(self, configurations: List[Tensor]) -> Tensor:
        phi = torch.zeros(self.batch_n(), self.contact_n(), 1)
        N = 0
        for i, (ball1, geometry1, config1) in \
                enumerate(zip(self.balls, self.ball_geometries, configurations)):
            for j, (ball2, geometry2, config2) in \
                    enumerate(zip(self.balls[(i + 1):],
                                  self.ball_geometries[(i + 1):],
                                  configurations[(i + 1):])):
                phi[:, N, :] = self.compute_phi_ballball(geometry1, geometry2, config1, config2)
                N += 1
            for j, wall in enumerate(self.walls):
                phi[:, N, :] = self.compute_phi_ballwall(wall, geometry1, config1)
                N += 1
        return phi

    def compute_Jn(self, configurations: List[Tensor]) -> Tensor:
        Jn = torch.zeros(self.batch_n(), self.contact_n(), 3 * self.ball_n())
        N = 0
        for i, (ball1, geometry1, config1) in \
                enumerate(zip(self.balls, self.ball_geometries, configurations)):
            for j, (ball2, geometry2, config2) in \
                    enumerate(zip(self.balls[(i + 1):],
                                  self.ball_geometries[(i + 1):],
                                  configurations[(i + 1):])):
                (Jni, Jnj) = self.compute_Jn_ballball(config1, config2)
                Jn[:, N, self.ball_inds(i)] = Jni
                Jn[:, N, self.ball_inds(i + j + 1)] = Jnj
                N += 1
            for j, wall in enumerate(self.walls):
                Jn[:, N, self.ball_inds(i)] = self.compute_Jn_ballwall(wall)
                N += 1
        return Jn

    def ball_inds(self, i):
        return i * 3 + torch.tensor([0, 1, 2])

    def compute_phi_t(self, configurations: List[Tensor]) -> Tensor:
        pass

    def compute_Jt_tilde(self, configurations: List[Tensor]) -> Tensor:
        Jt_tilde = torch.zeros(self.batch_n(), self.contact_n(), 3 * self.ball_n())
        N = 0
        for i, (ball1, geometry1, config1, mu1) in \
                enumerate(zip(self.balls, self.ball_geometries,
                              configurations, self.ball_mus)):
            for j, (ball2, geometry2, config2, mu2) in \
                    enumerate(zip(self.balls[(i + 1):], self.ball_geometries[(i + 1):],
                                  configurations[(i + 1):], self.ball_mus[(i + 1):])):
                mu12 = 2 * mu1 * mu2 / (mu1 + mu2)
                (Jti, Jtj) = self.compute_Jt_tilde_ballball(geometry1, geometry2,
                                                            config1, config2, mu12)
                Jt_tilde[:, N, self.ball_inds(i)] = Jti
                Jt_tilde[:, N, self.ball_inds(i + j + 1)] = Jtj
                N += 1
            for j, (wall, mu2) in enumerate(zip(self.walls, self.wall_mus)):
                mu12 = 2 * mu1 * mu2 / (mu1 + mu2)
                Jt_tilde[:, N, self.ball_inds(i)] = \
                    self.compute_Jt_tilde_ballwall(wall, geometry1, mu12)
                N += 1
        return Jt_tilde

    def contact_n(self) -> int:
        return (self.ball_n() * (self.ball_n() - 1)) // 2 + self.ball_n() * len(self.walls)
