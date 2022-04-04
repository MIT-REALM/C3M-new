import torch
import torch.nn as nn

from abc import ABC, abstractmethod


class Controller(nn.Module, ABC):

    def __init__(
            self,
            goal_point: torch.Tensor,
            u_eq: torch.Tensor,
            state_std: torch.Tensor,
            ctrl_std: torch.Tensor
    ):
        super().__init__()
        self.goal_point = goal_point
        self.u_eq = u_eq
        self.state_std = state_std
        self.ctrl_std = ctrl_std

    @abstractmethod
    def act(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def explore(self, x: torch.Tensor, std: float) -> torch.Tensor:
        mean = self.act(x)
        return torch.normal(mean=mean, std=std)

    def normalize_state(self, x: torch.Tensor) -> torch.Tensor:
        nonzero_std_dim = torch.nonzero(self.state_std)
        zero_mask = torch.ones(self.state_std.shape[0]).type_as(self.state_std)
        zero_mask[nonzero_std_dim] = 0
        x_trans = (x - self.goal_point) / (self.state_std + zero_mask)
        return x_trans

    def normalize_action(self, u: torch.Tensor) -> torch.Tensor:
        nonzero_std_dim = torch.nonzero(self.ctrl_std)
        zero_mask = torch.ones(self.ctrl_std.shape[0]).type_as(self.ctrl_std)
        zero_mask[nonzero_std_dim] = 0
        u_trans = (u - self.u_eq) / (self.ctrl_std + zero_mask)
        return u_trans

    def de_normalize_action(self, u_trans: torch.Tensor) -> torch.Tensor:
        return u_trans * self.ctrl_std + self.u_eq

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: torch.device):
        self.load_state_dict(torch.load(path, map_location=device))

    def disable_grad(self):
        for i in self.parameters():
            i.requires_grad = False

    def enable_grad(self):
        for i in self.parameters():
            i.requires_grad = True
