import torch

from .base import TrackingEnv
from .dubins_car import DubinsCarTracking


def make_tracking_env(
        env_id: str,
        device: torch.device = torch.device('cpu'),
) -> TrackingEnv:
    if env_id == 'DubinsCarTracking':
        return DubinsCarTracking(device)
    else:
        raise NotImplementedError(f'{env_id} not implemented')
