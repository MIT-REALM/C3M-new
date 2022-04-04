import torch
import numpy as np

from typing import Optional, Tuple
from abc import ABC, abstractmethod, abstractproperty
from gym.spaces import Box
from torch.autograd.functional import jacobian

from .utils import lqr, continuous_lyap

# gravitation acceleration
grav = 9.80665


class ControlAffineSystem(ABC):
    """
    Represents an abstract control-affine dynamical system.

    A control-affine dynamical system is one where the state derivatives are affine in
    the control input, e.g.:

        dx/dt = f(x) + g(x) u

    These can be used to represent a wide range of dynamical systems, and they have some
    useful properties when it comes to designing controllers.
    """

    def __init__(
            self,
            device: torch.device,
            dt: float = 0.01,
            params: Optional[dict] = None,
            controller_dt: Optional[float] = None
    ):
        super().__init__()

        self.device = device

        # validate parameters, raise error if they're not valid
        if params is not None and not self.validate_params(params):
            raise ValueError(f"Parameters not valid: {params}")

        if params is None:
            self.params = self.default_param()
        else:
            self.params = params

        # make sure the time step is valid
        assert dt > 0.0
        self.dt = dt

        if controller_dt is None:
            controller_dt = self.dt
        self.controller_dt = controller_dt

        self._state = None
        self._action = None
        self._K = None
        self._P = None

        self._t = 0  # current simulation time

    @abstractmethod
    def reset(self) -> torch.Tensor:
        """
        Reset the environment. Remember to do the following things:
            1. self._t = 0
            2. set initial state self._state

        Returns
        -------
        state: torch.Tensor
            call return self.state
        """
        pass

    @abstractmethod
    def step(self, u: torch.Tensor) -> Tuple[torch.Tensor, float, bool, dict]:
        """
        Simulate the environment for one step, using the same format as OpenAI gym

        Returns
        -------
        state: torch.Tensor
            self.state
        reward: float
            the reward of current (s, a)
        done: bool
            if this episode ends
        info: dict
            other information you want to return
        """
        pass

    @abstractmethod
    def render(self) -> np.ndarray:
        """
        Return the image of the current frame

        Returns
        -------
        data: np.ndarray
            the canvas of the current frame
        """
        pass

    @abstractmethod
    def default_param(self) -> dict:
        """
        The default parameters of the environment

        Returns
        -------
        params: dict
        """
        pass

    @abstractmethod
    def validate_params(self, params: dict) -> bool:
        """
        Check the given parameters to see if they are valid for the environment

        Returns
        -------
        valid: bool
        """
        pass

    @abstractmethod
    def state_name(self, dim: int) -> str:
        """
        Return the state name of the given dimension

        Returns
        -------
        name: str
        """
        pass

    @abstractmethod
    def _f(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the control-independent part of the control-affine dynamics.

        Parameters
        ----------
        x: torch.Tensor
            batch_size x self.n_dims tensor of state

        Returns
        -------
        f: torch.Tensor
            batch_size x self.n_dims x 1
        """
        pass

    @abstractmethod
    def _g(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the control-independent part of the control-affine dynamics.

        Parameters
        ----------
        x: torch.Tensor
            batch_size x self.n_dims tensor of state

        Returns
        -------
        g: torch.Tensor
            batch_size x self.n_dims x self.n_controls
        """
        pass

    @abstractproperty
    def n_dims(self) -> int:
        """
        number of states

        Returns
        -------
        n_dims: int
        """
        pass

    @abstractproperty
    def n_controls(self) -> int:
        """
        number of control inputs

        Returns
        -------
        n_controls: int
        """
        pass

    @abstractproperty
    def max_episode_steps(self) -> int:
        """
        Maximum episode steps. Usually if self._t is larger than this, "done" will be true in self.step.

        Returns
        -------
        steps: int
        """
        pass

    @abstractproperty
    def state_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the expected range of states for this
        system
        """
        pass

    @abstractproperty
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (upper, lower) describing the range of allowable control
        limits for this system
        """
        pass

    @property
    def observation_space(self) -> Box:
        return Box(
            low=self.state_limits[0].cpu().detach().numpy(),
            high=self.state_limits[1].cpu().detach().numpy()
        )

    @property
    def action_space(self) -> Box:
        return Box(
            low=self.control_limits[0].cpu().detach().numpy(),
            high=self.control_limits[1].cpu().detach().numpy()
        )

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        One-step simulation

        Parameters
        ----------
        x: torch.Tensor
            current state
        u: torch.Tensor
            current control input

        Returns
        -------
        x_next: torch.Tensor
            next state
        """
        x_dot = self.closed_loop_dynamics(x, u)
        return x + (x_dot * self.dt)

    def control_affine_dynamics(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return a tuple (f, g) representing the system dynamics in control-affine form:
            dx/dt = f(x) + g(x) u

        Parameters
        ----------
        x: torch.Tensor
            batch_size x self.n_dims tensor of state

        Returns
        -------
        f: torch.Tensor
            batch_size x self.n_dims x 1 representing the control-independent dynamics
        g: torch.Tensor
            batch_size x self.n_dims x self.n_controls representing the control-dependent dynamics
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim == 3:
            x = x.squeeze(-1)

        # sanity check on input
        assert x.ndim == 2
        assert x.shape[1] == self.n_dims

        return self._f(x), self._g(x)

    def closed_loop_dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Return the state derivatives at state x and control input u
            dx/dt = f(x) + g(x) u

        Parameters
        ----------
        x: torch.Tensor
            batch_size x self.n_dims tensor of state
        u: torch.Tensor
            batch_size x self.n_controls tensor of controls

        Returns
        -------
        x_dot: torch.Tensor
            batch_size x self.n_dims tensor of time derivatives of x
        """
        # Get the control-affine dynamics
        f, g = self.control_affine_dynamics(x)
        # Compute state derivatives using control-affine form
        x_dot = f + torch.bmm(g, u.unsqueeze(-1))
        return x_dot.view(x.shape)

    @torch.enable_grad()
    def compute_A_matrix(self) -> np.ndarray:
        """Compute the linearized continuous-time state-state derivative transfer matrix
        about the goal point"""
        # Linearize the system about the x = 0, u = 0
        x0 = self.goal_point
        u0 = self.u_eq

        def dynamics(x):
            return self.closed_loop_dynamics(x, u0).squeeze()

        A = jacobian(dynamics, x0).squeeze().cpu().numpy()
        A = np.reshape(A, (self.n_dims, self.n_dims))

        return A

    def compute_B_matrix(self) -> np.ndarray:
        """Compute the linearized continuous-time state-state derivative transfer matrix
        about the goal point"""

        # Linearize the system about the x = 0, u = 0
        B = self._g(self.goal_point).squeeze().cpu().numpy()
        B = np.reshape(B, (self.n_dims, self.n_controls))

        return B

    def linearized_ct_dynamics_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the continuous time linear dynamics matrices, dx/dt = Ax + Bu
        """
        A = self.compute_A_matrix()
        B = self.compute_B_matrix()

        return A, B

    def linearized_dt_dynamics_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the continuous time linear dynamics matrices, x_{t+1} = Ax_{t} + Bu
        """
        Act, Bct = self.linearized_ct_dynamics_matrices()
        A = np.eye(self.n_dims) + self.controller_dt * Act
        B = self.controller_dt * Bct

        return A, B

    def compute_linearized_controller(self):
        """
        Computes the linearized controller K and lyapunov matrix P.
        """
        # Compute the LQR gain matrix for the nominal parameters
        if not self.use_lqr:
            print('not using LQR')
            return

        try:
            Act, Bct = self.linearized_ct_dynamics_matrices()
            A, B = self.linearized_dt_dynamics_matrices()
        except NotImplementedError:
            print('not using LQR')
            return

        # Define cost matrices as identity
        Q = np.eye(self.n_dims)
        R = np.eye(self.n_controls)

        # Get feedback matrix
        K_np = lqr(A, B, Q, R)
        self._K = torch.tensor(K_np, device=self.device)

        Acl = Act - Bct @ K_np

        # use the standard Lyapunov equation
        self._P = torch.tensor(continuous_lyap(Acl, Q), device=self.device)

    def u_nominal(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the nominal control for the nominal parameters, using LQR unless overridden

        Parameters
        ----------
        x: torch.Tensor
            bs x self.n_dims tensor of state

        Returns
        -------
        u_nominal: torch.Tensor
            bs x self.n_controls tensor of controls
        """
        if self._K is None:
            raise KeyError('u_nominal is not computed, call compute_linearized_controller() first')

        # Compute nominal control from feedback + equilibrium control
        K = self._K.type_as(x)
        goal = self.goal_point.squeeze().type_as(x)
        u_nominal = -(K @ (x - goal).T).T

        # Adjust for the equilibrium setpoint
        u = u_nominal + self.u_eq.type_as(x)

        # clamp given the control limits
        upper_u_lim, lower_u_lim = self.control_limits
        for dim_idx in range(self.n_controls):
            u[:, dim_idx] = torch.clamp(
                u[:, dim_idx],
                min=lower_u_lim[dim_idx].item(),
                max=upper_u_lim[dim_idx].item(),
            )

        return u

    def sample_states(self, batch_size: int) -> torch.Tensor:
        """
        sample states from the env

        Parameters
        ----------
        batch_size: int

        Returns
        -------
        states: torch.Tensor
            sampled states from the env
        """
        high, low = self.state_limits
        if torch.isinf(low).any() or torch.isinf(high).any():
            states = torch.randn(batch_size, self.n_dims, device=self.device)  # todo: add mean and std
        else:
            rand = torch.rand(batch_size, self.n_dims, device=self.device)
            states = rand * (high - low) + low
        return states

    def sample_states_ref(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        high, low = self.state_limits
        if torch.isinf(low).any() or torch.isinf(high).any():
            states_ref = torch.randn(batch_size, self.n_dims, device=self.device)  # todo: add mean and std
        else:
            rand = torch.rand(batch_size, self.n_dims, device=self.device)
            states_ref = rand * (high - low) + low
        error = torch.rand(batch_size, self.n_dims, device=self.device) * 2 - 1
        std = 1 / np.sqrt(12) * (high - low)
        return states_ref + error * std * 0.5, states_ref

    def sample_ctrls(self, batch_size: int) -> torch.Tensor:
        high, low = self.control_limits
        if torch.isinf(low).any() or torch.isinf(high).any():
            ctrls = torch.randn(batch_size, self.n_controls, device=self.device)  # todo: add mean and std
        else:
            rand = torch.rand(batch_size, self.n_controls, device=self.device)
            ctrls = rand * (high - low) + low
        return ctrls

    def noisy_u_nominal(self, x: torch.Tensor, std: float, bias: float = 0) -> torch.Tensor:
        u = self.u_nominal(x)
        u += torch.randn_like(u) * std * (self.control_limits[0] - self.control_limits[1])
        u += bias

        # clamp given the control limits
        upper_u_lim, lower_u_lim = self.control_limits
        for dim_idx in range(self.n_controls):
            u[:, dim_idx] = torch.clamp(
                u[:, dim_idx],
                min=lower_u_lim[dim_idx].item(),
                max=upper_u_lim[dim_idx].item(),
            )

        return u

    @property
    def goal_point(self) -> torch.Tensor:
        return torch.zeros((1, self.n_dims), device=self.device)

    @property
    def state(self) -> torch.Tensor:
        if self._state is not None:
            return self._state.squeeze(0)
        else:
            raise ValueError('State is not initialized')

    @property
    def reward_range(self) -> tuple:
        return float('-inf'), float('inf')

    @property
    def metadata(self) -> dict:
        return {}

    @property
    def u_eq(self) -> torch.Tensor:
        return torch.zeros((1, self.n_controls), device=self.device)

    @abstractproperty
    def use_lqr(self) -> bool:
        return True


class TrackingEnv(ControlAffineSystem, ABC):

    def __init__(
            self,
            device: torch.device,
            dt: float = 0.01,
            params: Optional[dict] = None,
            controller_dt: Optional[float] = None
    ):
        super(TrackingEnv, self).__init__(device, dt, params, controller_dt)

    @abstractmethod
    def get_ref(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the reference state and action at the current time

        Returns
        -------
        states_ref: torch.Tensor
        action_ref: torch.Tensor
        """
        pass

    @abstractmethod
    def set_ref(self, states: torch.Tensor, actions: torch.Tensor):
        """
        Set the reference path to be the given one

        Parameters
        ----------
        states: torch.Tensor
        actions: torch.Tensor
        """
        pass
