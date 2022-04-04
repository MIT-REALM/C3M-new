import numpy as np
import scipy.linalg
import torch
import math
import matplotlib.pyplot as plt


from descartes import PolygonPatch
from shapely.geometry import Polygon, LineString
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def lqr(
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
        return_eigs: bool = False,
):
    """
    Solve the discrete time lqr controller.
        x_{t+1} = A x_t + B u_t
        cost = sum x.T*Q*x + u.T*R*u

    Code adapted from Mark Wilfred Mueller's continuous LQR code at
    https://www.mwm.im/lqr-controllers-with-python/
    Based on Bertsekas, p.151

    Yields the control law u = -K x
    """

    # first, try to solve the ricatti equation
    X = scipy.linalg.solve_discrete_are(A, B, Q, R)

    # compute the LQR gain
    K = scipy.linalg.inv(B.T @ X @ B + R) @ (B.T @ X @ A)

    if not return_eigs:
        return K
    else:
        eigVals, _ = scipy.linalg.eig(A - B * K)
        return K, eigVals


def continuous_lyap(Acl: np.ndarray, Q: np.ndarray):
    """Solve the continuous time lyapunov equation.

    Acl.T P + P Acl + Q = 0

    using scipy, which expects AP + PA.T = Q, so we need to transpose Acl and negate Q
    """
    P = scipy.linalg.solve_continuous_lyapunov(Acl.T, -Q)
    return P


def plot_line(ax, line: LineString):
    x, y = line.xy
    ax.plot(x, y, color='gray', linewidth=3, solid_capstyle='round', zorder=1)


def plot_poly(ax, poly: Polygon, color: str, alpha: float = 1.0):
    patch = PolygonPatch(poly, fc=color, ec="black", alpha=alpha, zorder=0)
    ax.add_patch(patch)


def get_rectangle(center: torch.Tensor, heading: float, length: float, width: float) -> Polygon:
    trans = torch.tensor([[np.cos(heading), -np.sin(heading)],
                          [np.sin(heading), np.cos(heading)]]).type_as(center)
    points = [
        torch.tensor([length / 2, width / 2]).type_as(center),
        torch.tensor([-length / 2, width / 2]).type_as(center),
        torch.tensor([-length / 2, -width / 2]).type_as(center),
        torch.tensor([length / 2, -width / 2]).type_as(center)
    ]

    for i, point in enumerate(points):
        points[i] = (torch.matmul(trans, point) + center).unsqueeze(0).numpy()
    points = np.concatenate(points, axis=0)

    return Polygon(points)


def cuboid_data2(o, size=(1, 1, 1)):
    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    X = np.array(X).astype(float)
    for i in range(3):
        X[:, :, i] *= size[i]
    X += np.array(o)
    return X


def plotCubeAt2(position, size=None, color=None):
    if not isinstance(color, (list, np.ndarray)):
        color = "C0"
    if not isinstance(size, (list, np.ndarray)):
        size = (1, 1, 1)
    return Poly3DCollection(cuboid_data2(position, size),
                            facecolors=np.repeat(color, 6))


def scale3d(pts, scale_list):
    """scale a 3d ndarray of points, and return the new ndarray"""

    assert len(scale_list) == 3

    rv = np.zeros(pts.shape)

    for i in range(pts.shape[0]):
        for d in range(3):
            rv[i, d] = scale_list[d] * pts[i, d]

    return rv


def rotate3d(pts, theta, psi, phi):
    """rotates an ndarray of 3d points, returns new list"""

    sinTheta = math.sin(theta)
    cosTheta = math.cos(theta)
    sinPsi = math.sin(psi)
    cosPsi = math.cos(psi)
    sinPhi = math.sin(phi)
    cosPhi = math.cos(phi)

    transform_matrix = np.array([
        [cosPsi * cosTheta, -sinPsi * cosTheta, sinTheta],
        [cosPsi * sinTheta * sinPhi + sinPsi * cosPhi,
         -sinPsi * sinTheta * sinPhi + cosPsi * cosPhi,
         -cosTheta * sinPhi],
        [-cosPsi * sinTheta * cosPhi + sinPsi * sinPhi,
         sinPsi * sinTheta * cosPhi + cosPsi * sinPhi,
         cosTheta * cosPhi]], dtype=float)

    rv = np.zeros(pts.shape)

    for i in range(pts.shape[0]):
        rv[i] = np.dot(pts[i], transform_matrix)

    return rv


def normalize_angle(theta: float):
    psi = theta
    while psi >= np.pi:
        psi -= 2 * np.pi
    while psi <= -np.pi:
        psi += 2 * np.pi
    return psi
