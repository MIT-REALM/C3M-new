import torch


def jacobian(f: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Calculate vector-vector jacobian

    Parameters
    ----------
    f: torch.Tensor
        bs x m x 1
    x: torch.Tensor
        bs x n x 1

    Returns
    -------
    J: torch.Tensor
        bs x m x n
    """
    f = f + 0. * x.sum()  # to avoid the case that f is independent of x
    bs = x.shape[0]
    J = torch.zeros(bs, f.shape[1], x.shape[1]).type_as(x)
    for i in range(f.shape[1]):
        J[:, i, :] = torch.autograd.grad(f[:, i, 0].sum(), x, create_graph=True)[0].squeeze(-1)
    return J


def jacobian_matrix(M: torch.Tensor, x: torch.Tensor):
    """
    Calculate matrix-vector jacobian

    Parameters
    ----------
    M: torch.Tensor
        bs x m x m
    x: torch.Tensor
        bs x n x 1

    Returns
    -------
    J: torch.Tensor
        bs x m x m x n
    """
    bs = x.shape[0]
    J = torch.zeros(bs, M.shape[-1], M.shape[-2], x.shape[1]).type_as(x)
    for i in range(M.shape[-2]):
        for j in range(M.shape[-1]):
            J[:, i, j, :] = torch.autograd.grad(M[:, i, j].sum(), x, create_graph=True)[0].squeeze(-1)
    return J


def weighted_grad(W, v, x, detach=False):
    assert v.size() == x.size()
    bs = x.shape[0]
    if detach:
        return (jacobian_matrix(W, x).detach() * v.view(bs, 1, 1, -1)).sum(dim=3)
    else:
        return (jacobian_matrix(W, x) * v.view(bs, 1, 1, -1)).sum(dim=3)


def loss_pos_matrix_random_sampling(A, dim=1024):
    # A: bs x d x d
    # z: K x d
    z = torch.randn(dim, A.size(-1)).type_as(A)
    z = z / z.norm(dim=1, keepdim=True)
    zTAz = (z.matmul(A) * z.view(1, dim, -1)).sum(dim=2).view(-1)
    negative_index = zTAz.detach().cpu().numpy() < 0
    if negative_index.sum() > 0:
        negative_zTAz = zTAz[negative_index]
        return -1.0 * (negative_zTAz.mean())
    else:
        return torch.tensor(0.).type_as(z).requires_grad_()
