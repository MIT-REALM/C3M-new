import argparse
import torch
import numpy as np
import os

from model.utils import set_seed, init_logger
from model.env.env import make_tracking_env
from model.controller.neural_c3m_controller import NeuralC3MController
from model.trainer.c3m_trainer import C3MTrainer


def train_c3m(args):
    set_seed(args.seed)
    device = torch.device('cuda' if (torch.cuda.is_available() and not args.no_cuda) else 'cpu')
    env = make_tracking_env(env_id=args.env, device=device)

    # set up logger
    log_path, writer, model_path = init_logger(args.log_path, args.env, args.seed, vars(args))

    # set up C3M controller
    state_std = 1 / np.sqrt(12) * (env.state_limits[0] - env.state_limits[1])
    ctrl_std = 1 / np.sqrt(12) * (env.control_limits[0] - env.control_limits[1])
    c3m_controller = NeuralC3MController(
        state_dim=env.n_dims,
        action_dim=env.n_controls,
        goal_point=env.goal_point,
        u_eq=env.u_eq,
        state_std=state_std,
        ctrl_std=ctrl_std,
    ).to(device)

    trainer = C3MTrainer(
        controller=c3m_controller,
        writer=writer,
        gt_env=env
    )
    trainer.train(args.n_iter, args.batch_size)
    c3m_controller.save(os.path.join(model_path, f'c3m.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # custom
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--n-iter', type=int, default=3000)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--no-cuda', action='store_true', default=False)

    # default
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log-path', type=str, default='./logs')

    args = parser.parse_args()
    train_c3m(args)
