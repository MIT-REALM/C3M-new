import torch
import argparse
import numpy as np
import cv2
import os
import time

from tqdm import tqdm

from model.utils import set_seed, read_settings
from model.controller.neural_c3m_controller import NeuralC3MController
from model.env.env import make_tracking_env


def execute_policy(cur_policy, env, args):
    video_path = os.path.join(args.path, 'videos')
    if not args.no_video and not os.path.exists(video_path):
        os.mkdir(video_path)

    rewards = []
    lengths = []

    out = None
    if not args.no_video:
        env.reset()
        data = env.render()
        out = cv2.VideoWriter(
            os.path.join(video_path, f'reward{0.0}.mov'),
            cv2.VideoWriter_fourcc(*'mp4v'),
            25,
            (data.shape[1], data.shape[0])
        )

    for i_epi in range(args.epi):
        epi_length = 0
        epi_reward = 0
        state = env.reset()
        t = 0
        while True:
            state_ref, action_ref = env.get_ref()
            action = cur_policy(state, state_ref, action_ref)

            next_state, reward, done, _ = env.step(action)
            epi_length += 1
            epi_reward += reward
            state = next_state
            t += 1

            if not args.no_video:
                if t % 5 == 0:
                    out.write(env.render())
            if done:
                tqdm.write(f'epi: {i_epi}, reward: {epi_reward:.2f}, length: {epi_length}')
                rewards.append(epi_reward)
                lengths.append(epi_length)
                break

    if not args.no_video:
        out.release()
        os.rename(os.path.join(video_path, f'reward{0.0}.mov'),
                  os.path.join(video_path, f'reward{np.mean(rewards):.2f}.mov'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # custom
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--epi', type=int, default=2)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--no-video', action='store_true', default=False)

    # default
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    # set up env
    settings = read_settings(args.path)
    set_seed(args.seed)
    device = torch.device('cpu')
    env = make_tracking_env(settings['env'], device=device)

    # load models
    model_path = os.path.join(args.path, 'models')
    state_std = 1 / np.sqrt(12) * (env.state_limits[0] - env.state_limits[1])
    ctrl_std = 1 / np.sqrt(12) * (env.control_limits[0] - env.control_limits[1])
    policy = NeuralC3MController(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        goal_point=env.goal_point,
        u_eq=env.u_eq,
        state_std=state_std,
        ctrl_std=ctrl_std
    ).to(device)
    policy.load(os.path.join(model_path, f'c3m.pkl'), device=device)
    policy = policy.controller.track

    print('> Processing...')
    start_time = time.time()
    execute_policy(policy, env, args)
    print(f'> Done in {time.time() - start_time:.0f}s')
