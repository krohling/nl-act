import time
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import h5py

from constants import (
    PUPPET_GRIPPER_POSITION_NORMALIZE_FN, 
    CAMERA_NAMES
)
from tasks import (
    ALL_TASKS,
    TASK_CONFIGS
)
from ee_sim_env import make_ee_sim_env
from sim_env import make_sim_env, BOX_POSE

import IPython
e = IPython.embed

# python record_multi_task_episodes.py --output_dir ./dataset --num_episodes 150

RENDER_CAM_NAME = 'angle'

def record_episodes(
        num_episodes=10, 
        output_dir='./data', 
        onscreen_render=False, 
        save_failed=False,
        inject_noise=False
    ):
    """
    Generate demonstration data in simulation.
    First rollout the policy (defined in ee space) in ee_sim_env. Obtain the joint trajectory.
    Replace the gripper joint positions with the commanded joint position.
    Replay this joint trajectory (as action sequence) in sim_env, and record all observations.
    Save this episode of data, and continue to next episode of data collection.
    """
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    success = []
    episode_idx = 0
    while episode_idx < num_episodes:
        print(f'{episode_idx=}')
        print('Rollout out EE space scripted policy')

        # Configure the task
        task_id = ALL_TASKS[episode_idx % len(ALL_TASKS)]
        task_name, task_cls, ee_task_cls, scripted_policy_cls, episode_len = TASK_CONFIGS[task_id].values()
        policy = scripted_policy_cls(inject_noise)
        print(f"Using task: {task_name}")

        # setup the environment
        env = make_ee_sim_env(task_cls=ee_task_cls)
        ts = env.reset()
        episode = [ts]

        # setup plotting
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][RENDER_CAM_NAME])
            plt.ion()
        for step in range(episode_len):
            action = policy(ts)
            ts = env.step(action)
            episode.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][RENDER_CAM_NAME])
                plt.pause(0.002)
        plt.close()

        episode_return = np.sum([ts.reward for ts in episode[1:]])
        # episode_max_reward = np.max([ts.reward for ts in episode[1:]])
        episode_final_reward = episode[-1].reward

        if episode_final_reward == env.task.max_reward:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")
            if not save_failed:
                print("Skipping this episode")
                continue

        joint_traj = [ts.observation['qpos'] for ts in episode]
        # replace gripper pose with gripper control
        gripper_ctrl_traj = [ts.observation['gripper_ctrl'] for ts in episode]
        for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
            left_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[0])
            right_ctrl = PUPPET_GRIPPER_POSITION_NORMALIZE_FN(ctrl[2])
            joint[6] = left_ctrl
            joint[6+7] = right_ctrl

        subtask_info = episode[0].observation['env_state'].copy() # box pose at step 0

        # clear unused variables
        del env
        del episode
        del policy

        # setup the environment
        print('Replaying joint commands')
        env = make_sim_env(task_cls=task_cls)
        BOX_POSE[0] = subtask_info # make sure the sim_env has the same object configurations as ee_sim_env
        ts = env.reset()

        episode_replay = [ts]
        # setup plotting
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(ts.observation['images'][RENDER_CAM_NAME])
            plt.ion()
        for t in range(len(joint_traj)): # note: this will increase episode length by 1
            action = joint_traj[t]
            ts = env.step(action)
            episode_replay.append(ts)
            if onscreen_render:
                plt_img.set_data(ts.observation['images'][RENDER_CAM_NAME])
                plt.pause(0.02)

        episode_return = np.sum([ts.reward for ts in episode_replay[1:]])
        # episode_max_reward = np.max([ts.reward for ts in episode_replay[1:]])
        
        episode_final_reward = episode_replay[1:][-1].reward
        # if episode_max_reward == env.task.max_reward:
        if episode_final_reward == env.task.max_reward:
            success.append(1)
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            success.append(0)
            print(f"{episode_idx=} Failed")
            print("Skipping this episode")
            continue

        plt.close()

        """
        For each timestep:
        observations
        - images
            - each_cam_name     (480, 640, 3) 'uint8'
        - qpos                  (14,)         'float64'
        - qvel                  (14,)         'float64'

        action                  (14,)         'float64'
        """

        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
        }
        for cam_name in CAMERA_NAMES:
            data_dict[f'/observations/images/{cam_name}'] = []

        # because the replaying, there will be eps_len + 1 actions and eps_len + 2 timesteps
        # truncate here to be consistent
        joint_traj = joint_traj[:-1]
        episode_replay = episode_replay[:-1]

        # len(joint_traj) i.e. actions: max_timesteps
        # len(episode_replay) i.e. time steps: max_timesteps + 1
        max_timesteps = len(joint_traj)
        while joint_traj:
            action = joint_traj.pop(0)
            ts = episode_replay.pop(0)
            data_dict['/observations/qpos'].append(ts.observation['qpos'])
            data_dict['/observations/qvel'].append(ts.observation['qvel'])
            data_dict['/action'].append(action)
            for cam_name in CAMERA_NAMES:
                data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

        # HDF5
        t0 = time.time()
        dataset_path = os.path.join(output_dir, f'episode_{episode_idx}')
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            root.attrs['task_id'] = task_id
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in CAMERA_NAMES:
                _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )
            # compression='gzip',compression_opts=2,)
            # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
            qpos = obs.create_dataset('qpos', (max_timesteps, 14))
            qvel = obs.create_dataset('qvel', (max_timesteps, 14))
            action = root.create_dataset('action', (max_timesteps, 14))

            for name, array in data_dict.items():
                root[name][...] = array
        print(f'Saving: {time.time() - t0:.1f} secs\n')
        episode_idx += 1

    print(f'Saved to {output_dir}')
    print(f'Success: {np.sum(success)} / {len(success)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', action='store', type=str, help='where to save output dataset', required=True)
    parser.add_argument('--num_episodes', action='store', type=int, default=1, help='number of episodes to record')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--save_failed', action='store_true')
    parser.add_argument('--inject_noise', action='store_true')

    args = parser.parse_args()
    record_episodes(
        num_episodes=args.num_episodes, 
        output_dir=args.output_dir,
        onscreen_render=args.onscreen_render, 
        save_failed=args.save_failed,
        inject_noise=args.inject_noise
    )

