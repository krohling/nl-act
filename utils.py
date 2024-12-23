import ast
import glob
import numpy as np
import pandas as pd
import torch
import os
import h5py
from torch.utils.data import DataLoader

import IPython

e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, 
                    episode_ids, 
                    dataset_dir, 
                    camera_names, 
                    norm_stats, 
                    df_instr,
                    instr_stats,
                ):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.df_embeddings = df_instr
        self.instr_embeddings_mean = instr_stats['mean']
        self.instr_embeddings_std = instr_stats['std']
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        task_id = None
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            task_id = root.attrs['task_id']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            # get all actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        if self.df_embeddings is not None:
            instr_embedding = sample_embeddings(task_id, self.df_embeddings)
            instr_embedding = (instr_embedding - self.instr_embeddings_mean) / self.instr_embeddings_std
            return image_data, qpos_data, action_data, is_pad, task_id, instr_embedding
        return image_data, qpos_data, action_data, is_pad, task_id


def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(
        dataset_dir, 
        batch_size_train, 
        batch_size_val,
        train_instr_path,
        val_instr_path,
        camera_names,
    ):
    print(f'\Loading Dataset from: {dataset_dir}\n')
    train_ratio = 0.8
    num_episodes = len(glob.glob(os.path.join(dataset_dir, '*.hdf5')))
    
    # obtain train test split
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    df_train_instr = pd.read_csv(train_instr_path)
    df_val_instr = pd.read_csv(val_instr_path)
    instr_stats = {
        "mean": torch.tensor(
            np.mean([
                ast.literal_eval(e) for e in (df_train_instr['embedding'].tolist())
            ], axis=0), dtype=torch.float32
        ),
        "std": torch.tensor(
            np.std([
                ast.literal_eval(e) for e in (df_train_instr['embedding'].tolist())
            ], axis=0), dtype=torch.float32
        ),
    }

    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, df_instr=df_train_instr, instr_stats=instr_stats)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, df_instr=df_val_instr, instr_stats=instr_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, instr_stats

def sample_embeddings(task_id, df_task, return_instruction=False):
    example = df_task[df_task['task_id'] == task_id].sample().iloc[0]
    if return_instruction:
        return torch.tensor(ast.literal_eval(example['embedding'])), example['instruction']

    return torch.tensor(ast.literal_eval(example['embedding']))

### env utils

def sample_2_box_poses():
    x_range_options = [
        [0.17, 0.2],
        [0.1, 0.13],
    ]

    x_range_1_idx = np.random.choice([0, 1])
    x_range_1 = x_range_options[x_range_1_idx]

    x_range_2_idx = 0 if x_range_1_idx == 1 else 1
    x_range_2 = x_range_options[x_range_2_idx]

    return sample_box_pose(x_range_1), sample_box_pose(x_range_2)

def sample_box_pose(x_range = [0.2, 0.3]):
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
