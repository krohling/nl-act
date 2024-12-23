import os
import time
import torch
import numpy as np
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange
import wandb

from constants import (
    CAMERA_NAMES,
    ONSCREEN_CAM,
    DT,
)
from tasks import (
    ALL_TASKS,
    TASK_CONFIGS,
    DEFAULT_EPISODE_LEN,
)
from utils import (
    load_data,
    sample_embeddings,
    sample_2_box_poses,
    compute_dict_mean,
    set_seed,
    detach_dict,
)
from policy import ACTPolicy
from visualize_episodes import save_videos
from sim_env import BOX_POSE, make_sim_env, DummyTask

import IPython
e = IPython.embed

# python train.py --dataset_dir ./dataset --train_instr_path ./data/instruction_embeddings.train.csv --val_instr_path ./data/instruction_embeddings.val.csv  --ckpt_dir ./output/checkpoints --eval --num_rollouts 15 --eval_instr_path ./data/instruction_embeddings.val.csv  --videos_dir ./output/videos

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "nl-act")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "kevin_ai")
WANDB_ENABLED = os.environ.get("WANDB_API_KEY", None) is not None

DEFAULT_MODEL_CONFIG = {
    'lr': 1e-5,
    'num_queries': 100,
    'kl_weight': 10,
    'hidden_dim': 512,
    'dim_feedforward': 3200,
    'state_dim': 14,
    'lr_backbone': 1e-5,
    'backbone': 'resnet18',
    'enc_layers': 4,
    'dec_layers': 7,
    'nheads': 8,
    'camera_names': CAMERA_NAMES,
}

def imitate_multi_task_episodes(
        dataset_dir,
        train_instr_path,
        val_instr_path,
        lr=1e-5,
        num_epochs=1,
        batch_size_train=1,
        batch_size_val=1,
        ckpt_dir=None,
        ckpt_frequency=2500,
        load_ckpt_path=None,
        seed=1,
        do_eval=False,
        eval_instr_path=None,
        eval_frequency=1,
        eval_wait=0,
        num_rollouts=10,
        onscreen_render=False,
        temporal_agg=False,
        videos_dir=None,
        model_config=DEFAULT_MODEL_CONFIG,
        wandb_run_name=None,
    ):
    if WANDB_ENABLED:
        wandb.login()
        if wandb_run_name is not None:
            wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=wandb_run_name)
        else:
            wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY)

    set_seed(args.seed)
    train_dataloader, val_dataloader, dataset_stats, embeddings_stats = load_data(
        dataset_dir=dataset_dir, 
        batch_size_train=batch_size_train, 
        batch_size_val=batch_size_val,
        train_instr_path=train_instr_path,
        val_instr_path=val_instr_path,
        camera_names=CAMERA_NAMES, 
    )
    train_bc(
        num_epochs=num_epochs,
        ckpt_dir=ckpt_dir,
        ckpt_frequency=ckpt_frequency,
        load_ckpt_path=load_ckpt_path,
        seed=seed,
        do_eval=do_eval,
        eval_instr_path=eval_instr_path,
        eval_frequency=eval_frequency,
        eval_wait=eval_wait,
        num_rollouts=num_rollouts,
        onscreen_render=onscreen_render,
        temporal_agg=temporal_agg,
        videos_dir=videos_dir,
        train_dataloader=train_dataloader, 
        val_dataloader=val_dataloader,
        dataset_stats=dataset_stats,
        embeddings_stats=embeddings_stats,
        policy_config={
            'lr': lr,
            'camera_names': CAMERA_NAMES,
            **model_config,
        }
    )


def train_bc(
        train_dataloader, 
        val_dataloader, 
        dataset_stats,
        embeddings_stats,
        policy_config,
        num_epochs=1,
        ckpt_dir='./checkpoints',
        ckpt_frequency=2500,
        do_eval=False,
        eval_instr_path=None,
        eval_frequency=1,
        eval_wait=0,
        num_rollouts=10,
        onscreen_render=False,
        videos_dir=None,
        temporal_agg=False,
        load_ckpt_path=None,
        seed=0,
    ):
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    set_seed(seed)
    policy = ACTPolicy({
        **policy_config,
        'policy_class': 'ACT',
        'num_epochs': num_epochs,
        'seed': seed,
    })
    if load_ckpt_path is not None:
        if torch.cuda.is_available():
            loading_status = policy.load_state_dict(torch.load(load_ckpt_path, weights_only=False))
        else:
            loading_status = policy.load_state_dict(torch.load(load_ckpt_path, weights_only=False, map_location=torch.device('cpu')))
        print(f"Loaded model checkpoint: {load_ckpt_path}")
        print(loading_status)
    
    if torch.cuda.is_available():
        policy.cuda()
    optimizer = policy.configure_optimizers()

    train_history = []
    validation_history = []
    for epoch in tqdm(range(num_epochs)):
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)
            epoch_val_loss = epoch_summary['loss']
        print(f'Val loss:   {epoch_val_loss:.5f}')
        
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')

        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        if WANDB_ENABLED:
            wandb.log({
                'epoch': epoch,
                'train_loss': epoch_train_loss,
                'val_loss': epoch_val_loss,
                **epoch_summary
            })

        def save_checkpoint(ckpt_path):
            state_dict = {
                **policy.state_dict(),
                'metadata': {
                    'epoch': epoch,
                    'train_loss': epoch_train_loss,
                    'val_loss': epoch_val_loss,
                    'seed': seed,
                    'policy_config': policy_config,
                    'dataset_stats': dataset_stats,
                    'embeddings_stats': embeddings_stats,
                    'task_conifigurations': TASK_CONFIGS,
                    'datetime': time.strftime('%Y-%m-%d %H:%M:%S'),
                }
            }
            torch.save(state_dict, ckpt_path)

        if (epoch+1) % ckpt_frequency == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_checkpoint_{epoch}.ckpt')
            save_checkpoint(ckpt_path)
            if WANDB_ENABLED:
                wandb.save(ckpt_path)

        if do_eval and (epoch+1) >= eval_wait and (epoch+1) % eval_frequency == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_eval_checkpoint_{epoch}.ckpt')
            save_checkpoint(ckpt_path)

            print(f'Evaluating checkpoint for epoch {epoch}...')
            start_time = time.time()

            epoch_videos_dir = None
            if videos_dir is not None:
                epoch_videos_dir = os.path.join(videos_dir, f'epoch_{epoch}')
                if not os.path.isdir(epoch_videos_dir):
                    os.makedirs(epoch_videos_dir)

            success_rate, avg_return, task_results = eval_bc(
                ckpt_path=ckpt_path,
                instr_path=eval_instr_path,
                num_rollouts=num_rollouts,
                onscreen_render=onscreen_render,
                temporal_agg=temporal_agg,
                videos_dir=epoch_videos_dir,
            )

            # Cleanup eval checkpoint
            os.remove(ckpt_path)

            print(f'Evaluation time: {(time.time() - start_time):.2f}s')
            print(f'{epoch=} {success_rate=} {avg_return=}')

            if WANDB_ENABLED:
                wandb.save(ckpt_path)
                
                wandb.log({
                    'epoch': epoch,
                    'num_rollouts': 10,
                    'success_rate': success_rate,
                    'avg_return': avg_return
                })

                for task_id in ALL_TASKS:
                    task_name = task_results[task_id]["task_name"]
                    wandb.log({
                        'epoch': epoch,
                        f'task_{task_name}_success_rate': np.mean(task_results[task_id]["success"]),
                        f'task_{task_name}_avg_return': np.mean(task_results[task_id]["return"])
                    })



def eval_bc(
        ckpt_path,
        instr_path=None,
        instr_text=None,
        num_rollouts = 1,
        onscreen_render=False,
        temporal_agg=False,
        seed=1000,
        videos_dir=None,
    ):
    if instr_path is None and instr_text is None:
        raise ValueError("Either instr_path or instr_text must be provided")

    set_seed(seed)

    # load policy
    policy, metadata = load_checkpoint(ckpt_path)
    
    dataset_stats = metadata['dataset_stats']
    policy_config = metadata['policy_config']
    state_dim = policy_config['state_dim']
    instr_embeddings_stats = metadata['embeddings_stats']
    instr_embeddings_mean = instr_embeddings_stats['mean']
    instr_embeddings_std = instr_embeddings_stats['std']
    
    if torch.cuda.is_available():
        policy.cuda()
    policy.eval()

    print(f"Loaded model checkpoint: {ckpt_path}")

    if instr_path is not None:
        df_instr = pd.read_csv(instr_path)
        print(f'Loaded instruction embeddings from {instr_path}')

    pre_process = lambda s_qpos: (s_qpos - dataset_stats['qpos_mean']) / dataset_stats['qpos_std']
    post_process = lambda a: a * dataset_stats['action_std'] + dataset_stats['action_mean']

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    # max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    episode_returns = []
    highest_rewards = []
    final_rewards = []
    task_results = {
        task_id: {
            "task_name": TASK_CONFIGS[task_id]['task_name'],
            "success": [], 
            "return": []
        }  for task_id in ALL_TASKS 
    }
    for rollout_id in range(num_rollouts):
        rollout_id += 0

        task_id = None
        if instr_text is not None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-mpnet-base-v2')
            instr_embedding = model.encode(instr_text)
            instr_embedding = torch.tensor(instr_embedding).float()
            task_cls = DummyTask
            episode_len = DEFAULT_EPISODE_LEN

            print(f'Rollout {rollout_id}, Max timesteps {episode_len}, Instruction {instr_text}')
        else:
            task_id = ALL_TASKS[rollout_id % len(ALL_TASKS)]
            task_name, task_cls, _, _, episode_len = TASK_CONFIGS[task_id].values()
            instr_embedding = sample_embeddings(task_id, df_instr)

            print(f'Rollout {rollout_id}, Task {task_id}, Task Name {task_name}, Max timesteps {episode_len}')

        instr_embedding = (instr_embedding - instr_embeddings_mean) / instr_embeddings_std
        instr_embedding = instr_embedding.unsqueeze(0)
        if torch.cuda.is_available():
            instr_embedding = instr_embedding.cuda()

        env = make_sim_env(task_cls=task_cls)
        env_max_reward = env.task.max_reward

        red_box_pose, blue_box_pose = sample_2_box_poses()
        BOX_POSE[0] = np.concatenate((red_box_pose, blue_box_pose)) 
        ts = env.reset()

        ### onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=ONSCREEN_CAM))
            plt.ion()

        ### evaluation loop
        if temporal_agg:
            if torch.cuda.is_available():
                all_time_actions = torch.zeros([episode_len, episode_len+num_queries, state_dim]).cuda()
            else:
                all_time_actions = torch.zeros([episode_len, episode_len+num_queries, state_dim])

        if torch.cuda.is_available():
            qpos_history = torch.zeros((1, episode_len, state_dim)).cuda()
        else:
            qpos_history = torch.zeros((1, episode_len, state_dim))
        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        with torch.inference_mode():
            for t in range(episode_len):
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id=ONSCREEN_CAM)
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                qpos_numpy = np.array(obs['qpos'])
                qpos = pre_process(qpos_numpy)
                if torch.cuda.is_available():
                    qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                else:
                    qpos = torch.from_numpy(qpos).float().unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(ts, CAMERA_NAMES)

                ### query policy
                if t % query_frequency == 0:
                    all_actions = policy(qpos, curr_image, instr_embedding=instr_embedding)
                if temporal_agg:
                    all_time_actions[[t], t:t+num_queries] = all_actions
                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    if torch.cuda.is_available():
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    else:
                        exp_weights = torch.from_numpy(exp_weights).unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    raw_action = all_actions[:, t % query_frequency]

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action

                ### step the environment
                ts = env.step(target_qpos)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)

                if ts.reward == env_max_reward:
                    break

            plt.close()

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        episode_final_reward = rewards[-1]
        final_rewards.append(episode_final_reward)

        if task_id is not None:
            task_results[task_id]["success"].append(episode_final_reward == env_max_reward)
            task_results[task_id]["return"].append(episode_return)

        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {episode_final_reward=}, {env_max_reward=}, Success: {episode_final_reward==env_max_reward}')

        if videos_dir is not None:
            if not os.path.isdir(videos_dir):
                os.makedirs(videos_dir)
            
            video_path = os.path.join(videos_dir, f'video_{rollout_id}.mp4')
            save_videos(image_list, DT, video_path=video_path)
            if WANDB_ENABLED:
                wandb.save(video_path)

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    # success_rate = np.mean(np.array(final_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        # more_or_equal_r = (np.array(final_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    return success_rate, avg_return, task_results


def load_checkpoint(ckpt_path):
    if torch.cuda.is_available():
        state_dict = torch.load(ckpt_path, weights_only=False)
    else:
        state_dict = torch.load(ckpt_path, weights_only=False, map_location=torch.device('cpu'))
    
    metadata = state_dict.pop("metadata")
    policy_config = metadata['policy_config']
    
    policy = ACTPolicy(policy_config)
    loading_status = policy.load_state_dict(state_dict)
    print(loading_status)
    
    return policy, metadata

def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad, _, instr_embedding = data
    
    if torch.cuda.is_available():
        image_data, qpos_data, action_data, is_pad, instr_embedding = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda(), instr_embedding.cuda()
    
    return policy(qpos_data, image_data, action_data, is_pad, instr_embedding=instr_embedding)


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    if torch.cuda.is_available():
        curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    else:
        curr_image = torch.from_numpy(curr_image / 255.0).float().unsqueeze(0)
    return curr_image


DATASET_DIR = os.environ.get("DATASET_DIR", './dataset')
TRAIN_INSTR_PATH = os.environ.get("TRAIN_INSTR_PATH", './data/instruction_embeddings.train.csv')
VAL_INSTR_PATH = os.environ.get("VAL_INSTR_PATH", './data/instruction_embeddings.val.csv')
LR = float(os.environ.get("LR", "1e-5"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "30000"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
CKPT_DIR = os.environ.get("CKPT_DIR", './output/checkpoints')
CKPT_FREQUENCY = int(os.environ.get("CKPT_FREQUENCY", "2500"))
LOAD_CKPT_PATH = os.environ.get("LOAD_CKPT_PATH", None)
SEED = int(os.environ.get("SEED", "0"))
EVAL = os.environ.get("EVAL", "False").lower() == "true"
EVAL_INSTR_PATH = os.environ.get("EVAL_INSTR_PATH", './data/instruction_embeddings.val.csv')
EVAL_FREQUENCY = int(os.environ.get("EVAL_FREQUENCY", "2500"))
EVAL_WAIT = int(os.environ.get("EVAL_WAIT", "0"))
NUM_ROLLOUTS = int(os.environ.get("NUM_ROLLOUTS", "10"))
VIDEOS_DIR = os.environ.get("VIDEOS_DIR", './output/videos')
ONSCREEN_RENDER = os.environ.get("ONSCREEN_RENDER", "False").lower() == "true"
TEMPORAL_AGG = os.environ.get("TEMPORAL_AGG", "False").lower() == "true"
WANDB_RUN_NAME = os.environ.get("WANDB_RUN_NAME", None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Training parameters
    parser.add_argument('--lr', action='store', type=float, default=LR, help='lr')
    parser.add_argument('--batch_size', action='store', type=int, default=BATCH_SIZE, help='batch size for training')
    parser.add_argument('--num_epochs', action='store', type=int, default=NUM_EPOCHS, help='number of epochs to train for')
    parser.add_argument('--dataset_dir', action='store', type=str, default=DATASET_DIR, help='path to the dataset')
    parser.add_argument('--train_instr_path', action='store', type=str, default=TRAIN_INSTR_PATH, help='path to the csv file containing instruction embeddings for training', required=True)
    parser.add_argument('--val_instr_path', action='store', type=str, default=VAL_INSTR_PATH, help='path to the csv file containing instruction embeddings for validation', required=True)
    parser.add_argument('--ckpt_dir', action='store', type=str, default=CKPT_DIR, help='where to store model checkpoints', required=True)
    parser.add_argument('--ckpt_frequency', action='store', type=int, default=CKPT_FREQUENCY, help='where to store model checkpoints')
    parser.add_argument('--load_ckpt_path', action='store', type=str, default=LOAD_CKPT_PATH, help='if specified, train existing model checkpoint')
    parser.add_argument('--seed', action='store', type=int, default=SEED, help='random seed')

    # Evaluation parameters
    parser.add_argument('--eval', action='store_true', default=EVAL, help='whether to evaluate the model during training')
    parser.add_argument('--eval_instr_path', action='store', type=str, default=EVAL_INSTR_PATH, help='path to the csv file containing instruction embeddings for evaluation')
    parser.add_argument('--eval_frequency', action='store', type=int, default=EVAL_FREQUENCY, help='how many epochs between evaluations')
    parser.add_argument('--eval_wait', action='store', type=int, default=EVAL_WAIT, help='how many epochs to wait before starting evaluations')
    parser.add_argument('--num_rollouts', action='store', type=int, default=NUM_ROLLOUTS, help='number of rollouts to perform during evaluation')
    parser.add_argument('--videos_dir', action='store', type=str, default=VIDEOS_DIR, help='number of rollouts to perform during evaluation')
    parser.add_argument('--onscreen_render', action='store_true', default=ONSCREEN_RENDER, help='whether to render the environment onscreen during evaluation')
    parser.add_argument('--temporal_agg', action='store_true', default=TEMPORAL_AGG, help='whether to use temporal aggregation during evaluation')

    # Parameters for ACT model
    parser.add_argument('--chunk_size', action='store', type=int, default=DEFAULT_MODEL_CONFIG['num_queries'], help='chunk_size')
    parser.add_argument('--kl_weight', action='store', type=int, default=DEFAULT_MODEL_CONFIG['kl_weight'], help='KL Weight')
    parser.add_argument('--hidden_dim', action='store', type=int, default=DEFAULT_MODEL_CONFIG['hidden_dim'], help='hidden_dim')
    parser.add_argument('--dim_feedforward', action='store', type=int, default=DEFAULT_MODEL_CONFIG['dim_feedforward'], help='dim_feedforward')
    parser.add_argument('--state_dim', action='store', type=int, default=DEFAULT_MODEL_CONFIG['state_dim'], help='state_dim')
    parser.add_argument('--lr_backbone', action='store', type=int, default=DEFAULT_MODEL_CONFIG['lr_backbone'], help='lr_backbone')
    parser.add_argument('--backbone', action='store', type=str, default=DEFAULT_MODEL_CONFIG['backbone'], help='backbone')
    parser.add_argument('--enc_layers', action='store', type=int, default=DEFAULT_MODEL_CONFIG['enc_layers'], help='enc_layers')
    parser.add_argument('--dec_layers', action='store', type=int, default=DEFAULT_MODEL_CONFIG['dec_layers'], help='dec_layers')
    parser.add_argument('--nheads', action='store', type=int, default=DEFAULT_MODEL_CONFIG['nheads'], help='nheads')

    # Misc parameters
    parser.add_argument('--wandb_run_name', action='store', type=str, default=WANDB_RUN_NAME, help='wandb run name')

    args = parser.parse_args()

    print("Starting...")
    model_config = {
        'num_queries': args.chunk_size,
        'kl_weight': args.kl_weight,
        'hidden_dim': args.hidden_dim,
        'dim_feedforward': args.dim_feedforward,
        'state_dim': args.state_dim,
        'lr_backbone': args.lr_backbone,
        'backbone': args.backbone,
        'enc_layers': args.enc_layers,
        'dec_layers': args.dec_layers,
        'nheads': args.nheads,
    }

    print("*"*50)
    print("Training parameters:")
    print(f"lr: {args.lr}")
    print(f"batch_size: {args.batch_size}")
    print(f"num_epochs: {args.num_epochs}")
    print(f"dataset_dir: {args.dataset_dir}")
    print(f"train_instr_path: {args.train_instr_path}")
    print(f"val_instr_path: {args.val_instr_path}")
    print(f"ckpt_dir: {args.ckpt_dir}")
    print(f"ckpt_frequency: {args.ckpt_frequency}")
    print(f"eval: {args.eval}")
    print(f"eval_instr_path: {args.eval_instr_path}")
    print(f"eval_frequency: {args.eval_frequency}")
    print(f"eval_wait: {args.eval_wait}")
    print(f"num_rollouts: {args.num_rollouts}")
    print(f"onscreen_render: {args.onscreen_render}")
    print(f"videos_dir: {args.videos_dir}")
    print(f"temporal_agg: {args.temporal_agg}")
    print(f"load_ckpt_path: {args.load_ckpt_path}")
    print(f"seed: {args.seed}")
    print("*"*50)
    print("*"*50)
    print("ACT model parameters:")
    print(model_config)
    print("*"*50)
    
    imitate_multi_task_episodes(
        lr=args.lr,
        num_epochs=args.num_epochs,
        batch_size_train=args.batch_size,
        batch_size_val=args.batch_size,
        ckpt_dir=args.ckpt_dir,
        ckpt_frequency=args.ckpt_frequency,
        do_eval=args.eval,
        eval_instr_path=args.eval_instr_path,
        eval_frequency=args.eval_frequency,
        eval_wait=args.eval_wait,
        num_rollouts=args.num_rollouts,
        onscreen_render=args.onscreen_render,
        temporal_agg=args.temporal_agg,
        videos_dir=args.videos_dir,
        load_ckpt_path=args.load_ckpt_path,
        seed=args.seed,
        dataset_dir=args.dataset_dir,
        train_instr_path=args.train_instr_path,
        val_instr_path=args.val_instr_path,
        model_config=model_config,
        wandb_run_name=args.wandb_run_name,
    )
