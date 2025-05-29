from pathlib import Path
import numpy as np
import torch
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
import urllib
import os
import ogbench
class OGCubeOfflineRLDataset(torch.utils.data.Dataset):
    '''
    '''
        
    def __init__(self, cfg: DictConfig, split: str = "training"):
        super().__init__()
        self.cfg = cfg
        #self.save_dir = cfg.save_dir # Using default save_dir, "~/.ogbench/data"
        self.env_id = cfg.env_id
        self.dataset_name = cfg.dataset
        self.n_frames = cfg.episode_len + 1
        self.gamma = cfg.gamma
        self.split = split
        if self.split != "training":
            self.jump = 1
        else:
            self.jump = cfg.jump

        #Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        self.dataset = self.get_dataset()
        # Use [position, velocity] as observation
        #self.dataset["observations"] = np.concatenate([self.dataset["qpos"], self.dataset["qvel"]], axis=-1)
        # Reshaping the observation
        joint_pos = self.dataset["observations"][:, :6]
        joint_vel = self.dataset["observations"][:, 6:12]
        effector_pos = self.dataset["observations"][:, 12:15]
        effector_yaw = self.dataset["observations"][:, 15:17] # [cos(theta), sin(theta)]
        gripper_opening = self.dataset["observations"][:, 17:18]
        gripper_contact = self.dataset["observations"][:, 18:19] # 0: no contact, 1: contact
        block_obs = []
        num_blocks = (self.dataset["observations"].shape[1] - 19) // 9
        for i in range(num_blocks):
            _block_obs = []
            _block_obs.append(self.dataset["observations"][:, 19 + i*9:19 + i*9 + 3]) # block_pos
            _block_obs.append(self.dataset["observations"][:, 19 + i*9 + 3:19 + i*9 + 7]) # block_quat
            _block_obs.append(self.dataset["observations"][:, 19 + i*9 + 7:19 + i*9 + 9]) # block_yaw
            block_obs.append(_block_obs)
        # Use effector_pos and block_pos as observation
        #self.dataset["observations"] = effector_pos
        #self.dataset["observations"] = np.concatenate([effector_pos] + [block_obs[i][0] for i in range(num_blocks)], axis=-1)
        self.dataset["observations"] = np.concatenate([block_obs[i][0] for i in range(num_blocks)], axis=-1)
        sample_length = 1001

        if self.n_frames > sample_length:
            self.n_frames = sample_length
        #assert self.n_frames <= sample_length, f"Episode length {self.n_frames} is greater than sample length {sample_length}"

        # Dataset Statistics
        print(f"Dataset: {self.dataset_name}")
        print(f"Total samples: {len(self.dataset['observations'])}, Subtrajectory length: {sample_length}")
        obs_mean = np.mean(self.dataset["observations"], axis=0)
        obs_std = np.std(self.dataset["observations"], axis=0)
        print(f"Observation shape: {self.dataset['observations'].shape}")
        print(f"observation_mean: [{','.join(np.array(obs_mean,dtype=str))}]")
        print(f"observation_std:  [{','.join(np.array(obs_std,dtype=str))}]")
        act_mean = np.mean(self.dataset["actions"], axis=0)
        act_std = np.std(self.dataset["actions"], axis=0)
        print(f"Action shape: {self.dataset['actions'].shape}")
        print(f"action_mean: [{','.join(np.array(act_mean,dtype=str))}]")
        print(f"action_std:  [{','.join(np.array(act_std,dtype=str))}]")

        # Dataset Reshaping
        raw_observations = np.reshape(self.dataset["observations"], (-1, sample_length, self.dataset["observations"].shape[-1]))
        raw_actions = np.reshape(self.dataset["actions"], (-1, sample_length, self.dataset["actions"].shape[-1]))
        raw_terminals = np.zeros((raw_observations.shape[0], sample_length)) # This will not be used in training and validation
        raw_terminals[:, -1] = 1
        raw_rewards = np.copy(raw_terminals)
        raw_values = self.compute_value(raw_rewards) * (1 - self.gamma) * 4 - 1

        # Dataset Preprocessing (Collecting episode_len trajectories in sliding window manner)
        self.observations, self.actions, self.rewards, self.values = [], [], [], []
        for i in range(raw_observations.shape[0]):
            for j in range(sample_length - self.n_frames + 1):
                self.observations.append(raw_observations[i, j:j+self.n_frames:self.jump])
                self.actions.append(raw_actions[i, j:j+self.n_frames:self.jump])
                self.rewards.append(raw_rewards[i, j:j+self.n_frames:self.jump])
                self.values.append(raw_values[i, j:j+self.n_frames:self.jump])
        self.observations = np.array(self.observations)
        self.actions = np.array(self.actions)
        self.rewards = np.array(self.rewards)
        self.values = np.array(self.values)
        
        # Preprocessed Dataset Statistics
        print(f"Preprocessed Dataset Statistics")
        print(f"Observation shape: {self.observations.shape}")
        print(f"Action shape: {self.actions.shape}")
        print(f"Reward shape: {self.rewards.shape}")
        print(f"Value shape: {self.values.shape}")
        #exit(1)

        self.total_samples = len(self.observations)

    def compute_value(self, reward):
        # numerical stable way to compute value
        value = np.copy(reward)
        for i in range(reward.shape[1] - 2, -1, -1):
            value[:, i] += self.gamma * value[:, i + 1]
        return value

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        observation = torch.from_numpy(self.observations[idx]).float() # (episode_len, obs_dim)
        action = torch.from_numpy(self.actions[idx]).float() # (episode_len, act_dim)
        reward = torch.from_numpy(self.rewards[idx]).float() # (episode_len,)
        value = torch.from_numpy(self.values[idx]).float() # (episode_len,)
        done = np.zeros(len(observation), dtype=bool)
        done[-1] = True
        nonterminal = torch.from_numpy(~done)
        return observation, action, reward, nonterminal

    def get_dataset(self):
        env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
            self.dataset_name,
            #self.save_dir, # Using default save_dir, "~/.ogbench/data"
            compact_dataset=True,
        )
        obs, info = env.reset()
        n_dim = 0
        for k, v in info.items():
            if isinstance(v, np.ndarray):
                print(f"{k}: {v.shape}, {v.shape[0] + n_dim}")
                n_dim += v.shape[0]
            else:
                print(f"{k}: {v}")
        if self.split == "training":
            return train_dataset
        else:
            return val_dataset

if __name__ == "__main__":
    from unittest.mock import MagicMock
    import os
    import matplotlib.pyplot as plt
    import gymnasium 
    os.chdir("../..")
    cfg = MagicMock()
    dataset_list = [
        #"visual-cube-single-play-v0",
        #"cube-single-play-v0",
        "cube-double-play-v0",
        #"cube-triple-play-v0",
        #"cube-quadruple-play-v0"
    ]
    for dataset in dataset_list:
        cfg.dataset = dataset
        #cfg.save_dir = ".ogbench/datasets" # Using default save_dir, "~/.ogbench/data"
        cfg.env_id = "-".join(dataset.split("-")[:2] + ["v0"])
        cfg.episode_len = 1000
        cfg.gamma = 1.0
        dataset = OGCubeOfflineRLDataset(cfg)
    exit(1)

    ## Dataset Visualization
    #obs = dataset.dataset["observations"][:300]
    #joint1_positions = obs[:, :3]
    #joint2_positions = obs[:, 3:6]
    #effector_positions = obs[:, 12:15]
    #block_positions = obs[:, 19:22]
    ## Showing it as gif
    #import imageio
    #images = []
    #for i in range(len(effector_positions)):
    #    # 3D
    #    fig = plt.figure()
    #    ax = fig.add_subplot(111, projection='3d')
    #    ax.scatter(joint1_positions[i, 0], joint1_positions[i, 1], joint1_positions[i, 2], c="green")
    #    ax.scatter(joint2_positions[i, 0], joint2_positions[i, 1], joint2_positions[i, 2], c="violet")
    #    ax.scatter(effector_positions[i, 0], effector_positions[i, 1], effector_positions[i, 2], c="red")
    #    ax.scatter(block_positions[i, 0], block_positions[i, 1], block_positions[i, 2], c="blue")
    #    ax.set_xlim(-10, 10)
    #    ax.set_ylim(-10, 10)
    #    ax.set_zlim(-10, 10)
    #    ax.set_xlabel('X')
    #    ax.set_ylabel('Y')
    #    ax.set_zlabel('Z')
    #    ax.view_init(elev=20, azim=30)
    #    plt.title(f"Frame {i}")
    #    plt.savefig(f"datasets/offline_rl/viz/tmp.png")
    #    plt.close()
    #    images.append(imageio.imread(f"datasets/offline_rl/viz/tmp.png"))
    #imageio.mimsave("datasets/offline_rl/viz/movie.gif", images)
    #exit(1)

    plt.figure()
    ## 3D plot
    #ax = plt.axes(projection='3d')
    ##ax.scatter3D(effector_positions[:, 0], effector_positions[:, 1], effector_positions[:, 2], c=np.arange(len(effector_positions)), cmap="Reds")
    #ax.scatter3D(block_positions[:, 0], block_positions[:, 1], block_positions[:, 2], c=np.arange(len(block_positions)), cmap="Blues")
    ## 2D plot
    #ax = plt.axes()
    #ax.scatter(block_positions[:, 0], block_positions[:, 1], c=np.arange(len(block_positions)), cmap="Blues")
    #plt.show()
    #save_path = Path(f"./datasets/offline_rl/viz")
    #save_path.mkdir(parents=True, exist_ok=True)
    #plt.savefig(save_path / f"{cfg.dataset}.png")
    #print("Done.");exit(1)

    o, a, r, n = dataset.__getitem__(10)
    print(o.shape, a.shape, r.shape, n.shape)
    plt.figure()
    plt.scatter(o[:, 0], o[:, 1], c=np.arange(len(o)), cmap="Reds")
    def convert_maze_string_to_grid(maze_string):
        lines = maze_string.split("\\")
        grid = [line[1:-1] for line in lines]
        return grid[1:-1]
    import os
    
    #os.environ['MUJOCO_GL'] = 'egl'
    #if 'SLURM_STEP_GPUS' in os.environ:
    #    os.environ['EGL_DEVICE_ID'] = os.environ['SLURM_STEP_GPUS']
    #ogbench.locomaze.maze.make_maze_env('point','maze',maze_type='giant')
    maze_string = "################\\#O#OOOOOO##OOOO#\\#O#O##O#O#OO##O#\\#OOO#OO#OOO#OOO#\\#O###O######O#O#\\#OOO#OOO#OOOO#O#\\###O#O#OO#O#O###\\#OOO#OO#OOO#OOO#\\#O#O#O######O#O#\\#O###OOO#OOO##O#\\#OOOOO#OOO#OOOO#\\################"
    grid = convert_maze_string_to_grid(maze_string)