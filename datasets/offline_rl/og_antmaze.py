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
class OGAntMazeOfflineRLDataset(torch.utils.data.Dataset):
    '''
    '''
        
    def __init__(self, cfg: DictConfig, split: str = "training"):
        super().__init__()
        self.cfg = cfg
        self.save_dir = cfg.save_dir # Using default save_dir, "~/.ogbench/data"
        self.env_id = cfg.env_id
        self.dataset_name = cfg.dataset
        self.split = split
        if self.split != "training":
            self.jump = 1
        else:
            self.jump = cfg.jump
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        self.dataset = self.get_dataset()

        # Infer sample_length from the first terminal index (all episodes equal length).
        term_indices = np.where(self.dataset["terminals"])[0]
        if len(term_indices) == 0:
            raise ValueError(f"No terminal transitions found in dataset '{self.dataset_name}'.")
        # ogbench compact_dataset places consecutive terminal markers at episode boundaries
        # (e.g. indices 199 & 200, 400 & 401, ...).  Use the last in the first run.
        sample_length = int(term_indices[0]) + 1
        if len(term_indices) > 1 and term_indices[1] == term_indices[0] + 1:
            sample_length = int(term_indices[1]) + 1

        # Compute episode_len as a fraction of sample_length (sliding-window slice size).
        ratio = float(getattr(cfg, "extract_episode_ratio_from_sample", 0.5))
        episode_len = int(sample_length * ratio)
        cfg.episode_len = episode_len  # expose to downstream (df_base, df_planning)
        self.n_frames = episode_len + 1

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

        # Dataset Reshaping — keep raw episodes in memory, slice lazily in __getitem__
        self._raw_obs = np.reshape(self.dataset["observations"], (-1, sample_length, self.dataset["observations"].shape[-1]))
        self._raw_actions = np.reshape(self.dataset["actions"], (-1, sample_length, self.dataset["actions"].shape[-1]))
        raw_terminals = np.zeros((self._raw_obs.shape[0], sample_length))
        raw_terminals[:, -1] = 1
        self._raw_rewards = raw_terminals

        self._n_episodes = self._raw_obs.shape[0]
        self._n_windows = sample_length - self.n_frames + 1
        self.total_samples = self._n_episodes * self._n_windows
        self.sample_length = sample_length
        print(f"Lazy dataset: {self._n_episodes} episodes × {self._n_windows} windows = {self.total_samples} samples")
        print(f"Raw obs shape: {self._raw_obs.shape}, Raw actions shape: {self._raw_actions.shape}")

        # Try to cache entire dataset to GPU VRAM for zero-copy training
        self.use_gpu_cache = False
        self._try_gpu_cache()

    def _try_gpu_cache(self):
        """Try to cache raw dataset arrays to GPU VRAM for zero-copy __getitem__.

        If the dataset fits within 70% of currently free VRAM, all arrays are
        moved to CUDA tensors and self.use_gpu_cache is set to True.
        The DataLoader must then use num_workers=0 (CUDA tensors cannot be
        shared across multiprocessing workers).

        Falls back to CPU loading if:
          - CUDA is unavailable
          - Dataset is too large (> 70% of free VRAM)
          - CUDA OOM occurs despite the size check (e.g. fragmentation)
        """
        if not torch.cuda.is_available():
            print("[GPU Cache] CUDA not available. Using CPU loading.")
            return

        obs_bytes = self._raw_obs.nbytes
        act_bytes = self._raw_actions.nbytes
        rew_bytes = self._raw_rewards.nbytes
        total_bytes = obs_bytes + act_bytes + rew_bytes
        total_mb = total_bytes / (1024 ** 2)

        free_vram, total_vram = torch.cuda.mem_get_info()
        free_mb = free_vram / (1024 ** 2)
        threshold_bytes = free_vram * 0.70

        print(f"[GPU Cache] Dataset: {total_mb:.1f} MB | Free VRAM: {free_mb:.1f} MB "
              f"(threshold: {threshold_bytes / (1024**2):.1f} MB = 70% of free)")

        if total_bytes > threshold_bytes:
            print(
                f"[GPU Cache] WARNING: Dataset ({total_mb:.1f} MB) exceeds 70% of free VRAM "
                f"({threshold_bytes / (1024**2):.1f} MB). Falling back to CPU DataLoader. "
                f"To enable GPU caching, reduce dataset size (fewer episodes / shorter episode_len) "
                f"or use a GPU with more VRAM (need ~{total_mb / 0.70:.0f} MB free)."
            )
            return

        print("[GPU Cache] Caching dataset to GPU VRAM ...")
        try:
            self._raw_obs_cuda     = torch.from_numpy(self._raw_obs).float().cuda()
            self._raw_actions_cuda = torch.from_numpy(self._raw_actions).float().cuda()
            self._raw_rewards_cuda = torch.from_numpy(self._raw_rewards).float().cuda()

            # Precompute nonterminal template — shape is fixed for all __getitem__ calls
            n_frames_out = len(range(0, self.n_frames, self.jump))
            done_tmpl = torch.zeros(n_frames_out, dtype=torch.bool, device="cuda")
            done_tmpl[-1] = True
            self._nonterminal_template = ~done_tmpl

            self.use_gpu_cache = True
            used_mb = (self._raw_obs_cuda.element_size() * self._raw_obs_cuda.nelement()
                       + self._raw_actions_cuda.element_size() * self._raw_actions_cuda.nelement()
                       + self._raw_rewards_cuda.element_size() * self._raw_rewards_cuda.nelement()) / (1024 ** 2)
            print(f"[GPU Cache] Dataset cached to GPU ({used_mb:.1f} MB). "
                  f"DataLoader will run with num_workers=0.")

        except torch.cuda.OutOfMemoryError:
            # Clean up partial allocations before falling back
            for attr in ("_raw_obs_cuda", "_raw_actions_cuda", "_raw_rewards_cuda", "_nonterminal_template"):
                if hasattr(self, attr):
                    delattr(self, attr)
            torch.cuda.empty_cache()
            print(
                f"[GPU Cache] WARNING: CUDA OOM while caching ({total_mb:.1f} MB). "
                f"This can happen due to VRAM fragmentation. Falling back to CPU loading."
            )

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        ep = idx // self._n_windows
        w  = idx %  self._n_windows
        if self.use_gpu_cache:
            # All data already lives on GPU — slice and make contiguous (fast GPU memcpy)
            observation = self._raw_obs_cuda[ep, w:w+self.n_frames:self.jump].contiguous()
            action      = self._raw_actions_cuda[ep, w:w+self.n_frames:self.jump].contiguous()
            reward      = self._raw_rewards_cuda[ep, w:w+self.n_frames:self.jump].contiguous()
            nonterminal = self._nonterminal_template  # precomputed, same shape every call
        else:
            observation = torch.from_numpy(self._raw_obs[ep, w:w+self.n_frames:self.jump].copy()).float()
            action      = torch.from_numpy(self._raw_actions[ep, w:w+self.n_frames:self.jump].copy()).float()
            reward      = torch.from_numpy(self._raw_rewards[ep, w:w+self.n_frames:self.jump].copy()).float()
            done = np.zeros(len(observation), dtype=bool)
            done[-1] = True
            nonterminal = torch.from_numpy(~done)
        return observation, action, reward, nonterminal

    def get_dataset(self):
        _, train_dataset, val_dataset = ogbench.make_env_and_datasets(
            self.dataset_name,
            self.save_dir, 
            compact_dataset=True,
        )
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
        "antmaze-medium-navigate-v0",
        "antmaze-medium-stitch-v0",
        "antmaze-medium-explore-v0",
        "antmaze-large-navigate-v0",
        "antmaze-large-stitch-v0",
        "antmaze-large-explore-v0",
        "antmaze-giant-navigate-v0",
        "antmaze-giant-stitch-v0",
        "antmaze-teleport-navigate-v0",
        "antmaze-teleport-stitch-v0",
        "antmaze-teleport-explore-v0",
    ]
    for dataset in dataset_list:
        cfg.dataset = dataset
        #cfg.save_dir = ".ogbench/datasets" # Using default save_dir, "~/.ogbench/data"
        cfg.env_id = "-".join(dataset.split("-")[:2] + ["v0"])
        if "navigate" in cfg.dataset:
            if "giant" in cfg.dataset:
                cfg.episode_len = 1000
            else:
                cfg.episode_len = 500
        elif "stitch" in cfg.dataset:
            cfg.episode_len = 100
        elif "explore" in cfg.dataset:
            cfg.episode_len = 250
        cfg.gamma = 1.0
        dataset = OGAntMazeOfflineRLDataset(cfg)
    exit(1)

    # Dataset Visualization
    obs = dataset.dataset["observations"]
    def convert_maze_string_to_grid(maze_string):
        lines = maze_string.split("\\")
        grid = [line[1:-1] for line in lines]
        return grid[1:-1]
    if "medium" in cfg.dataset:
        maze_string = "########\\#OO#OOO#\\#OOOO#O#\\###O#OO#\\##OOOO##\\#OO#O#O#\\#OO#OOO#\\########"
    elif "large" in cfg.dataset:
        maze_string = "#########\\#OOOOO#O#\\#O#O#OOO#\\#O#O#####\\#OOO#OOO#\\###O###O#\\#OOOOOOO#\\#O###O###\\#OOO#OOO#\\#O#O#O#O#\\#OOOOO#O#\\#########"
    elif "giant" in cfg.dataset:
        maze_string = "############\\#OOOOO#OOOO#\\###O#O#O##O#\\#OOO#OOOO#O#\\#O########O#\\#O#OOOOOOOO#\\#OOO#O#O#O##\\#O###OO##OO#\\#OOO##OO##O#\\###O#O#O#OO#\\##OO#OOO#O##\\#OO##O###OO#\\#O#OOOOOO#O#\\#O#O###O##O#\\#OOOOO#OOOO#\\############"
    elif "teleport" in cfg.dataset:
        maze_string = "#########\\#O##OOOO#\\#OOOO##O#\\#O##O##O#\\#OO#OOOO#\\#OO######\\##OOOOOO#\\#O#O###O#\\##OOOOOO#\\#OOO#####\\#O#OOOOO#\\#########"
    grid = convert_maze_string_to_grid(maze_string)
    plt.figure()
    steps = 2000
    plt.scatter(obs[:steps, 0]/4+1, obs[:steps, 1]/4+1, c=np.arange(len(obs[:steps])), cmap="Reds")
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell == "#":
                square = plt.Rectangle((i + 0.5, j + 0.5), 1, 1, edgecolor="black", facecolor="black")
                plt.gca().add_patch(square)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.gca().set_facecolor("lightgray")
    plt.gca().set_axisbelow(True)
    plt.gca().set_xticks(np.arange(1, len(grid), 0.5), minor=True)
    plt.gca().set_yticks(np.arange(1, len(grid[0]), 0.5), minor=True)
    plt.xlim([0.5, len(grid) + 0.5])
    plt.ylim([0.5, len(grid[0]) + 0.5])
    plt.tick_params(
        axis="both", which="both", bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False
    )
    plt.grid(True, color="white", which="minor", linewidth=4)
    plt.gca().spines["top"].set_linewidth(4)
    plt.gca().spines["right"].set_linewidth(4)
    plt.gca().spines["bottom"].set_linewidth(4)
    plt.gca().spines["left"].set_linewidth(4)
    plt.show()
    save_path = Path(f"./datasets/offline_rl/viz")
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path / f"{cfg.dataset}.png")
    print("Done.");exit(1)

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