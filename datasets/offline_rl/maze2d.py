from pathlib import Path
import numpy as np
import torch
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
import urllib
import os


def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


def download_dataset_from_url(save_dir, dataset_url):
    _, dataset_name = os.path.split(dataset_url)
    dataset_filepath = os.path.join(save_dir, dataset_name)
    if not os.path.exists(dataset_filepath):
        print("Downloading dataset:", dataset_url, "to", dataset_filepath)
        urllib.request.urlretrieve(dataset_url, dataset_filepath)
    if not os.path.exists(dataset_filepath):
        raise IOError("Failed to download dataset from %s" % dataset_url)
    return dataset_filepath


class Maze2dOfflineRLDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: DictConfig, split: str = "training"):
        super().__init__()
        self.cfg = cfg
        self.save_dir = cfg.save_dir
        self.dataset_url = cfg.dataset_url
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        self.dataset = self.get_dataset()
        self.gamma = cfg.gamma
        self.n_frames = cfg.episode_len + 1
        self.total_steps = len(self.dataset["observations"])
        self.dataset["values"] = self.compute_value(self.dataset["rewards"]) * (1 - self.gamma) * 4 - 1

    def compute_value(self, reward):
        # numerical stable way to compute value
        value = np.copy(reward)
        for i in range(len(reward) - 2, -1, -1):
            value[i] += self.gamma * value[i + 1]
        return value

    def __len__(self):
        return self.total_steps - self.n_frames + 1

    def __getitem__(self, idx):
        observation = torch.from_numpy(self.dataset["observations"][idx : idx + self.n_frames]).float()
        action = torch.from_numpy(self.dataset["actions"][idx : idx + self.n_frames]).float()
        reward = torch.from_numpy(self.dataset["rewards"][idx : idx + self.n_frames]).float()
        value = torch.from_numpy(self.dataset["values"][idx : idx + self.n_frames]).float()

        done = np.zeros(self.n_frames, dtype=bool)
        done[-1] = True
        nonterminal = torch.from_numpy(~done)

        goal = torch.zeros((self.n_frames, 0))

        return observation, action, reward, nonterminal

    def get_dataset(self):
        h5path = download_dataset_from_url(self.save_dir, self.dataset_url)
        data_dict = {}
        with h5py.File(h5path, "r") as dataset_file:
            for k in get_keys(dataset_file):
                try:  # first try loading as an array
                    data_dict[k] = dataset_file[k][:]
                except ValueError as e:  # try loading as a scalar
                    data_dict[k] = dataset_file[k][()]

        N_samples = data_dict["observations"].shape[0]

        if data_dict["rewards"].shape == (N_samples, 1):
            data_dict["rewards"] = data_dict["rewards"][:, 0]

        if data_dict["terminals"].shape == (N_samples, 1):
            data_dict["terminals"] = data_dict["terminals"][:, 0]

        return data_dict


if __name__ == "__main__":
    from unittest.mock import MagicMock
    import os
    import matplotlib.pyplot as plt
    import gym
    import d4rl

    os.chdir("../..")
    cfg = MagicMock()
    cfg.env_id = "maze2d-large-v1"
    cfg.episode_len = 800
    cfg.gamma = 1.0
    cfg.save_dir = 'data/maze2d'
    cfg.dataset_url = 'http://rail.eecs.berkeley.edu/datasets/offline_rl/maze2d/maze2d-large-sparse-v1.hdf5'
    dataset = Maze2dOfflineRLDataset(cfg)
    o = dataset.dataset["observations"][:, :2]
    ## Case 1
    #start_x, start_y = 2.9793, 9.0801
    #goal_x, goal_y = 7.0605, 1.0485
    # Case 2
    start_x, start_y = 4.9589, 4.0713
    goal_x, goal_y = 3.0443, 1.9158
    # when the indexes are continuous, then store the last one only
    raw_start_idx = np.where(np.linalg.norm(o - [start_x, start_y], axis=1) < 0.2)[0]
    start_idx = []
    for i in range(len(raw_start_idx)):
        if i == 0 or raw_start_idx[i] - raw_start_idx[i-1] != 1:
            start_idx.append(raw_start_idx[i])
    # when the indexes are continuous, then store the last one only
    raw_end_idx = np.where(np.linalg.norm(o - [goal_x, goal_y], axis=1) < 0.2)[0]
    end_idx = []
    for i in range(len(raw_end_idx)):
        if i == 0 or raw_end_idx[i] - raw_end_idx[i-1] != 1:
            end_idx.append(raw_end_idx[i])
    # start-end pair which length is shortest
    pairs = []
    start_i, end_i = 0, 0
    while True:
        if start_i >= len(start_idx) or end_i >= len(end_idx):
            break
        _start_idx = start_idx[start_i]
        _end_idx = end_idx[end_i]
        for i in range(start_i, len(start_idx)):
            start_i = i
            _start_idx = start_idx[i]
            if start_idx[i] > _end_idx:
                break
        for i in range(end_i, len(end_idx)):
            end_i = i
            _end_idx = end_idx[i]
            if end_idx[i] > _start_idx:
                break
        pairs.append((_start_idx, _end_idx))
        start_i += 1
        end_i += 1
    plt.figure()
    for start_idx, end_idx in pairs:
        #print(f"Length: {end_idx - start_idx + 1}")
        #if end_idx - start_idx + 1 > 800:
        if end_idx - start_idx + 1 > 400:
            continue
        plt.scatter(o[start_idx:(end_idx+1), 0], o[start_idx:(end_idx+1), 1], c=np.arange(end_idx-start_idx+1), cmap="Reds", s=1)
    #plt.scatter(o[:, 0], o[:, 1], c=np.arange(len(o)), cmap="Reds", s=1)

    def convert_maze_string_to_grid(maze_string):
        lines = maze_string.split("\\")
        grid = [line[1:-1] for line in lines]
        return grid[1:-1]

    maze_string = gym.make(cfg.env_id).str_maze_spec
    grid = convert_maze_string_to_grid(maze_string)

    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell == "#":
                square = plt.Rectangle((i + 0.5, j + 0.5), 1, 1, edgecolor="black", facecolor="black")
                plt.gca().add_patch(square)

    #start_x, start_y = o[..., 0, :2]
    start_circle = plt.Circle((start_x, start_y), 0.16, facecolor="white", edgecolor="black")
    plt.gca().add_patch(start_circle)
    inner_circle = plt.Circle((start_x, start_y), 0.08, color="black")
    plt.gca().add_patch(inner_circle)

    def draw_star(center, radius, num_points=5, color="black"):
        angles = np.linspace(0.0, 2 * np.pi, num_points, endpoint=False) + 5 * np.pi / (2 * num_points)
        inner_radius = radius / 2.0

        points = []
        for angle in angles:
            points.extend(
                [
                    center[0] + radius * np.cos(angle),
                    center[1] + radius * np.sin(angle),
                    center[0] + inner_radius * np.cos(angle + np.pi / num_points),
                    center[1] + inner_radius * np.sin(angle + np.pi / num_points),
                ]
            )

        star = plt.Polygon(np.array(points).reshape(-1, 2), color=color)
        plt.gca().add_patch(star)

    #goal_x, goal_y = o[..., -1, :2]
    goal_circle = plt.Circle((goal_x, goal_y), 0.16, facecolor="white", edgecolor="black")
    plt.gca().add_patch(goal_circle)
    draw_star((goal_x, goal_y), radius=0.08)

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
    plt.savefig(save_path / "maze2d_large_v1.png")
    print("Done.")

    #os.chdir("../..")
    #cfg = MagicMock()
    #cfg.env_id = "maze2d-medium-v1"
    #cfg.episode_len = 600
    #cfg.gamma = 1.0
    #dataset = Maze2dOfflineRLDataset(cfg)
    #o, a, r, n = dataset.__getitem__(10)
    #print(o.shape, a.shape, r.shape, n.shape)
    #plt.figure()
    #plt.scatter(o[:, 0], o[:, 1], c=np.arange(len(o)), cmap="Reds")

    #def convert_maze_string_to_grid(maze_string):
    #    lines = maze_string.split("\\")
    #    grid = [line[1:-1] for line in lines]
    #    return grid[1:-1]

    #maze_string = gym.make(cfg.env_id).str_maze_spec
    #grid = convert_maze_string_to_grid(maze_string)

    #for i, row in enumerate(grid):
    #    for j, cell in enumerate(row):
    #        if cell == "#":
    #            square = plt.Rectangle((i + 0.5, j + 0.5), 1, 1, edgecolor="black", facecolor="black")
    #            plt.gca().add_patch(square)

    #start_x, start_y = o[..., 0, :2]
    #start_circle = plt.Circle((start_x, start_y), 0.16, facecolor="white", edgecolor="black")
    #plt.gca().add_patch(start_circle)
    #inner_circle = plt.Circle((start_x, start_y), 0.08, color="black")
    #plt.gca().add_patch(inner_circle)

    #def draw_star(center, radius, num_points=5, color="black"):
    #    angles = np.linspace(0.0, 2 * np.pi, num_points, endpoint=False) + 5 * np.pi / (2 * num_points)
    #    inner_radius = radius / 2.0

    #    points = []
    #    for angle in angles:
    #        points.extend(
    #            [
    #                center[0] + radius * np.cos(angle),
    #                center[1] + radius * np.sin(angle),
    #                center[0] + inner_radius * np.cos(angle + np.pi / num_points),
    #                center[1] + inner_radius * np.sin(angle + np.pi / num_points),
    #            ]
    #        )

    #    star = plt.Polygon(np.array(points).reshape(-1, 2), color=color)
    #    plt.gca().add_patch(star)

    #goal_x, goal_y = o[..., -1, :2]
    #goal_circle = plt.Circle((goal_x, goal_y), 0.16, facecolor="white", edgecolor="black")
    #plt.gca().add_patch(goal_circle)
    #draw_star((goal_x, goal_y), radius=0.08)

    #plt.gca().set_aspect("equal", adjustable="box")
    #plt.gca().set_facecolor("lightgray")
    #plt.gca().set_axisbelow(True)
    #plt.gca().set_xticks(np.arange(1, len(grid), 0.5), minor=True)
    #plt.gca().set_yticks(np.arange(1, len(grid[0]), 0.5), minor=True)
    #plt.xlim([0.5, len(grid) + 0.5])
    #plt.ylim([0.5, len(grid[0]) + 0.5])
    #plt.tick_params(
    #    axis="both", which="both", bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False
    #)
    #plt.grid(True, color="white", which="minor", linewidth=4)
    #plt.gca().spines["top"].set_linewidth(4)
    #plt.gca().spines["right"].set_linewidth(4)
    #plt.gca().spines["bottom"].set_linewidth(4)
    #plt.gca().spines["left"].set_linewidth(4)
    #plt.show()
    #print("Done.")
