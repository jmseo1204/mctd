from datasets import Maze2dOfflineRLDataset, OGMaze2dOfflineRLDataset, OGAntMazeOfflineRLDataset 
from algorithms.diffusion_forcing import DiffusionForcingPlanning
from .exp_base import BaseLightningExperiment


class PlanningExperiment(BaseLightningExperiment):
    """
    A Partially Observed Markov Decision Process experiment
    """

    compatible_algorithms = dict(
        df_planning=DiffusionForcingPlanning,
    )

    compatible_datasets = dict(
        # Planning datasets
        maze2d_umaze=Maze2dOfflineRLDataset,
        maze2d_medium=Maze2dOfflineRLDataset,
        maze2d_large=Maze2dOfflineRLDataset,

        og_maze2d_medium_navigate=OGMaze2dOfflineRLDataset,
        og_maze2d_large_navigate=OGMaze2dOfflineRLDataset,
        og_maze2d_giant_navigate=OGMaze2dOfflineRLDataset,

        og_antmaze_medium_navigate=OGAntMazeOfflineRLDataset,
        og_antmaze_large_navigate=OGAntMazeOfflineRLDataset,
        og_antmaze_giant_navigate=OGAntMazeOfflineRLDataset,
        og_antmaze_teleport_navigate=OGAntMazeOfflineRLDataset,
    )
