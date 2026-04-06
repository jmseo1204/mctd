from datasets import Maze2dOfflineRLDataset, OGMaze2dOfflineRLDataset, OGAntMazeOfflineRLDataset 
from algorithms.diffusion_forcing import DiffusionForcingPlanning
from .exp_base import BaseLightningExperiment


class PlanningExperiment(BaseLightningExperiment):
    """
    A Partially Observed Markov Decision Process experiment
    """

    compatible_algorithms = dict(
        df_planning=DiffusionForcingPlanning,
        df_planning_2d=DiffusionForcingPlanning,
        df_planning_15d=DiffusionForcingPlanning,
        train_df_planning=DiffusionForcingPlanning,
    )

    compatible_datasets = dict(
        # Planning datasets
        maze2d_umaze=Maze2dOfflineRLDataset,
        maze2d_medium=Maze2dOfflineRLDataset,
        maze2d_large=Maze2dOfflineRLDataset,

        og_maze2d_medium_navigate=OGMaze2dOfflineRLDataset,
        og_maze2d_large_navigate=OGMaze2dOfflineRLDataset,
        og_maze2d_giant_navigate=OGMaze2dOfflineRLDataset,

        antmaze_medium_navigate=OGAntMazeOfflineRLDataset,
        antmaze_large_navigate=OGAntMazeOfflineRLDataset,
        antmaze_giant_navigate=OGAntMazeOfflineRLDataset,
        antmaze_giant_navigate_fullstate=OGAntMazeOfflineRLDataset,
        antmaze_giant_navigate_15d=OGAntMazeOfflineRLDataset,
        antmaze_teleport_navigate=OGAntMazeOfflineRLDataset,

        antmaze_giant_stitch=OGAntMazeOfflineRLDataset,
        antmaze_large_stitch=OGAntMazeOfflineRLDataset,
        antmaze_medium_stitch=OGAntMazeOfflineRLDataset,

        cube_single_play=OGAntMazeOfflineRLDataset,
        cube_double_play=OGAntMazeOfflineRLDataset,

        pointmaze_giant_stitch=OGAntMazeOfflineRLDataset,
        pointmaze_large_stitch=OGAntMazeOfflineRLDataset,
        pointmaze_medium_stitch=OGAntMazeOfflineRLDataset,
    )
