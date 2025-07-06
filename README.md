# Monte Carlo Tree Diffusion (MCTD)

This repository provides the **official PyTorch implementation** of Monte Carlo Tree Diffusion (MCTD) and Fast Monte Carlo Tree Diffusion (Fast-MCTD) for the point and ant maze planning tasks.

## Monte Carlo Tree Diffusion (MCTD)

Author: Jaesik Yoon, Hyeonseo Cho, Doojin Baek, Yoshua Bengio, Sungjin Ahn

*ICML 2025, Spotlight*

[[MCTD Paper]](https://arxiv.org/abs/2502.07202) [[MCTD project page]](https://sites.google.com/view/mctd-s2planning/home)

![MCTD](./assets/MCTD_overview.png)

Monte Carlo Tree Diffusion (MCTD) is a novel framework that improves the inference-time performance of diffusion models by integrating the denoising process with Monte Carlo Tree Search (MCTS).

## Fast Monte Carlo Tree Diffusion (Fast-MCTD)

Author: Jaesik Yoon*, Hyeonseo Cho*, Yoshua Bengio, Sungjin Ahn

![Fast-MCTD](./assets/Fast-MCTD_overview.png)

Fast Monte Carlo Tree Diffusion (Fast-MCTD) is an enhanced version of MCTD that improves computational efficiency through parallel tree search and abstract-level diffusion planning.

## Installation

The recommended method for setting up the environment is to use the provided Dockerfile.

The Dockerfile installs a customized version of the [OGBench](https://seohong.me/projects/ogbench/) benchmark. This customization serves two purposes: it incorporates velocity into the maze environment's observation space and removes randomness from the start and goal positions to reduce performance variance.

Before building the Docker image, download the MuJoCo 2.1.0 binaries from this [link](https://drive.google.com/drive/folders/1gwXsIzpTILXG6kZv1EDrLBeEKYpPAl-G?usp=drive_link) and place them in the `./dockerfile/mujoco/` directory.

```bash
docker build -t fmctd:0.1 dockerfile
```

This project uses Weights & Biases (WanDB) for logging experiments.

## Evaluation

### Pre-trained models

Evaluate the performance of MCTD and Fast-MCTD using the pre-trained models available at this [link](https://drive.google.com/drive/folders/1FoEkB83l1dNShupfKHmucchPJnbcLg3l?usp=share_link).

- `dql_trained_models.tar.gz`: Contains pre-trained models for DQL. Extract this archive to the `./dql/` directory.

- `planner_trained_models.tar.gz`: Contains pre-trained diffusion models for the point and ant maze tasks. Extract this archive to `./output/downloaded/<WANDB_ENTITY_NAME>/<WANDB_PROJECT_NAME>/`.

### Running the Evaluation

1. Create Jobs: After extracting the models, define the evaluation jobs. Example scripts, `insert_point_maze_validation_jobs.py` and `insert_antmaze_validation_jobs.py`, are provided. You will need to modify your WandB entity, project name, and other job configurations within these files.

2. Run Jobs: Execute the created jobs using the `run_jobs.py` script. To distribute jobs across multiple servers, configure the `available_gpus` variable within the script.

### WanDB logs

The Weights & Biases logs for the experiments reported in our paper are publicly available at this [link](https://wandb.ai/jaesikyoon/jaesik_mctd). These logs correspond to the configurations in the example job creation scripts.

### Summarize results

To aggregate the evaluation results, run the `summarize_results.py` script after setting the `group_names` variable.

```bash
python summarize_results.py
```

The results will be saved to the `exp_results` directory and printed to the terminal, as shown below:
```bash
{'group': 'PMMN-PMCTD', 'success_rate': '100±0', 'planning_time': '11.11±2.13'}
{'group': 'PMLN-PMCTD', 'success_rate': '98±0', 'planning_time': '8.41±1.34'}
{'group': 'PMGN-PMCTD', 'success_rate': '98±0', 'planning_time': '9.68±0.51'}
{'group': 'PMMN-FMCTD', 'success_rate': '100±0', 'planning_time': '1.91±0.20'}
{'group': 'PMLN-FMCTD', 'success_rate': '82±0', 'planning_time': '2.06±0.08'}
{'group': 'PMGN-FMCTD', 'success_rate': '98±0', 'planning_time': '2.71±0.28'}
```

## Training

To train MCTD and Fast-MCTD models from scratch, follow these steps:

1. Create Jobs: Use the example scripts `insert_diffusion_training_jobs.py` and `insert_dql_training_jobs.py` to create training jobs. Before running, you must configure your WandB entity, project name, and job parameters within these scripts.

2. Run Jobs: Once a job is created, execute it using the `run_jobs.py` script. To distribute training across multiple servers, configure the `available_gpus` variable within the script.

## References

```bibtex
@inproceedings{mctd,
  title={Monte Carlo Tree Diffusion for System 2 Planning},
  author={Yoon, Jaesik and Cho, Hyeonseo and Baek, Doojin and Bengio, Yoshua and Ahn, Sungjin},
  booktitle={International Conference on Machine Learning},
  year={2025},
}
```

```bibtex
@article{fast-mctd,
  title={Fast Monte Carlo Tree Diffusion: 100x Speedup via Parallel Sparse Planning},
  author={Yoon, Jaesik* and Cho, Hyeonseo* and Bengio, Yoshua and Ahn, Sungjin},
  journal={arXiv preprint arXiv:2506.09498},
  year={2025}
}
```

## Acknowledgement
This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research template [repo](https://github.com/buoyancy99/research-template), especially, it is based on Diffusion Forcing source code, [repo](https://github.com/buoyancy99/diffusion-forcing/tree/main).
