# Monte Carlo Tree Diffusion (MCTD)

This is the **official Pytorch implementation** of Monte Carlo Tree Diffusion (MCTD) and Fast Monte Carlo Tree Diffusion (Fast-MCTD).

This code includes the implementation of MCTD and Fast-MCTD for point and ant maze tasks.

## Monte Carlo Tree Diffusion (MCTD)

Author: Jaesik Yoon, Hyeonseo Cho, Yoshua Bengio, Sungjin Ahn

*ICML 2025, Spotlight*

[[MCTD Paper]](https://arxiv.org/abs/2502.07202) [[MCTD project page]](https://sites.google.com/view/mctd-s2planning/home)

![MCTD](./assets/MCTD_overview.png)

Monte Carlo Tree Diffusion (MCTD) is a new framework to improve the inference-time scaling on Diffusion denoising process by combining Monte Carlo Tree Search (MCTS) and Diffusion models.

## Fast Monte Carlo Tree Diffusion (Fast-MCTD)

Author: Jaesik Yoon*, Hyeonseo Cho*, Yoshua Bengio, Sungjin Ahn

![Fast-MCTD](./assets/Fast-MCTD_overview.png)

Fast Monte Carlo Tree Diffusion (Fast-MCTD) is the improved version of MCTD to optimize it's efficiency through parallel tree search and abstract-level diffusion planning.

## Installation

You can set the environment through the given docker file. It includes the installation of the customized version of Offline Goal-Conditioned RL benchmark, [OGBench](https://seohong.me/projects/ogbench/) for giving the velocity information on the maze environment as the observation and removing the randomness on the start and goal positions to reduce the variance of the performance. Before building the docker image, you need to download the mujoco 2.1.0 binary files from the [link](https://drive.google.com/drive/folders/1gwXsIzpTILXG6kZv1EDrLBeEKYpPAl-G?usp=drive_link) and put it in the `./dockerfile/mujoco/` directory.

```bash
docker build -t fmctd:0.1 dockerfile
```

The WanDB is used for logging the training process.

## Evaluation

### Pre-trained models

You can evaluate the performances of MCTDs through the given pre-trained models in the [link](https://drive.google.com/drive/folders/1FoEkB83l1dNShupfKHmucchPJnbcLg3l?usp=share_link).

`dql_trained_models.tar.gz` contains the pre-trained models for DQL, and it should be uncompressed in the `./dql/` directory.

`planner_trained_models.tar.gz` contains the pre-trained diffusion models for point and ant maze tasks. It should be uncompressed in the `./output/downloaded/<WANDB_ENTITY_NAME>/<WANDB_PROJECT_NAME>/` directory.

### Evaluation (Job creation and running)

After uncompressing the files, you can evaluate the performances of MCTDs by running the job creation script and job runner script. The examples of the job creation scripts are in the `insert_point_maze_validation_jobs.py` and `insert_antmaze_validation_jobs.py` files. You can create the job by modifying the wandb entity name, project name, and the job configurations.

After creating the job, you can run the job by running the job runner script. The job runner script is in the `run_jobs.py` file. You can run the jobs over multiple servers by setting the `available_gpus` variable in the script.

### WanDB logs

You can find the WanDB logs for the experiments through example job creation scripts in the [link](https://wandb.ai/jaesikyoon/jaesik_mctd).

### Summarize results

You can summarize the results by running the `summarize_results.py` file by setting the `group_names` variable in the script.

```bash
python summarize_results.py
```

The results will be saved in the `exp_results` directory, and printed out in the terminal as follows.

```bash
{'group': 'PMMN-PMCTD', 'success_rate': '100±0', 'planning_time': '11.11±2.13'}
{'group': 'PMLN-PMCTD', 'success_rate': '98±0', 'planning_time': '8.41±1.34'}
{'group': 'PMGN-PMCTD', 'success_rate': '98±0', 'planning_time': '9.68±0.51'}
{'group': 'PMMN-FMCTD', 'success_rate': '100±0', 'planning_time': '1.91±0.20'}
{'group': 'PMLN-FMCTD', 'success_rate': '82±0', 'planning_time': '2.06±0.08'}
{'group': 'PMGN-FMCTD', 'success_rate': '98±0', 'planning_time': '2.71±0.28'}
```

## Training

You can train the MCTD and Fast-MCTD by running the training scripts. The examples of the training scripts are in the `insert_diffusion_training_jobs.py` and `insert_dql_training_jobs.py` files. You can train the models by modifying the wandb entity name, project name, and the job configurations.

After creating the job, you can run the job by running the job runner script. The job runner script is in the `run_jobs.py` file. You can run the jobs over multiple servers by setting the `available_gpus` variable in the script.

## Acknowledgement

This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research template [repo](https://github.com/buoyancy99/research-template), especially, it is based on Diffusion Forcing source code, [repo](https://github.com/buoyancy99/diffusion-forcing/tree/main).