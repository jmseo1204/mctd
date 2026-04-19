# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import argparse
import gym
import numpy as np
import os
import torch
import json
import re

try:
    import wandb
except ImportError:
    wandb = None

import d4rl
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dql.utils import utils
from dql.utils.data_sampler import Data_Sampler
from dql.utils.logger import logger, setup_logger
#from torch.utils.tensorboard import SummaryWriter

#import diffuser.utils as hl_utils
#from diffuser.guides.policies import Policy
from matplotlib import pyplot as plt
import pdb

import ogbench

ANTMAZE_DEFAULT_HYPERPARAMETERS = {
    "lr": 3e-4,
    "eta": 1.0,
    "max_q_backup": False,
    "eval_freq": 1,
    "num_epochs": 200,
    "gn": 7.0,
    "top_k": 1,
    "p": 0.2,
}

# Current DQL defaults are shared across known maze sizes and dataset types, but
# keep the override tables in place so newly added antmaze variants do not
# require another hardcoded env-name list.
ANTMAZE_MAZE_TYPE_OVERRIDES = {
    "medium": {},
    "large": {},
    "giant": {},
    "teleport": {},
}

ANTMAZE_DATASET_TYPE_OVERRIDES = {
    "navigate": {},
    "stitch": {},
    "explore": {},
}


def parse_antmaze_env_name(env_name):
    match = re.fullmatch(r"antmaze-([^-]+)-(.+)-v0", env_name)
    if match is None:
        raise ValueError(
            f"Unsupported DQL env_name '{env_name}'. Expected antmaze-<maze_type>-<dataset_type>-v0."
        )
    return match.group(1), match.group(2)


def resolve_antmaze_hyperparameters(env_name):
    maze_type, dataset_type = parse_antmaze_env_name(env_name)
    hyperparameters = dict(ANTMAZE_DEFAULT_HYPERPARAMETERS)
    hyperparameters.update(ANTMAZE_MAZE_TYPE_OVERRIDES.get(maze_type, {}))
    hyperparameters.update(ANTMAZE_DATASET_TYPE_OVERRIDES.get(dataset_type, {}))
    return hyperparameters


def find_latest_checkpoint_epoch(results_dir):
    pattern = re.compile(r"actor_(\d+)\.pth$")
    latest_epoch = None
    if not os.path.isdir(results_dir):
        return None

    for entry in os.listdir(results_dir):
        match = pattern.fullmatch(entry)
        if match is None:
            continue
        epoch = int(match.group(1))
        if latest_epoch is None or epoch > latest_epoch:
            latest_epoch = epoch
    return latest_epoch


def load_variant_file(results_dir):
    variant_path = os.path.join(results_dir, "variant.json")
    if not os.path.exists(variant_path):
        return {}
    with open(variant_path) as f:
        return json.load(f)


def load_json_file(path, default=None):
    if default is None:
        default = {}
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def save_json_file(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def setup_wandb_run(results_dir, variant, args):
    if wandb is None:
        utils.print_banner(
            "wandb is not installed; continuing without WandB logging.",
            separator="!",
            num_star=90,
        )
        return None

    wandb_mode = os.environ.get("WANDB_MODE", "online")
    if wandb_mode == "disabled":
        utils.print_banner(
            "WANDB_MODE=disabled; continuing without WandB logging.",
            separator="!",
            num_star=90,
        )
        return None

    state_path = os.path.join(results_dir, "wandb_run.json")
    prior_state = load_json_file(state_path, default={})

    init_kwargs = {
        "project": os.environ.get("WANDB_PROJECT", "mctd_eval"),
        "name": os.path.basename(results_dir),
        "config": variant,
        "dir": results_dir,
        "group": f"dql-{args.env_name}",
        "mode": wandb_mode,
    }
    entity = os.environ.get("WANDB_ENTITY")
    if entity:
        init_kwargs["entity"] = entity

    run_id = prior_state.get("id")
    if run_id:
        init_kwargs["id"] = run_id
        init_kwargs["resume"] = "allow"

    run = wandb.init(**init_kwargs)
    if run is None:
        return None

    wandb.define_metric("epoch")
    wandb.define_metric("training_iters")
    wandb.define_metric("train/*", step_metric="epoch")

    save_json_file(
        state_path,
        {
            "id": run.id,
            "name": run.name,
            "project": run.project,
            "entity": run.entity,
            "url": run.url,
            "mode": wandb_mode,
        },
    )
    return run


def train_agent(env, state_dim, action_dim, max_action, device, output_dir, args, wandb_run=None):
    ## load hl_planner
    #class HLParser(hl_utils.Parser):
    #    dataset: str = args.env_name
    #    config: str = "config.antmaze_hl"

    #hlargs = HLParser().parse_args("plan")
    #hl_planner = load_hl_planner(hlargs)

    # Load buffer
    #dataset = env.get_dataset()
    _, dataset, _ = ogbench.make_env_and_datasets(args.env_name, compact_dataset=False)
    data_sampler = Data_Sampler(
        dataset,
        device,
        args.reward_tune,
        #K=hlargs.jump,
        K=args.target_steps,
        p=args.p,
    )
    utils.print_banner("Loaded buffer")

    state_dim = data_sampler.state_dim
    action_dim = data_sampler.action_dim

    from agents.ql_diffusion import Diffusion_QL as Agent

    agent = Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        discount=args.discount,
        tau=args.tau,
        max_q_backup=args.max_q_backup,
        beta_schedule=args.beta_schedule,
        n_timesteps=args.T,
        eta=args.eta,
        lr=args.lr,
        lr_decay=args.lr_decay,
        lr_maxt=args.num_epochs,
        grad_norm=args.gn,
        goal_dim=args.goal_dim,
        lcb_coef=args.lcb_coef,
    )

    start_epoch = 0
    if args.resume_dir:
        resume_epoch = args.resume_epoch
        if resume_epoch is None:
            resume_epoch = find_latest_checkpoint_epoch(args.resume_dir)
        if resume_epoch is None:
            raise FileNotFoundError(
                f"No actor checkpoint found in {args.resume_dir}"
            )

        utils.print_banner(
            f"Resuming from {args.resume_dir} at epoch {resume_epoch}",
            separator="*",
            num_star=90,
        )
        loaded_training_state = agent.load_model(args.resume_dir, resume_epoch)
        if not loaded_training_state:
            utils.print_banner(
                "Checkpoint optimizer/scheduler state not found; resuming with weights only.",
                separator="!",
                num_star=90,
            )
            agent.step = resume_epoch * args.num_steps_per_epoch
        start_epoch = resume_epoch

    early_stop = False
    stop_check = utils.EarlyStopping(tolerance=1, min_delta=0.0)
    writer = None  # SummaryWriter(output_dir)

    evaluations = []
    training_iters = start_epoch * args.num_steps_per_epoch
    max_timesteps = args.num_epochs * args.num_steps_per_epoch
    metric = 100.0
    if training_iters >= max_timesteps:
        utils.print_banner(
            f"Checkpoint already reached target epoch budget ({start_epoch}/{args.num_epochs}).",
            separator="*",
            num_star=90,
        )
        return

    utils.print_banner(f"Training Start", separator="*", num_star=90)
    i = 0
    while (training_iters < max_timesteps) and (not early_stop):
        iterations = int(args.eval_freq * args.num_steps_per_epoch)
        loss_metric = agent.train(
            data_sampler,
            iterations=iterations,
            batch_size=args.batch_size,
            log_writer=writer,
        )
        training_iters += iterations
        curr_epoch = int(training_iters // int(args.num_steps_per_epoch))

        # Logging
        utils.print_banner(f"Train step: {training_iters}", separator="*", num_star=90)
        logger.record_tabular("Trained Epochs", curr_epoch)
        logger.record_tabular("BC Loss", np.mean(loss_metric["bc_loss"]))
        logger.record_tabular("QL Loss", np.mean(loss_metric["ql_loss"]))
        logger.record_tabular("Actor Loss", np.mean(loss_metric["actor_loss"]))
        logger.record_tabular("Critic Loss", np.mean(loss_metric["critic_loss"]))

        if "actor_lr" in loss_metric:
            logger.record_tabular("Actor LR", loss_metric["actor_lr"])
        if "critic_lr" in loss_metric:
            logger.record_tabular("Critic LR", loss_metric["critic_lr"])
        logger.dump_tabular()

        if wandb_run is not None:
            wandb_payload = {
                "epoch": curr_epoch,
                "training_iters": training_iters,
                "train/bc_loss": float(np.mean(loss_metric["bc_loss"])),
                "train/ql_loss": float(np.mean(loss_metric["ql_loss"])),
                "train/actor_loss": float(np.mean(loss_metric["actor_loss"])),
                "train/critic_loss": float(np.mean(loss_metric["critic_loss"])),
                "train/ql_std": float(np.mean(loss_metric["ql_std"])),
                "train/ql_random_std": float(np.mean(loss_metric["ql_random_std"])),
            }
            if "actor_lr" in loss_metric:
                wandb_payload["train/actor_lr"] = float(loss_metric["actor_lr"])
            if "critic_lr" in loss_metric:
                wandb_payload["train/critic_lr"] = float(loss_metric["critic_lr"])
            wandb.log(wandb_payload, step=curr_epoch)

        ## Evaluation
        #with torch.no_grad():
        #    (
        #        eval_res,
        #        eval_res_std,
        #        eval_norm_res,
        #        eval_norm_res_std,
        #    ) = eval_policy(
        #        args,
        #        hl_planner,
        #        agent,
        #        args.env_name,
        #        args.seed,
        #        eval_episodes=100,
        #        output_dir=output_dir,
        #        replan=args.replan,
        #    )
        #evaluations.append(
        #    [
        #        eval_res,
        #        eval_res_std,
        #        eval_norm_res,
        #        eval_norm_res_std,
        #        np.mean(loss_metric["bc_loss"]),
        #        np.mean(loss_metric["ql_loss"]),
        #        np.mean(loss_metric["actor_loss"]),
        #        np.mean(loss_metric["critic_loss"]),
        #        curr_epoch,
        #    ]
        #)
        #np.save(os.path.join(output_dir, "eval"), evaluations)
        #logger.record_tabular("Average Episodic Reward", eval_res)
        #logger.record_tabular("Average Episodic N-Reward", eval_norm_res)
        #logger.dump_tabular()

        bc_loss = np.mean(loss_metric["bc_loss"])
        if args.early_stop:
            early_stop = stop_check(metric, bc_loss)

        metric = bc_loss

        #if args.save_best_model:
        agent.save_model(output_dir, curr_epoch)


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(
    args,
    hl_planner,
    policy,
    env_name,
    seed,
    eval_episodes=10,
    output_dir=None,
    replan=1,
):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)
    replan = replan

    scores = []
    if "ultra" in args.env_name:
        dist_thr = 13
    elif "large" in args.env_name:
        dist_thr = 6
    elif "medium" in args.env_name:
        dist_thr = 4

    for _ in range(eval_episodes):
        traj_return = 0.0
        state, done = eval_env.reset(), False
        target = eval_env.target_goal
        target = np.array(list(target) + [0.0] * 27)

        t = 0

        while not done:
            conditions = {0: state, hl_planner.diffusion_model.horizon - 1: target}
            tgt_dist = np.linalg.norm(target[:2] - state[:2])
            if tgt_dist > dist_thr:
                _, samples = hl_planner(conditions, batch_size=1, verbose=False)
                goal = samples.observations[0, 1]
                dist = np.linalg.norm(goal[:2] - state[:2])
                tgt_goal_dist = np.linalg.norm(target[:2] - goal[:2])
                n_try = 1
                while (n_try <= 5) and (
                    (dist > 7.0) or (dist <= 0.5) or (tgt_goal_dist > tgt_dist)
                ):
                    _, samples = hl_planner(conditions, batch_size=1, verbose=False)
                    goal = samples.observations[0, 1]
                    dist = np.linalg.norm(goal[:2] - state[:2])
                    tgt_goal_dist = np.linalg.norm(target[:2] - goal[:2])
                    n_try += 1
            else:
                goal[:2] = target[:2]
                dist = np.linalg.norm(goal[:2] - state[:2])

            # goal_idx = 1
            # while (dist < 0.5) and (goal_idx < samples.observations.shape[1] - 1):
            #     goal_idx += 1
            #     goal = samples.observations[0, goal_idx]
            #     dist = np.linalg.norm(goal[:2] - state[:2])

            j = 0
            while ((dist > 0.5) or (n_try > 5)) and (j < replan) and (not done):

                if args.goal_dim:
                    goal = goal[: args.goal_dim]
                action = policy.sample_action(state, goal)
                next_state, reward, done, _ = eval_env.step(action)
                state = next_state
                traj_return += reward
                dist = np.linalg.norm(goal[:2] - state[:2])
                j += 1
                t += 1
                pos = np.array2string(state[:2], precision=2, floatmode="fixed")
                subgoal_pos = np.array2string(goal[:2], precision=2, floatmode="fixed")
                print(
                    f"step t-{t}/j-{j}/s-{sum(scores)}:\t pos: {pos}, subgoal_pos: {subgoal_pos}"
                )
        scores.append(traj_return)

    avg_reward = np.mean(scores)
    std_reward = np.std(scores)

    normalized_scores = [eval_env.get_normalized_score(s) for s in scores]
    avg_norm_score = eval_env.get_normalized_score(avg_reward)
    std_norm_score = np.std(normalized_scores)

    utils.print_banner(
        f"Evaluation over {eval_episodes} episodes: {avg_reward:.2f} {avg_norm_score:.2f}"
    )
    return avg_reward, std_reward, avg_norm_score, std_norm_score


def load_hl_planner(args):
    diffusion_experiment = hl_utils.load_diffusion(
        args.logbase,
        args.dataset,
        args.diffusion_loadpath,
        epoch="latest",
    )

    diffusion = diffusion_experiment.ema
    dataset = diffusion_experiment.dataset

    policy = Policy(
        diffusion,
        dataset.normalizer,
    )

    return policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ### Experimental Setups ###
    parser.add_argument("--exp", default="exp", type=str)  # Experiment ID
    parser.add_argument(
        "--device", default=0, type=int
    )  # device, {"cpu", "cuda", "cuda:0", "cuda:1"}, etc
    parser.add_argument(
        #"--env_name", default="antmaze-medium-diverse-v2", type=str
        "--env_name", default="antmaze-medium-navigate-v0", type=str
    )  # OpenAI gym environment name
    parser.add_argument("--dir", default="dql/results", type=str)  # Logging directory
    parser.add_argument(
        "--seed", default=0, type=int
    )  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--num_steps_per_epoch", default=1000, type=int)

    ### Optimization Setups ###
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--lr_decay", action="store_true")
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--save_best_model", action="store_true")

    ### RL Parameters ###
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)

    ### Diffusion Setting ###
    parser.add_argument("--T", default=5, type=int)
    parser.add_argument("--beta_schedule", default="vp", type=str)
    ### Algo Choice ###
    parser.add_argument("--algo", default="ql", type=str)  # ['bc', 'ql']
    parser.add_argument(
        "--ms", default="offline", type=str, help="['online', 'offline']"
    )
    parser.add_argument("--goal_dim", default=2, type=int)
    parser.add_argument("--replan", default=15, type=int)
    parser.add_argument("--p", default=0.2, type=float)
    parser.add_argument("--lcb_coef", default=4.0, type=float)
    parser.add_argument("--target_steps", default=10, type=int)
    parser.add_argument("--reward_tune", default="cql_antmaze", type=str)
    parser.add_argument("--resume_dir", default=None, type=str)
    parser.add_argument("--resume_epoch", default=None, type=int)

    args = parser.parse_args()
    args.lr_decay = True
    resume_variant = {}

    if args.resume_dir is not None:
        args.resume_dir = os.path.normpath(args.resume_dir)
        resume_variant = load_variant_file(args.resume_dir)
        resume_env_name = resume_variant.get("env_name")
        if resume_env_name and args.env_name != resume_env_name:
            print(
                f"[resume] overriding env_name from {args.env_name} to {resume_env_name}"
            )
            args.env_name = resume_env_name

    maze_type, _ = parse_antmaze_env_name(args.env_name)
    resolved_hyperparameters = resolve_antmaze_hyperparameters(args.env_name)
    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    args.output_dir = f"{args.dir}"

    args.num_epochs = resolved_hyperparameters["num_epochs"]
    args.eval_freq = resolved_hyperparameters["eval_freq"]
    args.eval_episodes = 15 if "v2" in args.env_name else 100

    args.lr = resolved_hyperparameters["lr"]
    args.eta = resolved_hyperparameters["eta"]
    args.max_q_backup = resolved_hyperparameters["max_q_backup"]
    args.gn = resolved_hyperparameters["gn"]
    args.top_k = resolved_hyperparameters["top_k"]
    args.p = resolved_hyperparameters["p"]

    if args.resume_dir and resume_variant:
        for key, value in resume_variant.items():
            if key in {
                "device",
                "dir",
                "output_dir",
                "state_dim",
                "action_dim",
                "max_action",
            }:
                continue
            setattr(args, key, value)
        args.output_dir = f"{args.dir}"

    # Setup Logging
    file_name = f"{args.env_name}|{args.exp}|diffusion-{args.algo}|T-{args.T}"
    if args.lr_decay:
        file_name += "|lr_decay"
    file_name += f"|ms-{args.ms}"

    if args.ms == "offline":
        file_name += f"|k-{args.top_k}"
    file_name += f"|{args.seed}"
    file_name += f"|{args.goal_dim}"
    file_name += f"|{args.eta}"
    file_name += f"|{args.max_q_backup}"
    file_name += f"|{args.reward_tune}"
    file_name += f"|{args.p}"
    file_name += f"|{args.lcb_coef}"
    file_name += f"|{args.target_steps}"

    if args.resume_dir:
        results_dir = args.resume_dir
    else:
        results_dir = os.path.join(args.output_dir, file_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    utils.print_banner(f"Saving location: {results_dir}")
    variant = vars(args)

    #env = gym.make(args.env_name)
    env = ogbench.locomaze.maze.make_maze_env('ant', 'maze', maze_type=maze_type)

    #env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    variant.update(state_dim=state_dim)
    variant.update(action_dim=action_dim)
    variant.update(max_action=max_action)
    setup_logger(os.path.basename(results_dir), variant=variant, log_dir=results_dir)
    wandb_run = setup_wandb_run(results_dir, variant, args)
    utils.print_banner(
        f"Env: {args.env_name}, state_dim: {state_dim}, action_dim: {action_dim}"
    )

    try:
        train_agent(
            env,
            state_dim,
            action_dim,
            max_action,
            args.device,
            results_dir,
            args,
            wandb_run=wandb_run,
        )
    finally:
        if wandb_run is not None:
            wandb.finish()
