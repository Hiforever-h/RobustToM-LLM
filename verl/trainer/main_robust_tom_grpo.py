"""Dedicated RobustToM v3 GRPO entry point."""

from __future__ import annotations

import random

import hydra
import numpy as np
import ray
import torch
from omegaconf import OmegaConf

from grpo.reward_manager import ProcessRewardManager


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_config(config) -> None:
    if config.algorithm.adv_estimator != "grpo":
        raise ValueError("RobustToM training requires algorithm.adv_estimator=grpo")
    if config.actor_rollout_ref.actor.strategy != "fsdp":
        raise ValueError("The supported RobustToM single-GPU path uses FSDP")
    trajectory_count = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
    if not config.actor_rollout_ref.use_trajectory_batch_sizes:
        raise ValueError("RobustToM requires explicit trajectory batch sizes")
    if config.actor_rollout_ref.actor.ppo_mini_batch_size != trajectory_count:
        raise ValueError(
            "ppo_mini_batch_size must equal train_batch_size * rollout.n "
            f"({trajectory_count})"
        )
    if config.actor_rollout_ref.actor.ppo_mini_batch_size % \
            config.actor_rollout_ref.actor.ppo_micro_batch_size != 0:
        raise ValueError("PPO mini-batch size must be divisible by the micro-batch size")
    if config.data.max_prompt_length != 2048:
        raise ValueError("The audited v3 prompt limit is 2048")
    if config.data.max_response_length != 256:
        raise ValueError("The audited v3 rollout response limit is 256")


@hydra.main(config_path="config", config_name="robust_tom_grpo", version_base=None)
def main(config) -> None:
    _validate_config(config)
    if not ray.is_initialized():
        ray.init(runtime_env={"env_vars": {
            "TOKENIZERS_PARALLELISM": "true",
            "NCCL_DEBUG": "WARN",
        }})
    ray.get(main_task.remote(config))


@ray.remote
def main_task(config) -> None:
    from pprint import pprint

    from omegaconf import open_dict

    from verl.single_controller.ray import RayWorkerGroup
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
    from verl.utils import hf_tokenizer
    from verl.utils.fs import copy_local_path_from_hdfs
    from verl.workers.fsdp_workers import ActorRolloutRefWorker

    OmegaConf.resolve(config)
    pprint(OmegaConf.to_container(config, resolve=True))
    seed = int(config.trainer.seed)
    _set_seed(seed)

    resume_path = config.trainer.get("resume_from_path")
    if resume_path:
        with open_dict(config.actor_rollout_ref.model):
            config.actor_rollout_ref.model.path = resume_path
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    tokenizer = hf_tokenizer(local_path)

    actor_cls = ray.remote(ActorRolloutRefWorker)
    role_worker_mapping = {
        Role.ActorRollout: actor_cls,
        Role.RefPolicy: actor_cls,
    }
    pool_id = "global_pool"
    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec={
            pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        },
        mapping={
            Role.ActorRollout: pool_id,
            Role.RefPolicy: pool_id,
        },
    )
    reward_fn = ProcessRewardManager(tokenizer=tokenizer, num_examine=0)
    val_reward_fn = ProcessRewardManager(tokenizer=tokenizer, num_examine=1)
    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=RayWorkerGroup,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
    )
    trainer.init_workers()
    if resume_path:
        trainer.load_checkpoint(resume_path)
    trainer.fit()


if __name__ == "__main__":
    main()

