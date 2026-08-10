# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DAPO trainer with dynamic group filtering, isolated from the PPO trainer."""

import uuid
from collections import defaultdict
from pprint import pprint

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_advantage, compute_response_mask
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.tracking import Tracking


class RayDAPOTrainer(RayPPOTrainer):
    """PPO dataflow with DAPO's dynamic sampling before every actor update."""

    def _validate_dapo_config(self) -> None:
        filter_cfg = self.config.algorithm.get("filter_groups")
        if filter_cfg is None or not filter_cfg.get("enable", False):
            raise ValueError("Full DAPO requires algorithm.filter_groups.enable=True")
        if not filter_cfg.get("metric"):
            raise ValueError("Full DAPO requires algorithm.filter_groups.metric")
        if self.config.algorithm.adv_estimator != AdvantageEstimator.GRPO:
            raise ValueError("DAPO requires algorithm.adv_estimator=grpo")
        if self.config.algorithm.use_kl_in_reward:
            raise ValueError("The DAPO baseline disables KL in reward")
        if self.config.actor_rollout_ref.actor.use_kl_loss:
            raise ValueError("The DAPO baseline disables actor KL loss")
        if self.config.reward_model.launch_reward_fn_async:
            raise ValueError("The isolated DAPO trainer currently requires synchronous reward scoring")
        if self.use_critic or self.use_reference_policy:
            raise ValueError("DAPO uses neither a critic nor a reference policy")

    def _filter_groups(self, batch: DataProto) -> tuple[DataProto, dict[str, float]]:
        metric_name = self.config.algorithm.filter_groups.metric
        if metric_name == "seq_final_reward":
            batch.non_tensor_batch[metric_name] = batch.batch["token_level_rewards"].sum(dim=-1).cpu().numpy()
        elif metric_name == "seq_reward":
            batch.non_tensor_batch[metric_name] = batch.batch["token_level_scores"].sum(dim=-1).cpu().numpy()

        if metric_name not in batch.non_tensor_batch:
            available = sorted(batch.non_tensor_batch.keys())
            raise KeyError(f"DAPO filter metric {metric_name!r} is unavailable; available={available}")

        prompt_uid2values = defaultdict(list)
        for uid, value in zip(
            batch.non_tensor_batch["uid"],
            batch.non_tensor_batch[metric_name],
            strict=True,
        ):
            prompt_uid2values[uid].append(float(value))

        kept_uids = {
            uid
            for uid, values in prompt_uid2values.items()
            if len(values) == 1 or float(np.std(values)) > 0.0
        }
        kept_indices = [
            idx for idx, uid in enumerate(batch.non_tensor_batch["uid"]) if uid in kept_uids
        ]
        generated_prompts = len(prompt_uid2values)
        kept_prompts = len(kept_uids)
        stats = {
            "generated_prompts": float(generated_prompts),
            "kept_prompts": float(kept_prompts),
            "dropped_prompts": float(generated_prompts - kept_prompts),
        }
        return batch[kept_indices], stats

    @staticmethod
    def _reward_info_from_batch(batch: DataProto, keys: set[str]) -> dict[str, list]:
        output = {}
        for key in keys:
            if key in batch.non_tensor_batch:
                value = batch.non_tensor_batch[key]
                output[key] = value.tolist() if hasattr(value, "tolist") else list(value)
        return output

    def fit(self):
        self._validate_dapo_config()
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
            group_name=self.config.trainer.get("group_name", None),
        )

        self.global_steps = 0
        self._load_checkpoint()
        current_epoch = self.global_steps // len(self.train_dataloader)

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            RolloutSkip(self.config, self.actor_rollout_wg).wrap_generate_sequences()

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="DAPO Training")
        self.global_steps += 1
        self.max_steps_duration = 0
        last_val_metrics = None

        pending_batch = None
        pending_prompt_count = 0
        num_gen_batches = 0
        generated_prompts = 0.0
        kept_prompts = 0.0
        reward_extra_keys: set[str] = set()
        metrics = {}
        timing_raw: dict[str, float] = {}

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)

                if num_gen_batches == 0:
                    metrics = {}
                    timing_raw = {}
                    with marked_timer("start_profile", timing_raw):
                        self._start_profiling(
                            not prev_step_profile and curr_step_profile
                            if self.config.global_profiler.profile_continuous_steps
                            else curr_step_profile
                        )

                with marked_timer("step", timing_raw):
                    new_batch = DataProto.from_single_dict(batch_dict)
                    new_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )

                    gen_batch = self._get_gen_batch(new_batch)
                    gen_batch.meta_info["global_steps"] = self.global_steps
                    gen_batch_output = gen_batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n,
                        interleave=True,
                    )

                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                        for key, value in gen_batch_output.meta_info.get("timing", {}).items():
                            timing_raw[key] = timing_raw.get(key, 0.0) + value
                        gen_batch_output.meta_info.pop("timing", None)

                    new_batch = new_batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n,
                        interleave=True,
                    )
                    new_batch = new_batch.union(gen_batch_output)
                    new_batch.batch["response_mask"] = compute_response_mask(new_batch)

                    with marked_timer("reward", timing_raw, color="yellow"):
                        if self.use_rm and "rm_scores" not in new_batch.batch:
                            if not self.use_reward_loop:
                                rm_scores = self.rm_wg.compute_rm_score(new_batch)
                            else:
                                rm_scores = self.reward_loop_manager.compute_rm_score(new_batch)
                            new_batch = new_batch.union(rm_scores)
                        reward_tensor, reward_extra_infos = self._compute_or_extract_reward(
                            new_batch,
                            reward_fn=self.reward_fn,
                            return_dict=False,
                        )

                    new_batch.batch["token_level_scores"] = reward_tensor
                    new_batch.batch["token_level_rewards"] = reward_tensor
                    if reward_extra_infos:
                        reward_extra_keys.update(reward_extra_infos.keys())
                        new_batch.non_tensor_batch.update(
                            {key: np.asarray(value) for key, value in reward_extra_infos.items()}
                        )

                    filtered_batch, filter_stats = self._filter_groups(new_batch)
                    num_gen_batches += 1
                    generated_prompts += filter_stats["generated_prompts"]
                    kept_prompts += filter_stats["kept_prompts"]
                    pending_prompt_count += int(filter_stats["kept_prompts"])
                    if len(filtered_batch) > 0:
                        pending_batch = (
                            filtered_batch
                            if pending_batch is None
                            else DataProto.concat([pending_batch, filtered_batch])
                        )

                    required_prompts = int(self.config.data.train_batch_size)
                    if pending_prompt_count < required_prompts:
                        max_batches = int(self.config.algorithm.filter_groups.max_num_gen_batches)
                        print(
                            f"[dapo] qualified prompts {pending_prompt_count}/{required_prompts}; "
                            f"generation batches={num_gen_batches}/{max_batches if max_batches > 0 else 'unlimited'}"
                        )
                        if max_batches > 0 and num_gen_batches >= max_batches:
                            raise RuntimeError(
                                "DAPO reached algorithm.filter_groups.max_num_gen_batches before "
                                "collecting enough non-zero-variance prompt groups"
                            )
                        continue

                    trajectories = required_prompts * int(self.config.actor_rollout_ref.rollout.n)
                    batch = pending_batch[:trajectories]
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)
                    reward_extra_infos_dict = self._reward_info_from_batch(batch, reward_extra_keys)
                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1
                    ).tolist()

                    images_seqlens_all = []
                    for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                        if "image_grid_thw" in multi_modal_input:
                            images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens_all

                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_log_probs = bool(
                        rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    )
                    if bypass_log_probs:
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy = agg_loss(
                                loss_mat=entropys,
                                loss_mask=batch.batch["response_mask"],
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            metrics.update(
                                {
                                    "actor/entropy": entropy.detach().item(),
                                    "perf/mfu/actor_infer": old_log_prob_mfu,
                                }
                            )
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)

                    assert "old_log_probs" in batch.batch
                    if (
                        rollout_corr_config is not None
                        and "rollout_log_probs" in batch.batch
                        and not bypass_log_probs
                    ):
                        from verl.trainer.ppo.rollout_corr_helper import (
                            compute_rollout_correction_and_add_to_batch,
                        )

                        batch, correction_metrics = compute_rollout_correction_and_add_to_batch(
                            batch, rollout_corr_config
                        )
                        metrics.update(correction_metrics)

                    with marked_timer("adv", timing_raw, color="brown"):
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=self.config.algorithm.get(
                                "norm_adv_by_std_in_grpo", True
                            ),
                            config=self.config.algorithm,
                        )

                    with marked_timer("update_actor", timing_raw, color="red"):
                        actor_output = self._update_actor(batch)
                    metrics.update(reduce_metrics(actor_output.meta_info["metrics"]))

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(
                            batch, reward_extra_infos_dict, timing_raw, rollout_data_dir
                        )

                is_last_step = self.global_steps >= self.total_training_steps
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                esi_close = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                if self.config.trainer.save_freq > 0 and (
                    is_last_step
                    or self.global_steps % self.config.trainer.save_freq == 0
                    or esi_close
                ):
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                next_step_profile = (
                    self.global_steps + 1 in self.config.global_profiler.steps
                    if self.config.global_profiler.steps is not None
                    else False
                )
                with marked_timer("stop_profile", timing_raw):
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                prev_step_profile = curr_step_profile
                curr_step_profile = next_step_profile

                self.max_steps_duration = max(self.max_steps_duration, timing_raw["step"])
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                        "train/num_gen_batches": num_gen_batches,
                        "dapo/generated_prompts": generated_prompts,
                        "dapo/qualified_prompts": kept_prompts,
                        "dapo/filter_keep_ratio": kept_prompts / max(generated_prompts, 1.0),
                    }
                )
                metrics.update(compute_data_metrics(batch=batch, use_critic=False))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                metrics.update(
                    compute_throughout_metrics(
                        batch=batch,
                        timing_raw=timing_raw,
                        n_gpus=self.resource_pool_manager.get_n_gpus(),
                    )
                )
                metrics.update(
                    compute_variance_proxy_metrics(
                        batch=batch,
                        gradient_norm=metrics.get("actor/grad_norm"),
                    )
                )

                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)
                logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)

                pending_batch = None
                pending_prompt_count = 0
                num_gen_batches = 0
                generated_prompts = 0.0
                kept_prompts = 0.0
                reward_extra_keys = set()
                self.global_steps += 1

                if hasattr(self.train_dataset, "on_batch_end"):
                    self.train_dataset.on_batch_end(batch=batch)

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

        progress_bar.close()
        raise RuntimeError(
            "DAPO exhausted trainer.total_epochs before reaching trainer.total_training_steps; "
            "increase trainer.total_epochs"
        )
