"""DAPO reward manager adapted to this repository's task reward metadata."""

from collections import defaultdict

import torch

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.dapo import DAPORewardManager


@register("sdpo_dapo")
class SDPOCompatibleDAPORewardManager(DAPORewardManager):
    """DAPO shaping while preserving metadata expected by local reward functions."""

    def __call__(self, data: DataProto, return_dict: bool = False):
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_printed = {}

        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = int(data_item.batch["attention_mask"][:prompt_length].sum())
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = int(data_item.batch["attention_mask"][prompt_length:].sum())
            valid_response_ids = response_ids[:valid_response_length]
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = dict(data_item.non_tensor_batch.get("extra_info", {}))
            extra_info["num_turns"] = data_item.non_tensor_batch.get("__num_turns__")
            extra_info["rollout_reward_scores"] = data_item.non_tensor_batch.get("reward_scores", {})
            extra_info["truncated"] = not (
                valid_response_ids == self.tokenizer.eos_token_id
            ).any().item()

            result = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            if isinstance(result, dict):
                score = float(result["score"])
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = float(result)
                reward_extra_info["acc"].append(score)

            reward = score
            if self.overlong_buffer_cfg.enable:
                expected_len = self.max_resp_len - self.overlong_buffer_cfg.len
                exceed_len = valid_response_length - expected_len
                overlong_reward = min(
                    -exceed_len
                    / self.overlong_buffer_cfg.len
                    * self.overlong_buffer_cfg.penalty_factor,
                    0.0,
                )
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            reward_tensor[i, max(valid_response_length - 1, 0)] = reward
            if already_printed.get(data_source, 0) < self.num_examine:
                already_printed[data_source] = already_printed.get(data_source, 0) + 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        return reward_tensor
