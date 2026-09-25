# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
Single Process Actor
"""

import contextlib
import hashlib
import logging
import os
import time
from copy import deepcopy
from types import SimpleNamespace
from typing import Optional

import torch
from torch import nn
from PIL import Image
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto

def _generation_position_ids(model, position_ids: torch.Tensor) -> torch.Tensor:
    if position_ids.dim() != 3 or position_ids.shape[1] != 3:
        return position_ids
    model_config = getattr(model, "config", None)
    if model_config is None:
        model_config = getattr(getattr(model, "module", None), "config", None)
    if getattr(model_config, "model_type", None) in {"qwen3_5", "qwen3_5_moe"}:
        return position_ids.transpose(0, 1).contiguous()
    return position_ids
from verl.trainer.ppo.core_algos import (
    agg_loss,
    build_batch_mean_jsd_gate,
    build_batch_mean_multi_negative_probability_gate,
    build_per_response_mean_jsd_gate,
    compute_log_jsd,
    compute_topk_tail_jsd,
    compute_self_distillation_loss,
    compute_teacher_trajectory_sft_loss,
    get_policy_loss_fn,
    kl_penalty,
)
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name, get_torch_device
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.metric import AggregationType, Metric, reduce_metrics
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.vision_opd import add_red_frame
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, slice_input_tensor, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class TrustRegionTeacher(nn.Module):
    def __init__(self, ref_module: nn.Module, student_module: nn.Module, mix_coef: float) -> None:
        super().__init__()
        self.ref_module = ref_module
        self.student_module = student_module
        self.mix_coef = float(mix_coef)

    def forward(self, *args, **kwargs):
        ref_out = self.ref_module(*args, **kwargs)
        student_out = self.student_module(*args, **kwargs)
        ref_logits = ref_out.logits if hasattr(ref_out, "logits") else ref_out[0]
        student_logits = student_out.logits if hasattr(student_out, "logits") else student_out[0]
        logits = torch.lerp(ref_logits, student_logits, self.mix_coef)
        return SimpleNamespace(logits=logits)


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.teacher_module: Optional[nn.Module] = None
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.use_dynamic_bsz = self.config.get("use_dynamic_bsz", False)

        self.use_prefix_grouper = self.config.get("use_prefix_grouper", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_prefix_grouper={self.use_prefix_grouper}")

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

        # Sum of squared probabilities computation (for optimal_token_baseline)
        # Only initialize if calculate_sum_pi_squared config is enabled
        if self.config.get("calculate_sum_pi_squared", False):
            self.calculate_sum_pi_squared_from_logits = (
                torch.compile(verl_F.calculate_sum_pi_squared_from_logits, dynamic=True)
                if self.config.get("use_torch_compile", True)
                else verl_F.calculate_sum_pi_squared_from_logits
            )
            assert not (self.use_fused_kernels or self.use_prefix_grouper), (
                "calculate_sum_pi_squared is not supported with "
                f"{self.use_fused_kernels=} or {self.use_prefix_grouper=} for now."
            )

    def _update_teacher(self) -> None:
        self_distillation_cfg = getattr(self.config, "self_distillation", None)
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
        if not self_distillation_cfg or loss_mode != "vopd":
            return

        teacher_model_source = getattr(self_distillation_cfg, "teacher_model_source", "legacy")
        if teacher_model_source != "legacy":
            return
        teacher_regularization = getattr(self_distillation_cfg, "teacher_regularization", "ema")
        if self.teacher_module is None or self.teacher_module is self.actor_module:
            raise ValueError("Teacher updates require a separate teacher_module in the actor worker.")
        with torch.no_grad():
            if teacher_regularization == "ema":
                update_rate = getattr(self_distillation_cfg, "teacher_update_rate", 0.0)
                if update_rate == 0.0:
                    return
                for teacher_param, student_param in zip(
                    self.teacher_module.parameters(),
                    self.actor_module.parameters(),
                ):
                    student_data = student_param.data.to(device=teacher_param.device)
                    teacher_param.data.mul_(1.0 - update_rate).add_(student_data, alpha=update_rate)
                return

            if teacher_regularization == "progressive":
                teacher_update_interval = getattr(self_distillation_cfg, "teacher_update_interval", None)
                if teacher_update_interval is None:
                    raise ValueError("Progressive teacher requires self_distillation.teacher_update_interval.")
                global_steps = getattr(self, "_current_global_steps", None)
                if global_steps is None or global_steps % teacher_update_interval != 0:
                    return
                for teacher_param, student_param in zip(
                    self.teacher_module.parameters(),
                    self.actor_module.parameters(),
                ):
                    teacher_param.data.copy_(student_param.data.to(device=teacher_param.device))
                for teacher_buffer, student_buffer in zip(
                    self.teacher_module.buffers(),
                    self.actor_module.buffers(),
                ):
                    teacher_buffer.data.copy_(student_buffer.data.to(device=teacher_buffer.device))
                return

            return

    @torch.no_grad()
    def generate_crop_teacher_trajectories(self, prompts: DataProto) -> DataProto:
        """Sample complete v9-13/v9-14 responses from the resident EMA crop teacher."""
        cfg = self.config.self_distillation
        target_mode = cfg.get("distillation_target_mode")
        if target_mode not in {"crop_only_v9_13", "crop_only_v9_14"}:
            raise ValueError("crop-teacher generation is isolated to crop_only_v9_13/v9_14")
        if self.teacher_module is None or self.teacher_module is self.actor_module:
            raise ValueError(f"{target_mode} requires a separate resident EMA teacher")

        from tensordict import TensorDict
        from transformers import GenerationConfig
        from verl.utils.model import extract_multi_modal_inputs
        from verl.utils.torch_functional import get_response_mask

        prompts = prompts.to(get_device_id())
        do_sample = bool(prompts.meta_info["do_sample"])
        temperature = float(prompts.meta_info["temperature"])
        top_k = int(prompts.meta_info["top_k"])
        top_p = float(prompts.meta_info["top_p"])
        response_length = int(prompts.meta_info["response_length"])
        if not do_sample or temperature != 1.0 or top_k != -1 or top_p != 1.0:
            raise ValueError(
                f"{target_mode} must reuse student sampling: do_sample=True, "
                "temperature=1.0, top_k=-1, top_p=1.0"
            )

        input_ids = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        eos_token_ids = self._termination_token_ids(self.teacher_module)
        if not eos_token_ids:
            eos = prompts.meta_info["eos_token_id"]
            eos_token_ids = [int(eos)] if isinstance(eos, int) else [int(token_id) for token_id in eos]
        pad_token_id = int(prompts.meta_info["pad_token_id"])
        generation_config = GenerationConfig(
            do_sample=True, num_beams=1, temperature=temperature, top_k=0, top_p=top_p, num_return_sequences=1
        )
        multi_modal_inputs = extract_multi_modal_inputs(
            prompts.non_tensor_batch.get("multi_modal_inputs", [])
        )
        # The generation path bypasses the Qwen3.5 PPO forward wrapper that normally
        # casts image tensors to the vision tower dtype.  Crop-teacher prompts are
        # freshly assembled by the trainer, so normalize their floating-point
        # multimodal tensors here before calling HF ``generate``.
        teacher_model = self.teacher_module
        vision = getattr(teacher_model, "visual", None)
        if vision is None:
            teacher_model_body = getattr(teacher_model, "model", None)
            vision = getattr(teacher_model_body, "visual", None)
        if vision is None:
            raise ValueError(f"{target_mode} requires a Qwen-compatible teacher vision module")
        # Match the dtype/device of the actual patch projection used by
        # Qwen3.5. The first parameter of a wrapped/FSDP vision tower can be
        # a positional embedding or another auxiliary tensor with a different
        # dtype, which would reintroduce the Conv3D bias/input mismatch seen in
        # the failed v9-13 run.
        patch_embed = getattr(vision, "patch_embed", None)
        projection = getattr(patch_embed, "proj", None) if patch_embed is not None else None
        projection_weight = getattr(projection, "weight", None) if projection is not None else None
        if projection_weight is not None:
            vision_dtype = projection_weight.dtype
            vision_device = projection_weight.device
        else:
            vision_param = next(vision.parameters())
            vision_dtype = vision_param.dtype
            vision_device = vision_param.device

        def _cast_multimodal(value, key=None):
            if torch.is_tensor(value):
                if key in {"pixel_values", "pixel_values_videos"} and value.is_floating_point():
                    return value.to(device=vision_device, dtype=vision_dtype)
                return value.to(device=vision_device)
            if isinstance(value, dict):
                return {item_key: _cast_multimodal(item, item_key) for item_key, item in value.items()}
            if isinstance(value, list):
                return [_cast_multimodal(item, key) for item in value]
            if isinstance(value, tuple):
                return tuple(_cast_multimodal(item, key) for item in value)
            return value

        multi_modal_inputs = _cast_multimodal(multi_modal_inputs)
        was_training = self.teacher_module.training
        self.teacher_module.eval()
        device_api = get_torch_device()
        previous_rng_state = device_api.get_rng_state()
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        teacher_seed = 20260913 + int(prompts.meta_info.get("global_steps", 0)) * world_size + rank
        device_api.manual_seed(teacher_seed)
        param_ctx = contextlib.nullcontext()
        if isinstance(self.teacher_module, FSDP):
            param_ctx = FSDP.summon_full_params(self.teacher_module, writeback=False, recurse=False)
        try:
            with param_ctx, torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
                # DataProto tensors are batch-first, but Qwen3.5's M-RoPE
                # implementation expects (3, batch, seq). Convert only at
                # the HF generation boundary; keeping batch-first storage is
                # required for repeat/padding/dispatch semantics.
                generation_position_ids = _generation_position_ids(self.teacher_module, position_ids)
                output = self.teacher_module.generate(
                    input_ids=input_ids, attention_mask=attention_mask, position_ids=generation_position_ids,
                    **multi_modal_inputs, max_new_tokens=response_length, eos_token_id=eos_token_ids,
                    pad_token_id=pad_token_id, generation_config=generation_config, output_scores=False,
                    return_dict_in_generate=True, use_cache=True,
                )
        finally:
            device_api.set_rng_state(previous_rng_state)
            self.teacher_module.train(was_training)
        sequence = output.sequences
        target_length = input_ids.shape[1] + response_length
        if sequence.shape[1] < target_length:
            sequence = torch.cat((sequence, torch.full(
                (sequence.shape[0], target_length - sequence.shape[1]), pad_token_id,
                dtype=sequence.dtype, device=sequence.device,
            )), dim=1)
        if sequence.shape[1] != target_length:
            raise ValueError("crop teacher generated an unexpected sequence length")
        responses = sequence[:, input_ids.shape[1]:]
        response_mask = get_response_mask(responses, eos_token=eos_token_ids, dtype=attention_mask.dtype)
        return DataProto(
            batch=TensorDict({"responses": responses, "response_mask": response_mask}, batch_size=responses.shape[0]),
            non_tensor_batch=prompts.non_tensor_batch, meta_info={},
        )

    @staticmethod
    def _has_non_empty_multi_modal_inputs(multi_modal_inputs) -> bool:
        if multi_modal_inputs is None:
            return False
        for inputs in multi_modal_inputs:
            if inputs is None:
                continue
            inputs = getattr(inputs, "data", inputs)
            if isinstance(inputs, dict):
                if not inputs:
                    continue
                for value in inputs.values():
                    if value is None:
                        continue
                    if isinstance(value, torch.Tensor) and value.numel() == 0:
                        continue
                    return True
            else:
                return True
        return False

    @staticmethod
    def _termination_token_ids(module: nn.Module) -> list[int]:
        model = getattr(module, "module", module)
        model_config = getattr(model, "config", None)
        configs = [
            model_config,
            getattr(model_config, "text_config", None),
            getattr(model, "generation_config", None),
        ]
        token_ids = set()
        for config in configs:
            eos_token_id = getattr(config, "eos_token_id", None)
            if eos_token_id is None:
                continue
            if isinstance(eos_token_id, int):
                token_ids.add(eos_token_id)
            else:
                token_ids.update(int(token_id) for token_id in eos_token_id)
        return sorted(token_ids)

    @staticmethod
    def _add_tail_bucket(log_probs: torch.Tensor) -> torch.Tensor:
        log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
        log_s = torch.clamp(log_s, max=-1e-7)
        tail_log = torch.log(-torch.expm1(log_s))
        return torch.cat([log_probs, tail_log], dim=-1)

    @staticmethod
    def _build_response_positions(
        response_start_idx: torch.Tensor,
        response_length: int,
        seqlen: int,
    ) -> torch.Tensor:
        if response_start_idx.dim() != 1:
            raise ValueError(f"response_start_idx must be rank-1, got shape {tuple(response_start_idx.shape)}")
        if (response_start_idx < 1).any():
            raise ValueError("response_start_idx must be >= 1 so response logits have a preceding context token.")

        offsets = torch.arange(response_length, device=response_start_idx.device, dtype=response_start_idx.dtype)
        response_positions = response_start_idx.unsqueeze(1) - 1 + offsets.unsqueeze(0)
        if response_positions.numel() > 0:
            if response_positions.min().item() < 0 or response_positions.max().item() >= seqlen:
                raise ValueError(
                    f"Response positions out of bounds for seqlen={seqlen}: "
                    f"min={response_positions.min().item()}, max={response_positions.max().item()}"
                )
        return response_positions.to(dtype=torch.long)

    @staticmethod
    def _select_response_positions(
        hidden_states: torch.Tensor,
        response_length: int,
        response_start_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if response_start_idx is None:
            return hidden_states[:, -response_length - 1 : -1, ...]

        response_positions = DataParallelPPOActor._build_response_positions(
            response_start_idx=response_start_idx,
            response_length=response_length,
            seqlen=hidden_states.size(1),
        )
        if hidden_states.dim() == 2:
            return torch.gather(hidden_states, dim=1, index=response_positions)

        gather_index = response_positions.view(
            response_positions.size(0),
            response_positions.size(1),
            *([1] * (hidden_states.dim() - 2)),
        ).expand(response_positions.size(0), response_positions.size(1), *hidden_states.shape[2:])
        return torch.gather(hidden_states, dim=1, index=gather_index)

    @staticmethod
    def _select_response_positions_from_unpadded(
        hidden_states: torch.Tensor,
        indices: torch.Tensor,
        batch_size: int,
        seqlen: int,
        response_length: int,
        response_start_idx: Optional[torch.Tensor] = None,
        selection_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Select response rows without materializing a padded ``[B, S, ...]`` tensor."""
        if hidden_states.dim() < 1:
            raise ValueError("hidden_states must have at least one dimension")
        if selection_mask is not None and selection_mask.shape != (batch_size, response_length):
            raise ValueError("selection_mask must have shape [batch_size, response_length]")
        if indices.dim() != 1:
            raise ValueError(f"indices must be rank-1, got shape {tuple(indices.shape)}")
        if indices.numel() == 0:
            return (
                hidden_states.new_zeros((int(selection_mask.sum().item()), *hidden_states.shape[1:]))
                if selection_mask is not None
                else hidden_states.new_zeros((batch_size, response_length, *hidden_states.shape[1:]))
            )
        if hidden_states.size(0) != indices.numel():
            raise ValueError(
                "The unpadded tensor and indices disagree: "
                f"hidden_states has {hidden_states.size(0)} rows, indices has {indices.numel()}"
            )

        if response_start_idx is None:
            response_start_idx = torch.full(
                (batch_size,),
                seqlen - response_length,
                device=indices.device,
                dtype=torch.long,
            )
        else:
            response_start_idx = response_start_idx.to(device=indices.device, dtype=torch.long)
        response_positions = DataParallelPPOActor._build_response_positions(
            response_start_idx=response_start_idx,
            response_length=response_length,
            seqlen=seqlen,
        )

        batch_offsets = torch.arange(batch_size, device=indices.device, dtype=torch.long).unsqueeze(1) * seqlen
        padded_positions = (batch_offsets + response_positions).reshape(-1)
        if selection_mask is not None:
            padded_positions = padded_positions[selection_mask.to(device=indices.device, dtype=torch.bool).reshape(-1)]

        # ``unpad_input`` returns row-major flattened indices, so searchsorted maps
        # each padded response position to its row in the remove-padding tensor.
        unpadded_rows = torch.searchsorted(indices, padded_positions)
        valid = unpadded_rows < indices.numel()
        safe_rows = unpadded_rows.clamp(max=indices.numel() - 1)
        valid &= indices.index_select(0, safe_rows) == padded_positions
        selected = hidden_states.index_select(0, safe_rows)
        selected[~valid] = 0
        if selection_mask is not None:
            return selected
        return selected.reshape(batch_size, response_length, *hidden_states.shape[1:])

    def _dump_self_distillation_log_probs(
        self,
        *,
        meta_info: dict,
        self_distillation_cfg,
        dump_chunks: list[dict[str, torch.Tensor]],
    ) -> None:
        dump_root = self_distillation_cfg.get("log_prob_dump_dir", None)
        if not dump_root or not dump_chunks:
            return

        global_step = meta_info.get("global_steps")
        if global_step is None:
            return

        experiment_name = os.environ.get("EXPERIMENT", "unknown_experiment")
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        normalized_root = os.path.normpath(dump_root)
        if os.path.basename(normalized_root) == experiment_name:
            save_dir = normalized_root
        else:
            save_dir = os.path.join(normalized_root, experiment_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{int(global_step)}.rank{rank}.pt")

        student_log_probs = torch.cat([chunk["student_log_probs"] for chunk in dump_chunks], dim=0)
        teacher_log_probs = torch.cat([chunk["teacher_log_probs"] for chunk in dump_chunks], dim=0)
        reference_teacher_log_probs = None
        if all("reference_teacher_log_probs" in chunk for chunk in dump_chunks):
            reference_teacher_log_probs = torch.cat(
                [chunk["reference_teacher_log_probs"] for chunk in dump_chunks],
                dim=0,
            )

        payload = {
            "student_log_probs": student_log_probs,
            "teacher_log_probs": teacher_log_probs,
            "global_step": int(global_step),
            "rank": rank,
            "experiment_name": experiment_name,
            "distribution_size": int(student_log_probs.shape[-1]),
            "num_valid_tokens": int(student_log_probs.shape[0]),
            "distillation_topk": self_distillation_cfg.get("distillation_topk", None),
            "distillation_add_tail": bool(self_distillation_cfg.get("distillation_add_tail", False)),
            "distillation_target_mode": self_distillation_cfg.get("distillation_target_mode", "standard"),
        }
        if reference_teacher_log_probs is not None:
            payload["reference_teacher_log_probs"] = reference_teacher_log_probs
        torch.save(payload, save_path)

    def _forward_micro_batch(
        self,
        micro_batch: dict[str, torch.Tensor],
        temperature: float,
        distillation_temperature: Optional[float] = None,
        calculate_entropy: bool = False,
        return_all_logps: bool = False,
        all_logps_mask: Optional[torch.Tensor] = None,
        distill_topk: Optional[int] = None,
        allow_all_logps_with_topk: bool = False,
        topk_indices: Optional[torch.Tensor] = None,
        module: Optional[nn.Module] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            dict[str, torch.Tensor]:
                log_probs: (bs, response_len)
                if calculate_entropy is True:
                    entropys: (bs, response_len)
                if calculate_sum_pi_squared is False:
                    sum_pi_squared: (bs, response_len)
                if distill_topk or topk_indices is set:
                    topk_logps: (bs, response_len, k)
                    topk_indices: (bs, response_len, k)
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)
        sum_pi_squared_checkpointing = self.config.get("sum_pi_squared_checkpointing", False)
        distillation_temperature = distillation_temperature or temperature
        if distillation_temperature <= 0:
            raise ValueError(f"distillation_temperature must be positive, got {distillation_temperature}")
        use_topk = distill_topk is not None or topk_indices is not None
        compute_all_logps = return_all_logps and (not use_topk or allow_all_logps_with_topk)
        if all_logps_mask is not None:
            if not compute_all_logps:
                raise ValueError("all_logps_mask requires return_all_logps=True without top-k")
            if all_logps_mask.shape != micro_batch["responses"].shape:
                raise ValueError("all_logps_mask must match responses shape")
        return_topk_indices = use_topk and topk_indices is None
        if (return_all_logps or use_topk) and self.use_fused_kernels:
            raise ValueError("Logit distillation requires disabling fused kernels.")

        model = module or self.actor_module

        # PrefixGrouper path for shared-prefix optimization
        if self.use_prefix_grouper:
            can_use_pg = (
                not self.use_remove_padding
                and not self.use_ulysses_sp
                and not self.use_fused_kernels
                and not self.use_dynamic_bsz
                and not return_all_logps
                and not use_topk
            )
            if can_use_pg and "response_mask" in micro_batch and "uid" in micro_batch:
                from verl.trainer.ppo.prefix_grouper_utils import forward_micro_batch_with_prefix_grouper

                return forward_micro_batch_with_prefix_grouper(
                    micro_batch=micro_batch,
                    model=model,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    device_name=self.device_name,
                    param_dtype=self.param_dtype,
                    use_chunking_entropy=self.config.get("entropy_from_logits_with_chunking", False),
                )

        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            response_start_idx = micro_batch.get("response_start_idx")
            if response_start_idx is not None:
                response_start_idx = response_start_idx.to(device=input_ids.device, dtype=torch.long)
                if response_start_idx.shape != (batch_size,):
                    raise ValueError(
                        f"response_start_idx shape must be ({batch_size},), got {tuple(response_start_idx.shape)}"
                    )
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                is_mask_all_zero = attention_mask.sum() == 0
                if is_mask_all_zero:
                    input_ids_rmpad = torch.zeros(
                        (1, self.ulysses_sequence_parallel_size),
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    if position_ids.dim() == 3:
                        position_ids_rmpad = torch.zeros(
                            (position_ids.shape[0], 1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    else:
                        position_ids_rmpad = torch.zeros(
                            (1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(model, "module", model).config,
                        "vision_config",
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = model(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)
                    distillation_logits_rmpad = (
                        logits_rmpad
                        if distillation_temperature == temperature
                        else logits_rmpad * (temperature / distillation_temperature)
                    )

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        # ((total_nnz / sp) + pad)
                        entropy_rmpad = (
                            self.compute_entropy_from_logits(logits_rmpad)
                            if not self.config.entropy_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)
                        )

                    if use_topk:
                        if topk_indices is None:
                            topk = min(distill_topk, distillation_logits_rmpad.shape[-1])
                            topk_logits_rmpad, topk_indices_rmpad = torch.topk(
                                distillation_logits_rmpad, topk, dim=-1
                            )
                        else:
                            topk = topk_indices.size(-1)
                            full_topk_indices = torch.zeros(
                                batch_size,
                                seqlen,
                                topk,
                                device=topk_indices.device,
                                dtype=topk_indices.dtype,
                            )
                            if response_start_idx is None:
                                full_topk_indices[:, -response_length - 1 : -1, :] = topk_indices
                            else:
                                response_positions = self._build_response_positions(
                                    response_start_idx=response_start_idx.to(device=topk_indices.device),
                                    response_length=response_length,
                                    seqlen=seqlen,
                                )
                                batch_indices = torch.arange(batch_size, device=topk_indices.device).unsqueeze(1)
                                full_topk_indices[batch_indices, response_positions, :] = topk_indices
                            topk_indices_rmpad = index_first_axis(
                                rearrange(full_topk_indices, "b s k -> (b s) k"), indices
                            )
                            if self.use_ulysses_sp:
                                topk_indices_rmpad = slice_input_tensor(
                                    topk_indices_rmpad.unsqueeze(0), dim=1, padding=True
                                ).squeeze(0)
                            topk_logits_rmpad = torch.gather(
                                distillation_logits_rmpad, dim=-1, index=topk_indices_rmpad
                            )
                        logsumexp_rmpad = torch.logsumexp(distillation_logits_rmpad, dim=-1, keepdim=True)
                        topk_logps_rmpad = topk_logits_rmpad - logsumexp_rmpad

                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = (
                            self.calculate_sum_pi_squared_from_logits(logits_rmpad)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(
                                self.calculate_sum_pi_squared_from_logits, logits_rmpad
                            )
                        )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if use_topk:
                        topk_logps_rmpad = gather_outputs_and_unpad(
                            topk_logps_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                        if return_topk_indices:
                            topk_indices_rmpad = gather_outputs_and_unpad(
                                topk_indices_rmpad,
                                gather_dim=0,
                                unpad_dim=0,
                                padding_size=pad_size,
                            )
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = gather_outputs_and_unpad(
                            sum_pi_squared_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )

                if compute_all_logps:
                    # Full-vocabulary distillation only needs response rows. Computing
                    # prompt rows and padding them back to [B, S, V] creates a second
                    # multi-gigabyte tensor and is unnecessary.
                    if self.use_ulysses_sp:
                        distillation_logits_rmpad = gather_outputs_and_unpad(
                            distillation_logits_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    response_distillation_logits = self._select_response_positions_from_unpadded(
                        hidden_states=distillation_logits_rmpad,
                        indices=indices,
                        batch_size=batch_size,
                        seqlen=seqlen,
                        response_length=response_length,
                        response_start_idx=response_start_idx,
                        selection_mask=all_logps_mask,
                    )
                    all_logps = torch.log_softmax(response_distillation_logits, dim=-1)

                if is_mask_all_zero:
                    log_probs = log_probs[:0]
                    if calculate_entropy:
                        entropy_rmpad = entropy_rmpad[:0]
                    if use_topk:
                        topk_logps_rmpad = topk_logps_rmpad[:0]
                        if return_topk_indices:
                            topk_indices_rmpad = topk_indices_rmpad[:0]

                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if calculate_sum_pi_squared:
                    full_sum_pi_squared = pad_input(
                        hidden_states=sum_pi_squared_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if use_topk:
                    full_topk_logps = pad_input(
                        hidden_states=topk_logps_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    if return_topk_indices:
                        full_topk_indices = pad_input(
                            hidden_states=topk_indices_rmpad,
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = self._select_response_positions(
                        full_entropy.squeeze(-1),
                        response_length=response_length,
                        response_start_idx=response_start_idx,
                    )
                if calculate_sum_pi_squared:
                    # (bsz, response_length)
                    sum_pi_squared = self._select_response_positions(
                        full_sum_pi_squared.squeeze(-1),
                        response_length=response_length,
                        response_start_idx=response_start_idx,
                    )
                log_probs = self._select_response_positions(
                    full_log_probs.squeeze(-1),
                    response_length=response_length,
                    response_start_idx=response_start_idx,
                )
                if use_topk:
                    topk_logps = self._select_response_positions(
                        full_topk_logps,
                        response_length=response_length,
                        response_start_idx=response_start_idx,
                    )
                    if return_topk_indices:
                        topk_indices = self._select_response_positions(
                            full_topk_indices,
                            response_length=response_length,
                            response_start_idx=response_start_idx,
                        )

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = self._select_response_positions(
                        logits,
                        response_length=response_length,
                        response_start_idx=response_start_idx,
                    )
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    distillation_logits = (
                        logits
                        if distillation_temperature == temperature
                        else logits * (temperature / distillation_temperature)
                    )
                    if compute_all_logps:
                        if all_logps_mask is not None:
                            distillation_logits = distillation_logits[all_logps_mask.to(distillation_logits.device)]
                        all_logps = torch.log_softmax(distillation_logits, dim=-1)
                    if use_topk:
                        if topk_indices is None:
                            topk = min(distill_topk, distillation_logits.size(-1))
                            topk_logits, topk_indices = torch.topk(distillation_logits, topk, dim=-1)
                        else:
                            topk_logits = torch.gather(distillation_logits, dim=-1, index=topk_indices)
                        logsumexp = torch.logsumexp(distillation_logits, dim=-1, keepdim=True)
                        topk_logps = topk_logits - logsumexp
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared = (
                            self.calculate_sum_pi_squared_from_logits(logits)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.calculate_sum_pi_squared_from_logits, logits)
                        )

            outputs = {"log_probs": log_probs}
            if calculate_entropy:
                outputs["entropys"] = entropy
            if calculate_sum_pi_squared:
                outputs["sum_pi_squared"] = sum_pi_squared
            if compute_all_logps:
                outputs["all_logps"] = all_logps
            if use_topk:
                outputs["topk_logps"] = topk_logps
                if return_topk_indices:
                    outputs["topk_indices"] = topk_indices
            return outputs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
            return grad_norm

        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy: bool = False) -> dict[str, torch.Tensor]:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            dict[str, torch.Tensor]: a dict containing keys
                - ``log_probs``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``entropys``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``sum_pi_squared``: tensor of shape [batch_size, response_length]. torch.float32.
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)

        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        has_multi_modal_inputs = self._has_non_empty_multi_modal_inputs(
            data.non_tensor_batch.get("multi_modal_inputs")
        )

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        if self.use_prefix_grouper:
            select_keys += [k for k in ["prompts", "response_mask"] if k in data.batch]
            if "uid" in data.non_tensor_batch:
                non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        sum_pi_squared_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
            with torch.no_grad():
                outputs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(outputs["log_probs"])
            if calculate_entropy:
                entropy_lst.append(outputs["entropys"])
            if calculate_sum_pi_squared:
                sum_pi_squared_lst.append(outputs["sum_pi_squared"])

        log_probs = torch.concat(log_probs_lst, dim=0)
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if calculate_sum_pi_squared:
            sum_pi_squared = torch.concat(sum_pi_squared_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if calculate_sum_pi_squared:
                sum_pi_squared = restore_dynamic_batch(sum_pi_squared, batch_idx_list)

        outputs = {"log_probs": log_probs}
        if calculate_entropy:
            outputs["entropys"] = entropys
        if calculate_sum_pi_squared:
            outputs["sum_pi_squared"] = sum_pi_squared
        return outputs

    @staticmethod
    def _crop_iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = max(1.0, ax2 - ax1) * max(1.0, ay2 - ay1)
        area_b = max(1.0, bx2 - bx1) * max(1.0, by2 - by1)
        return inter / max(1.0, area_a + area_b - inter)

    def _sample_near_negative_crop_box(self, image_size, positive_box, seed, max_iou):
        width, height = [float(v) for v in image_size]
        if width < 1.0 or height < 1.0:
            raise ValueError(f"invalid image size for negative crop: {image_size}")

        x1, y1, x2, y2 = [float(v) for v in positive_box]
        x1, x2 = sorted((max(0.0, min(x1, width)), max(0.0, min(x2, width))))
        y1, y2 = sorted((max(0.0, min(y1, height)), max(0.0, min(y2, height))))
        if x2 <= x1:
            x1 = min(max(0.0, x1), max(0.0, width - 1.0))
            x2 = min(width, x1 + 1.0)
        if y2 <= y1:
            y1 = min(max(0.0, y1), max(0.0, height - 1.0))
            y2 = min(height, y1 + 1.0)
        positive = (x1, y1, x2, y2)

        crop_w = min(width, max(1.0, x2 - x1))
        crop_h = min(height, max(1.0, y2 - y1))
        max_left = width - crop_w
        max_top = height - crop_h
        if max_left < 0.0 or max_top < 0.0:
            raise ValueError("negative crop size cannot exceed the source image size")

        max_iou = max(0.0, min(float(max_iou), 1.0))
        pos_cx, pos_cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        min_center_dx = crop_w * (1.0 - max_iou) / (1.0 + max_iou)
        min_center_dy = crop_h * (1.0 - max_iou) / (1.0 + max_iou)

        import random
        rng = random.Random(int(seed))
        candidates = []
        counter = 0

        def add_candidate(left, top):
            nonlocal counter
            left = max(0.0, min(float(left), max_left))
            top = max(0.0, min(float(top), max_top))
            candidate = (left, top, left + crop_w, top + crop_h)
            iou = self._crop_iou(candidate, positive)
            if iou <= max_iou + 1e-12:
                cand_cx = (candidate[0] + candidate[2]) * 0.5
                cand_cy = (candidate[1] + candidate[3]) * 0.5
                center_distance = (
                    ((cand_cx - pos_cx) / max(crop_w, 1.0)) ** 2
                    + ((cand_cy - pos_cy) / max(crop_h, 1.0)) ** 2
                ) ** 0.5
                gap_x = max(positive[0] - candidate[2], candidate[0] - positive[2], 0.0)
                gap_y = max(positive[1] - candidate[3], candidate[1] - positive[3], 0.0)
                boundary_distance = (
                    (gap_x / max(crop_w, 1.0)) ** 2
                    + (gap_y / max(crop_h, 1.0)) ** 2
                ) ** 0.5
                candidates.append((center_distance, boundary_distance, iou, counter, candidate))
            counter += 1

        def add_center(center_x, center_y):
            add_candidate(center_x - crop_w * 0.5, center_y - crop_h * 0.5)

        for gap_scale in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0):
            gap_x = gap_scale * crop_w
            gap_y = gap_scale * crop_h
            for shift in (0.0, -0.25, 0.25, -0.5, 0.5, -0.75, 0.75):
                add_candidate(x1 - crop_w - gap_x, y1 + shift * crop_h)
                add_candidate(x2 + gap_x, y1 + shift * crop_h)
                add_candidate(x1 + shift * crop_w, y1 - crop_h - gap_y)
                add_candidate(x1 + shift * crop_w, y2 + gap_y)

        for scale in (1.0, 1.03, 1.1, 1.25, 1.5, 2.0, 2.75, 3.5):
            for sign in (-1.0, 1.0):
                for shift in (0.0, -0.25, 0.25, -0.5, 0.5):
                    add_center(pos_cx + sign * min_center_dx * scale, pos_cy + shift * crop_h)
                    add_center(pos_cx + shift * crop_w, pos_cy + sign * min_center_dy * scale)
                for sign_y in (-1.0, 1.0):
                    add_center(pos_cx + sign * min_center_dx * scale, pos_cy + sign_y * min_center_dy * scale)

        for _ in range(256):
            axis = rng.randrange(3)
            if axis == 0:
                offset_x = rng.choice((-1.0, 1.0)) * rng.uniform(min_center_dx, max(min_center_dx, 3.5 * crop_w))
                offset_y = rng.uniform(-0.75 * crop_h, 0.75 * crop_h)
            elif axis == 1:
                offset_x = rng.uniform(-0.75 * crop_w, 0.75 * crop_w)
                offset_y = rng.choice((-1.0, 1.0)) * rng.uniform(min_center_dy, max(min_center_dy, 3.5 * crop_h))
            else:
                offset_x = rng.choice((-1.0, 1.0)) * rng.uniform(min_center_dx, max(min_center_dx, 3.0 * crop_w))
                offset_y = rng.choice((-1.0, 1.0)) * rng.uniform(min_center_dy, max(min_center_dy, 3.0 * crop_h))
            add_center(pos_cx + offset_x, pos_cy + offset_y)

        if not candidates:
            grid_x = min(33, max(2, int(width // max(crop_w * 0.5, 1.0)) + 1))
            grid_y = min(33, max(2, int(height // max(crop_h * 0.5, 1.0)) + 1))
            for ix in range(grid_x):
                left = max_left * ix / max(1, grid_x - 1)
                for iy in range(grid_y):
                    top = max_top * iy / max(1, grid_y - 1)
                    add_candidate(left, top)

        if not candidates:
            raise ValueError(f"could not sample same-size near negative crop below max IoU={max_iou:.4f}")
        return min(candidates, key=lambda item: item[:4])[-1]

    def _sample_negative_crop(self, image, positive_box, output_size, seed, cfg):
        image = image.convert("RGB") if isinstance(image, Image.Image) else Image.open(image).convert("RGB")
        width, height = image.size
        x1, y1, x2, y2 = [float(v) for v in positive_box]
        positive = (max(0.0, min(x1, width - 1)), max(0.0, min(y1, height - 1)),
                    max(1.0, min(x2, width)), max(1.0, min(y2, height)))
        import random
        rng = random.Random(int(seed))
        if cfg.get("distillation_target_mode") == "crop_only_v9_5":
            candidate = self._sample_near_negative_crop_box(
                image.size,
                positive_box,
                seed,
                cfg.get("negative_crop_max_iou", 0.1),
            )
            crop = image.crop(tuple(int(round(v)) for v in candidate)).resize(
                output_size, Image.Resampling.BICUBIC
            )
            if cfg.get("crop_red_frame_enabled", False):
                crop = add_red_frame(crop, cfg.get("crop_red_frame_width", 4))
            return crop
        min_area, max_area = cfg.get("negative_crop_min_scale", 0.05), cfg.get("negative_crop_max_scale", 0.5)
        candidates = []
        for _ in range(64):
            area = width * height * rng.uniform(min_area, max_area)
            aspect = 2.0 ** rng.uniform(-1.0, 1.0)
            cw = max(2.0, min(width, (area * aspect) ** 0.5))
            ch = max(2.0, min(height, area / cw))
            left = rng.uniform(0.0, max(0.0, width - cw))
            top = rng.uniform(0.0, max(0.0, height - ch))
            candidate = (left, top, left + cw, top + ch)
            iou = self._crop_iou(candidate, positive)
            candidates.append((iou, candidate))
            if iou <= cfg.get("negative_crop_max_iou", 0.1):
                crop = image.crop(tuple(int(round(v)) for v in candidate)).resize(
                    output_size, Image.Resampling.BICUBIC
                )
                if cfg.get("crop_red_frame_enabled", False):
                    crop = add_red_frame(crop, cfg.get("crop_red_frame_width", 4))
                return crop
        iou, candidate = min(candidates, key=lambda item: item[0])
        if iou > cfg.get("negative_crop_max_iou", 0.1):
            raise ValueError(f"could not sample negative crop below max IoU: best={iou:.4f}")
        crop = image.crop(tuple(int(round(v)) for v in candidate)).resize(
            output_size, Image.Resampling.BICUBIC
        )
        if cfg.get("crop_red_frame_enabled", False):
            crop = add_red_frame(crop, cfg.get("crop_red_frame_width", 4))
        return crop

    @staticmethod
    def _pad_position_records(records, device):
        if records[0].dim() == 1:
            return torch.nn.utils.rnn.pad_sequence(records, batch_first=True, padding_value=0).to(device)
        max_len = max(record.shape[-1] for record in records)
        result = torch.zeros((len(records), records[0].shape[0], max_len), dtype=records[0].dtype, device=device)
        for i, record in enumerate(records):
            result[i, :, : record.shape[-1]] = record.to(device)
        return result

    @staticmethod
    def _negative_rollout_seed(base_seed, step, uid, rollout_id):
        seed_material = f"{base_seed}:{step}:{uid}:{rollout_id}"
        return int.from_bytes(hashlib.sha256(seed_material.encode("utf-8")).digest()[:8], "big")

    def _build_negative_rollout_batch(self, model_inputs, cfg, negative_index=0):
        if self.processor is None:
            raise RuntimeError("crop_full_abs_contrast_rank requires actor.processor")
        records = []
        step = int(self._current_global_steps or 0)
        for row_idx in range(model_inputs["responses"].shape[0]):
            full_images = model_inputs["ranking_full_images"][row_idx]
            positive_images = model_inputs["ranking_positive_images"][row_idx]
            if len(positive_images) != 1:
                raise ValueError("crop_full_abs_contrast_rank currently requires exactly one image per sample")
            positive_image = positive_images[0]
            if not isinstance(positive_image, Image.Image):
                if isinstance(positive_image, dict):
                    positive_image = positive_image.get("image", positive_image.get("path"))
                positive_image = Image.open(positive_image).convert("RGB")
            if cfg.get("distillation_target_mode") == "crop_only_v9_7":
                crop = Image.new("RGB", positive_image.size, color=(0, 0, 0))
            else:
                if len(full_images) != 1:
                    raise ValueError("crop_full_abs_contrast_rank currently requires exactly one image per sample")
                rollout_id = model_inputs["ranking_rollout_id"][row_idx]
                if cfg.get("distillation_target_mode") == "crop_only_v10_2":
                    # One negative crop is shared by all rollouts expanded from
                    # this sample encounter. A fresh UID and global step are
                    # assigned when the sample is encountered again next epoch.
                    rollout_id = f"sample_shared:{negative_index}"
                elif negative_index:
                    rollout_id = f"{rollout_id}:{negative_index}"
                seed = self._negative_rollout_seed(
                    cfg.get("negative_crop_seed", 20260813), step,
                    model_inputs["ranking_uid"][row_idx], rollout_id,
                )
                crop = self._sample_negative_crop(
                    full_images[0], model_inputs["ranking_bbox"][row_idx], positive_image.size, seed, cfg
                )
            messages = deepcopy(model_inputs["ranking_raw_prompt"][row_idx])
            image_count = 0
            for message in messages:
                if isinstance(message.get("content"), list):
                    for item in message["content"]:
                        if isinstance(item, dict) and item.get("type") == "image":
                            item["image"] = crop
                            image_count += 1
            if image_count != 1:
                raise ValueError(f"expected one image placeholder, found {image_count}")
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            processed = dict(self.processor(
                text=[prompt], images=[crop], videos=None, return_tensors="pt", truncation=True,
                max_length=cfg.get("max_reprompt_len", 10240),
            ))
            prompt_ids = processed.pop("input_ids").squeeze(0)
            processed.pop("attention_mask")
            processed.pop("mm_token_type_ids", None)
            response_start = int(model_inputs["teacher_response_start_idx"][row_idx])
            base_prompt_ids = model_inputs["teacher_input_ids"][row_idx, :response_start].detach().cpu()
            if not torch.equal(prompt_ids, base_prompt_ids):
                raise ValueError("negative crop changed the teacher prompt token layout; resize/grid invariant failed")
            records.append({
                "input_ids": model_inputs["teacher_input_ids"][row_idx],
                "attention_mask": model_inputs["teacher_attention_mask"][row_idx],
                "position_ids": model_inputs["teacher_position_ids"][row_idx],
                "response_start_idx": torch.tensor(response_start, dtype=torch.long),
                "responses": model_inputs["responses"][row_idx].detach().cpu(),
                "multi_modal_inputs": processed,
            })
        device = model_inputs["responses"].device
        return {
            "responses": torch.stack([record["responses"] for record in records]).to(device),
            "input_ids": torch.stack([record["input_ids"] for record in records]).to(device),
            "attention_mask": torch.stack([record["attention_mask"] for record in records]).to(device),
            "position_ids": torch.stack([record["position_ids"] for record in records]).to(device),
            "response_start_idx": torch.stack([record["response_start_idx"] for record in records]).to(device),
            "multi_modal_inputs": [record["multi_modal_inputs"] for record in records],
        }

    def _precompute_multi_negative_probability_gate(self, data, temperature, pad_token_id):
        """Build the v8-8 gate from four negative-teacher sampled-token gaps."""
        cfg = self.self_distillation_cfg
        response_mask = data.batch["response_mask"] * data.batch["self_distillation_mask"].unsqueeze(1)
        negative_count = 4
        gap_values = torch.zeros(
            (*response_mask.shape, negative_count), dtype=torch.float32
        )
        row_offset = 0
        gate_chunk_size = int(cfg.get("gate_forward_chunk_size", 1))
        teacher_model = self.teacher_module or self.actor_module
        for chunk in data.split(gate_chunk_size):
            chunk = chunk.to(get_device_id())
            inputs = {**chunk.batch, **chunk.non_tensor_batch, "pad_token_id": pad_token_id}
            valid_mask = (
                inputs["response_mask"] * inputs["self_distillation_mask"].unsqueeze(1)
            ).bool()
            chunk_size = inputs["responses"].shape[0]
            if valid_mask.any():
                positive = {
                    "responses": inputs["responses"],
                    "input_ids": inputs["teacher_input_ids"],
                    "attention_mask": inputs["teacher_attention_mask"],
                    "position_ids": inputs["teacher_position_ids"],
                    "response_start_idx": inputs["teacher_response_start_idx"],
                    "multi_modal_inputs": inputs.get("teacher_multi_modal_inputs"),
                }
                token_ids = inputs["responses"][valid_mask].to(dtype=torch.long)
                with torch.no_grad():
                    positive_outputs = self._forward_micro_batch(
                        positive,
                        temperature=temperature,
                        distillation_temperature=cfg.distillation_temperature,
                        return_all_logps=True,
                        all_logps_mask=valid_mask,
                        module=teacher_model,
                    )
                    positive_logps = positive_outputs["all_logps"]
                    positive_probs = positive_logps.detach().gather(
                        -1, token_ids.unsqueeze(-1)
                    ).squeeze(-1).exp().float().cpu()
                del positive_outputs, positive_logps, positive
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                chunk_gaps = []
                for negative_index in range(1, negative_count + 1):
                    negative_batch = self._build_negative_rollout_batch(
                        inputs, cfg, negative_index=negative_index
                    )
                    with torch.no_grad():
                        negative_outputs = self._forward_micro_batch(
                            negative_batch,
                            temperature=temperature,
                            distillation_temperature=cfg.distillation_temperature,
                            return_all_logps=True,
                            all_logps_mask=valid_mask,
                            module=teacher_model,
                        )
                        negative_logps = negative_outputs["all_logps"]
                        negative_probs = negative_logps.detach().gather(
                            -1, token_ids.unsqueeze(-1)
                        ).squeeze(-1).exp().float().cpu()
                    chunk_gaps.append((positive_probs - negative_probs).abs())
                    # Keep only the sampled-token gap; release the full-vocabulary
                    # logits and multimodal tensors before the next negative forward.
                    del negative_outputs, negative_logps, negative_probs, negative_batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                chunk_gaps = torch.stack(chunk_gaps, dim=-1)
                valid_cpu = valid_mask.detach().cpu()
                gap_values[row_offset : row_offset + chunk_size][valid_cpu] = chunk_gaps
                del chunk_gaps, positive_probs, token_ids
            row_offset += chunk_size

        device = get_device_id()
        gaps_device = gap_values.to(device)
        valid_device = response_mask.to(device)
        ranking_gate, threshold = build_batch_mean_multi_negative_probability_gate(
            gaps_device, valid_device, distributed=True
        )
        cap = cfg.get("max_gated_tokens_per_batch", None)
        if cap is not None:
            scores = torch.topk(gaps_device, k=2, dim=-1, largest=False).values.mean(dim=-1)
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            candidates = [
                (float(scores[r, t]), rank, r, t)
                for r, t in ranking_gate.nonzero(as_tuple=False).tolist()
            ]
            if torch.distributed.is_initialized():
                gathered = [None] * torch.distributed.get_world_size()
                torch.distributed.all_gather_object(gathered, candidates)
                candidates = [item for part in gathered for item in part]
            selected = {
                (rank_i, r, t)
                for _, rank_i, r, t in sorted(candidates, reverse=True)[: int(cap)]
            }
            ranking_gate = torch.tensor(
                [[(rank, r, t) in selected for t in range(ranking_gate.shape[1])] for r in range(ranking_gate.shape[0])],
                dtype=torch.bool,
                device=ranking_gate.device,
            )
        gated_count = ranking_gate.sum().to(torch.long)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(gated_count, op=torch.distributed.ReduceOp.SUM)
        del gaps_device, gap_values
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return ranking_gate.cpu(), float(threshold), int(gated_count)

    def _precompute_crop_full_rank_gate(self, data, temperature, pad_token_id):
        cfg = self_distillation_cfg = self.self_distillation_cfg
        response_mask = data.batch["response_mask"] * data.batch["self_distillation_mask"].unsqueeze(1)
        if cfg.get("distillation_target_mode") in {"crop_only_v9_11", "crop_only_v9_12", "crop_only_v9_13", "crop_only_v9_14", "crop_only_v10_2"}:
            sample_filter = data.batch["self_distillation_sample_filter"].to(
                device=response_mask.device, dtype=response_mask.dtype
            )
            response_mask = response_mask * sample_filter.unsqueeze(1)
        if cfg.get("ranking_token_selection", "batch_mean_jsd") == "multi_negative_probability":
            return self._precompute_multi_negative_probability_gate(data, temperature, pad_token_id)
        if cfg.get("ranking_token_selection", "batch_mean_jsd") == "all_valid":
            gate = response_mask.bool()
            selected_count = gate.sum().to(torch.long)
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(selected_count, op=torch.distributed.ReduceOp.SUM)
            return gate.cpu(), 0.0, int(selected_count)

        teacher_model = self.teacher_module or self.actor_module
        distillation_target_mode = cfg.get("distillation_target_mode")
        use_student_topk_gate = distillation_target_mode == "crop_only_v9_1"
        use_student_full_vocab_gate = distillation_target_mode in {"crop_only_v9_2", "crop_only_v9_3", "crop_only_v9_5", "crop_only_v9_6", "crop_only_v9_7", "crop_only_v9_8", "crop_only_v9_9", "crop_only_v9_10", "crop_only_v9_11", "crop_only_v9_13", "crop_only_v9_14", "crop_only_v9_15", "crop_only_v10_2"}
        jsd = torch.zeros(response_mask.shape, dtype=torch.float32, device=get_device_id())
        row_offset = 0
        gate_chunk_size = int(cfg.get("gate_forward_chunk_size", 1))
        for chunk in data.split(gate_chunk_size):
            chunk = chunk.to(get_device_id())
            inputs = {**chunk.batch, **chunk.non_tensor_batch, "pad_token_id": pad_token_id}
            student = {"responses": inputs["responses"], "input_ids": inputs["input_ids"],
                       "attention_mask": inputs["attention_mask"], "position_ids": inputs["position_ids"],
                       "multi_modal_inputs": inputs.get("multi_modal_inputs")}
            positive = {"responses": inputs["responses"], "input_ids": inputs["teacher_input_ids"],
                        "attention_mask": inputs["teacher_attention_mask"], "position_ids": inputs["teacher_position_ids"],
                        "response_start_idx": inputs["teacher_response_start_idx"],
                        "multi_modal_inputs": inputs.get("teacher_multi_modal_inputs")}
            with torch.no_grad():
                if use_student_topk_gate:
                    student_outputs = self._forward_micro_batch(
                        student, temperature=temperature,
                        distillation_temperature=cfg.distillation_temperature,
                        distill_topk=100, return_all_logps=False, module=self.actor_module,
                    )
                    student_topk = student_outputs["topk_logps"]
                    positive_outputs = self._forward_micro_batch(
                        positive, temperature=temperature,
                        distillation_temperature=cfg.distillation_temperature,
                        distill_topk=100, topk_indices=student_outputs["topk_indices"],
                        return_all_logps=False, module=teacher_model,
                    )
                    jsd[row_offset : row_offset + student_topk.shape[0]] = compute_topk_tail_jsd(
                        student_topk, positive_outputs["topk_logps"]
                    )
                    chunk_rows = student_topk.shape[0]
                elif use_student_full_vocab_gate:
                    student_outputs = self._forward_micro_batch(
                        student, temperature=temperature,
                        distillation_temperature=cfg.distillation_temperature,
                        return_all_logps=True, module=self.actor_module,
                    )
                    positive_outputs = self._forward_micro_batch(
                        positive, temperature=temperature,
                        distillation_temperature=cfg.distillation_temperature,
                        return_all_logps=True, module=teacher_model,
                    )
                    student_all = student_outputs["all_logps"]
                    positive_all = positive_outputs["all_logps"]
                    jsd[row_offset : row_offset + student_all.shape[0]] = compute_log_jsd(
                        student_all, positive_all
                    )
                    chunk_rows = student_all.shape[0]
                else:
                    full = {"responses": inputs["responses"], "input_ids": inputs["reference_teacher_input_ids"],
                            "attention_mask": inputs["reference_teacher_attention_mask"], "position_ids": inputs["reference_teacher_position_ids"],
                            "response_start_idx": inputs["reference_teacher_response_start_idx"],
                            "multi_modal_inputs": inputs.get("reference_teacher_multi_modal_inputs")}
                    p = self._forward_micro_batch(positive, temperature=temperature,
                        distillation_temperature=cfg.distillation_temperature, return_all_logps=True, module=teacher_model)["all_logps"]
                    f = self._forward_micro_batch(full, temperature=temperature,
                        distillation_temperature=cfg.distillation_temperature, return_all_logps=True, module=teacher_model)["all_logps"]
                    jsd[row_offset : row_offset + p.shape[0]] = compute_log_jsd(p, f)
                    chunk_rows = p.shape[0]
            row_offset += chunk_rows
        if distillation_target_mode == "crop_only_v9_3":
            gate, response_thresholds = build_per_response_mean_jsd_gate(
                jsd, response_mask.to(jsd.device)
            )
            # Keep the existing scalar metric for compatibility. The actual
            # gate uses one threshold per response row.
            threshold = float(response_thresholds.mean().detach().item())
        else:
            gate, threshold = build_batch_mean_jsd_gate(jsd, response_mask.to(jsd.device), distributed=True)
        if distillation_target_mode in {"crop_only_v9_2", "crop_only_v9_8"}:
            wrong_mask = data.batch["self_distillation_wrong_mask"].to(device=gate.device) > 0
            gate = gate & wrong_mask.unsqueeze(1)
        cap = cfg.get("max_gated_tokens_per_batch", None)
        if cap is not None:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            candidates = [(float(jsd[r, t]), rank, r, t) for r, t in gate.nonzero(as_tuple=False).tolist()]
            if torch.distributed.is_initialized():
                gathered = [None] * torch.distributed.get_world_size()
                torch.distributed.all_gather_object(gathered, candidates)
                candidates = [item for part in gathered for item in part]
            selected = {(rank_i, r, t) for _, rank_i, r, t in sorted(candidates, reverse=True)[: int(cap)]}
            gate = torch.tensor([[(rank, r, t) in selected for t in range(gate.shape[1])] for r in range(gate.shape[0])],
                                dtype=torch.bool, device=gate.device)
        gated_count = gate.sum().to(torch.long)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(gated_count, op=torch.distributed.ReduceOp.SUM)
        return gate.cpu(), float(threshold), int(gated_count)

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        self._current_global_steps = data.meta_info.get("global_steps")
        # make sure we are in training mode
        self.actor_module.train()
        if data.meta_info.get("teacher_sft_global_valid_tokens") is not None:
            # Recompute the denominator from the dispatched local SFT shard and
            # aggregate it across DP ranks.  The driver-side value is only a
            # fallback; using it directly would normalize by a local token count.
            local_sft_tokens = torch.zeros((), device=get_device_id(), dtype=torch.float32)
            if "response_mask" in data.batch and "teacher_trajectory_mask" in data.batch:
                local_sft_tokens = (
                    data.batch["response_mask"].to(get_device_id(), dtype=torch.float32)
                    * data.batch["teacher_trajectory_mask"].to(get_device_id(), dtype=torch.float32).unsqueeze(1)
                ).sum()
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(local_sft_tokens, op=torch.distributed.ReduceOp.SUM)
            self.config.global_batch_info["teacher_sft_global_valid_tokens"] = max(
                float(local_sft_tokens.item()), 1.0
            )

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

        self_distillation_enabled = loss_mode == "vopd"
        self_distillation_cfg = getattr(self.config, "self_distillation", None)
        distillation_target_mode = (
            self_distillation_cfg.get("distillation_target_mode", "standard")
            if self_distillation_cfg is not None
            else "standard"
        )
        is_v9_13_teacher_sft_batch = bool(data.meta_info.get("v9_13_teacher_sft_batch", False))
        grpo_loss_weight = float(
            self_distillation_cfg.get("grpo_loss_weight", 0.0)
            if self_distillation_cfg is not None
            else 0.0
        )
        grpo_positive_advantage_gate = bool(
            self_distillation_cfg.get("grpo_positive_advantage_gate", False)
            if self_distillation_cfg is not None
            else False
        )
        use_crop_black_contrast = distillation_target_mode == "crop_black_contrast"
        use_crop_full_abs_rank = distillation_target_mode in {
            "crop_full_abs_contrast_rank",
            "crop_full_abs_contrast_rank_v2",
            "crop_full_abs_contrast_rank_v3",
            "crop_full_abs_contrast_rank_v4",
            "crop_full_abs_contrast_rank_v5",
            "crop_full_abs_contrast_rank_v6",
            "crop_only_v8_1",
            "crop_only_v8_2",
            "crop_only_v8_3",
            "crop_only_v8_4",
            "crop_only_v8_5",
            "crop_only_v8_6",
            "crop_only_v8_7",
            "crop_only_v8_3_2",
            "crop_only_v8_4_2",
            "crop_only_v8_5_2",
            "crop_only_v8_6_2",
            "crop_only_v8_8",
            "crop_only_v8_9",
            "crop_only_v8_10",
            "crop_only_v9_1",
            "crop_only_v9_2",
            "crop_only_v9_3",
            "crop_only_v9_5",
            "crop_only_v9_6",
            "crop_only_v9_7",
            "crop_only_v9_8",
            "crop_only_v9_9",
            "crop_only_v9_15",
            "crop_only_v9_10",
            "crop_only_v9_11",
            "crop_only_v10_2",
            "crop_only_v9_12",
            "crop_only_v9_13",
            "crop_only_v9_14",
            "crop_only_v10_1",
        }
        if is_v9_13_teacher_sft_batch:
            use_crop_full_abs_rank = False
        ungated_weighted_auxiliary_modes = {"crop_only_v8_3", "crop_only_v8_4", "crop_only_v8_5", "crop_only_v8_6", "crop_only_v8_7", "crop_only_v8_3_2", "crop_only_v8_4_2", "crop_only_v8_5_2", "crop_only_v8_6_2"}
        ranking_token_selection = self_distillation_cfg.get("ranking_token_selection", "batch_mean_jsd")
        use_reference_teacher = (
            use_crop_black_contrast
            or distillation_target_mode == "crop_full_abs_contrast_rank"
            or distillation_target_mode in {"crop_only_v8_4", "crop_only_v8_6", "crop_only_v8_4_2", "crop_only_v8_6_2"}
            or (
                use_crop_full_abs_rank
                and distillation_target_mode not in {"crop_only_v9_1", "crop_only_v9_2", "crop_only_v9_3", "crop_only_v9_5", "crop_only_v9_6", "crop_only_v9_7", "crop_only_v9_8", "crop_only_v9_9", "crop_only_v9_10", "crop_only_v9_11", "crop_only_v9_13", "crop_only_v9_14", "crop_only_v9_15", "crop_only_v10_2"}
                and distillation_target_mode not in ungated_weighted_auxiliary_modes
                and ranking_token_selection == "batch_mean_jsd"
            )
        )
        self.self_distillation_cfg = self_distillation_cfg
        if self_distillation_enabled:
            if self_distillation_cfg is None:
                raise ValueError(f"loss_mode={loss_mode} requires actor.self_distillation config.")
            self_distillation_required_keys = (
                {"teacher_trajectory_mask"}
                if is_v9_13_teacher_sft_batch
                else {
                    "teacher_input_ids", "teacher_attention_mask", "teacher_position_ids",
                    "teacher_response_start_idx", "self_distillation_mask",
                }
            )
            if use_reference_teacher:
                self_distillation_required_keys.update(
                    {
                        "reference_teacher_input_ids",
                        "reference_teacher_attention_mask",
                        "reference_teacher_position_ids",
                        "reference_teacher_response_start_idx",
                    }
                )
            if distillation_target_mode in {"crop_only_v9_2", "crop_only_v9_8"}:
                self_distillation_required_keys.add("self_distillation_wrong_mask")
            if distillation_target_mode in {"crop_only_v9_11", "crop_only_v9_12", "crop_only_v9_13", "crop_only_v9_14", "crop_only_v10_2"} and not is_v9_13_teacher_sft_batch:
                self_distillation_required_keys.add("self_distillation_sample_filter")
            assert self_distillation_required_keys.issubset(set(data.batch.keys())), f"Missing required keys: {self_distillation_required_keys - set(data.batch.keys())}"

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
        ]
        if not self_distillation_enabled or "advantages" in data.batch.keys():
            select_keys.append("advantages")
        if distillation_target_mode in {"crop_only_v9_9", "crop_only_v9_15"}:
            if "inverse_advantage_weights" not in data.batch.keys():
                raise ValueError(f"{distillation_target_mode} requires advantage weights computed by the trainer")
            select_keys.append("inverse_advantage_weights")
        if (self.use_prefix_grouper or is_v9_13_teacher_sft_batch) and "prompts" in data.batch.keys():
            select_keys.append("prompts")
        if self.config.use_kl_loss and not is_v9_13_teacher_sft_batch:
            select_keys.append("ref_log_prob")
        if self_distillation_enabled:
            select_keys.extend(list(self_distillation_required_keys))
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = self._has_non_empty_multi_modal_inputs(
            data.non_tensor_batch.get("multi_modal_inputs")
        )
        has_teacher_multi_modal_inputs = self._has_non_empty_multi_modal_inputs(
            data.non_tensor_batch.get("teacher_multi_modal_inputs")
        )
        has_reference_teacher_multi_modal_inputs = self._has_non_empty_multi_modal_inputs(
            data.non_tensor_batch.get("reference_teacher_multi_modal_inputs")
        )
        non_tensor_select_keys = []
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")
        if has_teacher_multi_modal_inputs:
            non_tensor_select_keys.append("teacher_multi_modal_inputs")
        if has_reference_teacher_multi_modal_inputs:
            non_tensor_select_keys.append("reference_teacher_multi_modal_inputs")
        if use_crop_full_abs_rank:
            non_tensor_select_keys.extend(
                key for key in ("ranking_raw_prompt", "ranking_positive_images", "ranking_full_images", "ranking_bbox", "ranking_uid", "ranking_rollout_id")
                if key in data.non_tensor_batch
            )
        if self.use_prefix_grouper and "uid" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("uid")
        if is_v9_13_teacher_sft_batch:
            non_tensor_select_keys.extend(
                key for key in ("uid", "raw_prompt")
                if key in data.non_tensor_batch and key not in non_tensor_select_keys
            )

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_crop_full_abs_rank:
            if distillation_target_mode in ungated_weighted_auxiliary_modes:
                ranking_gate = (
                    data.batch["response_mask"]
                    * data.batch["self_distillation_mask"].unsqueeze(1)
                ).bool()
                selected_count = ranking_gate.sum().to(device=get_device_id(), dtype=torch.long)
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(selected_count, op=torch.distributed.ReduceOp.SUM)
                ranking_batch_token_count = int(selected_count)
                ranking_threshold = 0.0
                ranking_gate = ranking_gate.cpu()
            else:
                ranking_gate, ranking_threshold, ranking_batch_token_count = self._precompute_crop_full_rank_gate(
                    data, temperature=temperature, pad_token_id=pad_token_id
                )
            data.batch["ranking_gate"] = ranking_gate
        else:
            ranking_threshold = 0.0
            ranking_batch_token_count = 0

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {
            "actor/pg_loss": 0.0,
            "actor/kl_loss": 0.0,
        }
        if self_distillation_enabled:
            metrics["actor/grpo_loss"] = 0.0
            metrics["actor/grpo_loss_weighted"] = 0.0
            if use_crop_full_abs_rank:
                if distillation_target_mode not in ungated_weighted_auxiliary_modes:
                    threshold_key = (
                        "self_distillation/batch_probability_gap_threshold"
                        if distillation_target_mode == "crop_only_v8_8"
                        else "self_distillation/batch_jsd_threshold"
                    )
                    metrics[threshold_key] = ranking_threshold
                    metrics["self_distillation/batch_gated_tokens"] = ranking_batch_token_count
                metrics["self_distillation/batch_ranking_selected_tokens"] = ranking_batch_token_count
            metrics["actor/vopd_loss"] = 0.0
            metrics["actor/vopd_loss_weighted"] = 0.0
        distill_dump_chunks = []
        stage_wall_time_totals = None
        if self_distillation_enabled:
            stage_wall_time_totals = {
                "timing_s/update_actor/student_forward": 0.0,
                "timing_s/update_actor/teacher_forward": 0.0,
                "timing_s/update_actor/reference_teacher_forward": 0.0,
                "timing_s/update_actor/loss_compute": 0.0,
                "timing_s/update_actor/backward": 0.0,
                "timing_s/update_actor/optimizer_step": 0.0,
                "timing_s/update_actor/teacher_ema_update": 0.0,
            }
        did_update = False
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs.get("advantages")

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)
                    self_distillation_mask = model_inputs.get("self_distillation_mask") if self_distillation_enabled else None
                    self_distillation_sample_filter = (
                        model_inputs.get("self_distillation_sample_filter")
                        if distillation_target_mode in {"crop_only_v9_11", "crop_only_v9_12", "crop_only_v9_13", "crop_only_v9_14", "crop_only_v10_2"}
                        else None
                    )
                    policy_fallback_mask = None
                    if self_distillation_enabled and self_distillation_mask is not None:
                        policy_fallback_mask = (self_distillation_mask <= 0.5).to(response_mask.dtype)
                        micro_batch_metrics["actor/policy_fallback_fraction"] = (
                            policy_fallback_mask.float().mean().detach().item()
                        )

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    teacher_regularization = self_distillation_cfg.get("teacher_regularization", "ema")
                    teacher_model_source = self_distillation_cfg.get("teacher_model_source", "legacy")
                    use_trust_region_teacher = teacher_model_source == "legacy" and teacher_regularization == "trust-region"
                    if use_trust_region_teacher and self.use_fused_kernels:
                        raise ValueError("trust-region teacher requires disabling fused kernels to access logits.")
                    # all return: (bsz, response_length)
                    needs_auxiliary_full_logits = distillation_target_mode in {
                        "crop_only_v8_1",
                        "crop_only_v8_2",
                        "crop_only_v8_3",
                        "crop_only_v8_4",
                        "crop_only_v8_5",
                        "crop_only_v8_6",
                        "crop_only_v8_7",
                        "crop_only_v8_3_2",
                        "crop_only_v8_4_2",
                        "crop_only_v8_5_2",
                        "crop_only_v8_6_2",
                        "crop_only_v8_8",
                        "crop_only_v8_9",
                        "crop_only_v8_10",
                        "crop_only_v9_2",
                        "crop_only_v9_3",
                        "crop_only_v9_5",
                        "crop_only_v9_6",
                        "crop_only_v9_7",
                        "crop_only_v9_8",
                        "crop_only_v9_9",
                        "crop_only_v9_15",
                        "crop_only_v9_10",
                        "crop_only_v9_11",
                        "crop_only_v10_2",
                        "crop_only_v9_12",
                        "crop_only_v9_13",
                        "crop_only_v9_14",
                        "crop_only_v10_1",
                    }
                    if is_v9_13_teacher_sft_batch:
                        needs_auxiliary_full_logits = False
                    return_all_logps = self_distillation_cfg.full_logit_distillation and (
                        not self_distillation_cfg.distillation_topk or needs_auxiliary_full_logits
                    )
                    distill_topk = self_distillation_cfg.distillation_topk if self_distillation_cfg.full_logit_distillation else None
                    if is_v9_13_teacher_sft_batch:
                        return_all_logps = False
                        distill_topk = None
                    distillation_temperature = self_distillation_cfg.get("distillation_temperature", temperature)
                    student_forward_start = time.perf_counter()
                    outputs = self._forward_micro_batch(
                        model_inputs,
                        temperature=temperature,
                        distillation_temperature=distillation_temperature,
                        calculate_entropy=calculate_entropy,
                        allow_all_logps_with_topk=needs_auxiliary_full_logits,
                        return_all_logps=return_all_logps,
                        distill_topk=distill_topk,
                    )
                    if self_distillation_enabled:
                        student_forward_time = time.perf_counter() - student_forward_start
                        stage_wall_time_totals["timing_s/update_actor/student_forward"] += student_forward_time
                    log_prob = outputs["log_probs"]
                    entropy = outputs["entropys"] if calculate_entropy else None
                    student_all_logps = outputs.get("all_logps") if return_all_logps else None
                    student_topk_logps = outputs.get("topk_logps") if distill_topk else None
                    student_topk_indices = outputs.get("topk_indices") if distill_topk else None

                    if is_v9_13_teacher_sft_batch:
                        sft_loss, sft_metrics = compute_teacher_trajectory_sft_loss(
                            student_log_probs=log_prob,
                            response_mask=response_mask,
                            teacher_trajectory_mask=model_inputs["teacher_trajectory_mask"],
                            global_valid_token_count=self.config.global_batch_info.get(
                                "teacher_sft_global_valid_tokens"
                            ),
                            dp_size=(torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1),
                        )
                        # The SFT helper already divides by the global valid-token
                        # count. Summing its micro-batch gradients therefore gives
                        # the global token mean without PPO mini-batch rescaling.
                        loss = sft_loss
                        backward_start = time.perf_counter()
                        if self.scaler is not None:
                            self.scaler.scale(loss).backward()
                        else:
                            loss.backward()
                        stage_wall_time_totals["timing_s/update_actor/backward"] += (
                            time.perf_counter() - backward_start
                        )
                        metrics["actor/pg_loss"] += sft_loss.detach().item()
                        metrics["actor/vopd_loss"] += sft_loss.detach().item()
                        metrics["actor/vopd_loss_weighted"] += sft_loss.detach().item()
                        append_to_dict(metrics, sft_metrics)
                        continue

                    # for fully_async_policy
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    if self_distillation_enabled:
                        teacher_inputs = {
                            "responses": model_inputs["responses"],
                            "input_ids": model_inputs["teacher_input_ids"],
                            "attention_mask": model_inputs["teacher_attention_mask"],
                            "position_ids": model_inputs["teacher_position_ids"],
                            "response_start_idx": model_inputs["teacher_response_start_idx"],
                        }
                        if "teacher_multi_modal_inputs" in model_inputs:
                            teacher_inputs["multi_modal_inputs"] = model_inputs["teacher_multi_modal_inputs"]
                        teacher_model = self.teacher_module or self.actor_module
                        if use_trust_region_teacher and (
                            self.teacher_module is None or self.teacher_module is self.actor_module
                        ):
                            raise ValueError("trust-region teacher requires a separate teacher_module in the actor worker.")
                        with torch.no_grad():
                            teacher_forward_start = time.perf_counter()
                            teacher_outputs = self._forward_micro_batch(
                                teacher_inputs,
                                temperature=temperature,
                                distillation_temperature=distillation_temperature,
                                calculate_entropy=False,
                                allow_all_logps_with_topk=needs_auxiliary_full_logits,
                                return_all_logps=return_all_logps,
                                distill_topk=distill_topk,
                                topk_indices=student_topk_indices,
                                module=teacher_model,
                            )
                            teacher_forward_time = time.perf_counter() - teacher_forward_start
                        stage_wall_time_totals["timing_s/update_actor/teacher_forward"] += teacher_forward_time
                        teacher_log_prob = teacher_outputs["log_probs"]
                        teacher_all_logps = teacher_outputs.get("all_logps") if return_all_logps else None
                        teacher_topk_logps = teacher_outputs.get("topk_logps") if distill_topk else None
                        reference_teacher_all_logps = None
                        if use_reference_teacher:
                            reference_teacher_inputs = {
                                "responses": model_inputs["responses"],
                                "input_ids": model_inputs["reference_teacher_input_ids"],
                                "attention_mask": model_inputs["reference_teacher_attention_mask"],
                                "position_ids": model_inputs["reference_teacher_position_ids"],
                                "response_start_idx": model_inputs["reference_teacher_response_start_idx"],
                            }
                            if "reference_teacher_multi_modal_inputs" in model_inputs:
                                reference_teacher_inputs["multi_modal_inputs"] = model_inputs[
                                    "reference_teacher_multi_modal_inputs"
                                ]
                            with torch.no_grad():
                                reference_teacher_forward_start = time.perf_counter()
                                reference_teacher_outputs = self._forward_micro_batch(
                                    reference_teacher_inputs,
                                    temperature=temperature,
                                    distillation_temperature=distillation_temperature,
                                    calculate_entropy=False,
                                    return_all_logps=True,
                                    module=teacher_model,
                                )
                                reference_teacher_forward_time = (
                                    time.perf_counter() - reference_teacher_forward_start
                                )
                            stage_wall_time_totals[
                                "timing_s/update_actor/reference_teacher_forward"
                            ] += reference_teacher_forward_time
                            reference_teacher_all_logps = reference_teacher_outputs["all_logps"]
                        negative_teacher_gated_logps = None
                        opdv_positive_sampled_logps = None
                        opdv_negative_sampled_logps = None
                        local_ranking_gate = model_inputs.get("ranking_gate") if use_crop_full_abs_rank else None
                        if use_crop_full_abs_rank:
                            if local_ranking_gate is None:
                                raise ValueError("crop_full_abs_contrast_rank requires ranking_gate")
                            # Keep FSDP collectives aligned across ranks: every actor
                            # micro-batch performs one rollout-level negative forward.
                            negative_batch = self._build_negative_rollout_batch(
                                model_inputs, self_distillation_cfg
                            )
                            if distillation_target_mode == "crop_only_v9_1":
                                if student_topk_indices is None:
                                    raise ValueError("crop_only_v9_1 requires student top-k indices for negative ranking")
                                with torch.no_grad():
                                    negative_outputs = self._forward_micro_batch(
                                        negative_batch,
                                        temperature=temperature,
                                        distillation_temperature=distillation_temperature,
                                        distill_topk=100,
                                        topk_indices=student_topk_indices,
                                        return_all_logps=False,
                                        module=teacher_model,
                                    )
                                negative_teacher_gated_logps = negative_outputs["topk_logps"][local_ranking_gate]
                                del negative_outputs
                            else:
                                with torch.no_grad():
                                    negative_outputs = self._forward_micro_batch(
                                        negative_batch,
                                        temperature=temperature,
                                        distillation_temperature=distillation_temperature,
                                        return_all_logps=True,
                                        all_logps_mask=local_ranking_gate,
                                        module=teacher_model,
                                    )
                                negative_teacher_gated_logps = negative_outputs["all_logps"]
                                if distillation_target_mode == "crop_only_v10_1":
                                    opdv_positive_sampled_logps = teacher_log_prob.detach()
                                    opdv_negative_sampled_logps = negative_outputs["log_probs"].detach()
                                del negative_outputs
                        if self_distillation_cfg.get("log_prob_dump_dir", None):
                            if distill_topk:
                                student_distill_log_probs = student_topk_logps
                                teacher_distill_log_probs = teacher_topk_logps
                                if self_distillation_cfg.distillation_add_tail:
                                    student_distill_log_probs = self._add_tail_bucket(student_distill_log_probs)
                                    teacher_distill_log_probs = self._add_tail_bucket(teacher_distill_log_probs)
                            else:
                                student_distill_log_probs = student_all_logps
                                teacher_distill_log_probs = teacher_all_logps

                            if student_distill_log_probs is None or teacher_distill_log_probs is None:
                                raise ValueError("Missing distillation log_probs for dump.")

                            loss_mask = response_mask
                            if self_distillation_mask is not None:
                                loss_mask = loss_mask * self_distillation_mask.unsqueeze(1)
                            if self_distillation_sample_filter is not None:
                                loss_mask = loss_mask * self_distillation_sample_filter.unsqueeze(1)
                            valid_rows = loss_mask > 0
                            if valid_rows.any():
                                dump_chunk = {
                                    "student_log_probs": student_distill_log_probs[valid_rows]
                                    .detach()
                                    .cpu()
                                    .to(torch.float32),
                                    "teacher_log_probs": teacher_distill_log_probs[valid_rows]
                                    .detach()
                                    .cpu()
                                    .to(torch.float32),
                                }
                                if reference_teacher_all_logps is not None:
                                    dump_chunk["reference_teacher_log_probs"] = (
                                        reference_teacher_all_logps[valid_rows]
                                        .detach()
                                        .cpu()
                                        .to(torch.float32)
                                    )
                                distill_dump_chunks.append(dump_chunk)
                        contrastive_excluded_token_ids = None
                        if (
                            use_crop_black_contrast
                            and self_distillation_cfg.get("contrastive_zero_termination_delta", True)
                        ):
                            contrastive_excluded_token_ids = self._termination_token_ids(teacher_model)
                        loss_compute_start = time.perf_counter()
                        vopd_loss, vopd_metrics = compute_self_distillation_loss(
                            student_log_probs=log_prob,
                            teacher_log_probs=teacher_log_prob,
                            response_mask=response_mask,
                            self_distillation_config=self_distillation_cfg,
                            old_log_probs=old_log_prob,
                            student_all_log_probs=student_all_logps,
                            teacher_all_log_probs=teacher_all_logps,
                            student_topk_log_probs=student_topk_logps,
                            teacher_topk_log_probs=teacher_topk_logps,
                            reference_teacher_all_log_probs=reference_teacher_all_logps,
                            negative_teacher_gated_log_probs=negative_teacher_gated_logps,
                            opdv_positive_sampled_log_probs=opdv_positive_sampled_logps,
                            opdv_negative_sampled_log_probs=opdv_negative_sampled_logps,
                            sampled_response_token_ids=(model_inputs["responses"] if local_ranking_gate is not None else None),
                            ranking_gate=local_ranking_gate,
                            ranking_batch_token_count=ranking_batch_token_count,
                            contrastive_excluded_token_ids=contrastive_excluded_token_ids,
                            self_distillation_mask=self_distillation_mask,
                            self_distillation_sample_filter=self_distillation_sample_filter,
                            self_distillation_wrong_mask=model_inputs.get("self_distillation_wrong_mask"),
                            inverse_advantage_weights=model_inputs.get("inverse_advantage_weights"),
                            loss_agg_mode=loss_agg_mode,
                            rollout_is_weights=rollout_is_weights,
                            batch_num_tokens=(
                                self.config.global_batch_info.get("batch_num_tokens")
                                or int(response_mask.sum().item())
                            ),
                            global_batch_size=self.config.global_batch_info.get("global_batch_size"),
                            loss_scale_factor=self.config.global_batch_info.get("loss_scale_factor"),
                        )
                        loss_compute_time = time.perf_counter() - loss_compute_start
                        stage_wall_time_totals["timing_s/update_actor/loss_compute"] += loss_compute_time

                        effective_distillation_mask = self_distillation_mask
                        if self_distillation_sample_filter is not None:
                            effective_distillation_mask = (
                                effective_distillation_mask
                                * self_distillation_sample_filter.to(
                                    device=effective_distillation_mask.device,
                                    dtype=effective_distillation_mask.dtype,
                                )
                            )
                        if distillation_target_mode == "crop_only_v9_8":
                            effective_distillation_mask = (
                                effective_distillation_mask
                                * model_inputs["self_distillation_wrong_mask"].to(
                                    device=effective_distillation_mask.device,
                                    dtype=effective_distillation_mask.dtype,
                                )
                            )
                        vopd_metrics["self_distillation/empty_target_batch"] = (
                            effective_distillation_mask.sum().item() == 0
                        )
                        micro_batch_metrics.update(vopd_metrics)

                        if (
                            policy_fallback_mask is not None
                            and policy_fallback_mask.any().item()
                            and grpo_loss_weight <= 0.0
                        ):
                            if advantages is None:
                                raise ValueError(
                                    "Mixed SDPO/GRPO fallback requires advantages for samples without teacher images."
                                )
                            policy_loss_fn = get_policy_loss_fn("vanilla")
                            grpo_loss, grpo_metrics = policy_loss_fn(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=advantages,
                                response_mask=response_mask * policy_fallback_mask.unsqueeze(1),
                                loss_agg_mode=loss_agg_mode,
                                config=self.config,
                                rollout_is_weights=rollout_is_weights,
                            )
                            pg_loss = vopd_loss + grpo_loss
                            micro_batch_metrics.update(
                                {f"actor/policy_fallback/{key.split('/', 1)[1]}": value for key, value in grpo_metrics.items()}
                            )
                        elif grpo_loss_weight > 0.0:
                            if advantages is None:
                                raise ValueError(
                                    "Additional GRPO loss requires advantages computed by the trainer."
                                )
                            grpo_response_mask = response_mask
                            if grpo_positive_advantage_gate:
                                response_token_count = response_mask.sum(dim=-1).clamp_min(1.0)
                                sample_advantage = (
                                    (advantages * response_mask).sum(dim=-1) / response_token_count
                                ).detach()
                                positive_sample_mask = (sample_advantage > 0).to(response_mask.dtype)
                                grpo_response_mask = grpo_response_mask * positive_sample_mask.unsqueeze(1)
                                micro_batch_metrics["actor/grpo_positive_advantage_fraction"] = (
                                    positive_sample_mask.float().mean().detach().item()
                                )
                            policy_loss_fn = get_policy_loss_fn("vanilla")
                            grpo_loss, grpo_metrics = policy_loss_fn(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=advantages,
                                response_mask=grpo_response_mask,
                                loss_agg_mode=loss_agg_mode,
                                config=self.config,
                                rollout_is_weights=rollout_is_weights,
                            )
                            pg_loss = vopd_loss + grpo_loss_weight * grpo_loss
                            micro_batch_metrics.update(
                                {f"actor/grpo/{key.split('/', 1)[1]}": value for key, value in grpo_metrics.items()}
                            )
                            micro_batch_metrics["actor/grpo_loss_weight"] = grpo_loss_weight
                        else:
                            grpo_loss = None
                            pg_loss = vopd_loss
                    else:
                        # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                        # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                        policy_loss_fn = get_policy_loss_fn(loss_mode)

                        # Compute policy loss (any function is expected to return 2 values)
                        pg_loss, pg_metrics = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_is_weights=rollout_is_weights,
                        )
                        micro_batch_metrics.update(pg_metrics)

                    # Skip if using bypass_mode loss (metrics already computed in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    policy_loss = pg_loss
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    backward_start = time.perf_counter()
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    if self_distillation_enabled:
                        backward_time = time.perf_counter() - backward_start
                        stage_wall_time_totals["timing_s/update_actor/backward"] += backward_time

                    metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                    if self_distillation_enabled:
                        metrics["actor/vopd_loss"] += vopd_loss.detach().item() * loss_scale_factor
                        metrics["actor/vopd_loss_weighted"] += vopd_loss.detach().item() * loss_scale_factor
                        if grpo_loss is not None:
                            metrics["actor/grpo_loss"] += grpo_loss.detach().item() * loss_scale_factor
                            if grpo_loss_weight > 0.0:
                                metrics["actor/grpo_loss_weighted"] += (
                                    (grpo_loss * grpo_loss_weight).detach().item() * loss_scale_factor
                                )
                    append_to_dict(metrics, micro_batch_metrics)

                optimizer_step_start = time.perf_counter()
                grad_norm = self._optimizer_step()
                if self_distillation_enabled:
                    optimizer_step_time = time.perf_counter() - optimizer_step_start
                if torch.isfinite(grad_norm).item():
                    did_update = True
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                if self_distillation_enabled:
                    stage_wall_time_totals["timing_s/update_actor/optimizer_step"] += optimizer_step_time
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        if self_distillation_enabled and distill_dump_chunks:
            self._dump_self_distillation_log_probs(
                meta_info=data.meta_info,
                self_distillation_cfg=self_distillation_cfg,
                dump_chunks=distill_dump_chunks,
            )
        if did_update and not is_v9_13_teacher_sft_batch:
            teacher_update_start = time.perf_counter()
            self._update_teacher()
            if self_distillation_enabled:
                stage_wall_time_totals["timing_s/update_actor/teacher_ema_update"] += (
                    time.perf_counter() - teacher_update_start
                )
        if self_distillation_enabled:
            for key, total_time in stage_wall_time_totals.items():
                metrics[key] = Metric(aggregation=AggregationType.MAX, value=total_time)
        metric_keys_to_keep_unreduced = set(stage_wall_time_totals.keys()) if stage_wall_time_totals is not None else set()
        local_metrics_to_reduce = {
            key: value
            for key, value in metrics.items()
            if isinstance(value, list) or (isinstance(value, Metric) and key not in metric_keys_to_keep_unreduced)
        }
        if local_metrics_to_reduce:
            reduced_local_metrics = reduce_metrics(local_metrics_to_reduce)
            metrics.update(reduced_local_metrics)
        return metrics
