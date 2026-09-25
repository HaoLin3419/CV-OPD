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

from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING

from verl.base_config import BaseConfig
from verl.trainer.config import CheckpointConfig
from verl.utils.profiler.config import ProfilerConfig

from .engine import FSDPEngineConfig, McoreEngineConfig
from .model import HFModelConfig
from .optimizer import OptimizerConfig

__all__ = [
    "SelfDistillationConfig",
    "PolicyLossConfig",
    "RouterReplayConfig",
    "ActorConfig",
    "FSDPActorConfig",
    "McoreActorConfig",
]


@dataclass
class SelfDistillationConfig(BaseConfig):
    """Configuration for self-distillation loss.

    Args:
        Distillation is enabled when policy_loss.loss_mode is "vopd".
        full_logit_distillation (bool): Whether to use full-logit KL distillation.
        alpha (float): KL interpolation coefficient. 0.0=forward KL, 1.0=reverse KL, in-between=JSD.
        gamma (float): Weight applied to the SDPO loss.
        success_reward_threshold (float): Minimum sequence reward to be considered successful.
        teacher_regularization (str): Teacher regularization mode. Options: "ema", "trust-region", "progressive".
        teacher_update_rate (float): EMA update rate for teacher weights, or trust-region mixing coefficient.
        teacher_update_interval (Optional[int]): Hard-sync the teacher to the current student every N actor updates
            when teacher_regularization="progressive".
        distillation_topk (Optional[int]): If set, use top-k logits for distillation.
        distillation_add_tail (bool): Whether to add a tail bucket for top-k distillation.
        max_reprompt_len (int): Maximum length of the reprompted prompt.
        reprompt_truncation (str): Truncation method for the reprompted prompt (recommended to use "right" or "error").
        dont_reprompt_on_self_success (bool): Whether to not reprompt on self-success.
        remove_thinking_from_demonstration (bool): Whether to remove <think>...</think> tags from successful demonstrations before reprompting.
        is_clip (Optional[float]): Clip value for distillation IS ratio; None disables IS weighting.
        reprompt_template (str): Template for reprompting. Uses {prompt}, {solution}, {feedback} placeholders.
        solution_template (str): Template for formatting solution section. Uses {successful_previous_attempt} placeholder.
        feedback_template (str): Template for formatting feedback section. Uses {feedback_raw} placeholder.
        include_environment_feedback (bool): Whether to include environment feedback in reprompting for wrong attempts.
        environment_feedback_only_without_solution (bool): If True, only use feedback when no solution is available (ignore feedback when solution exists).
        reprompt_template_feedback (str): Template for reprompting with feedback but no solution.
        reprompt_template_feedback_solution (str): Template for reprompting with both feedback and solution.
        teacher_always_on (bool): Whether to distill every sample directly from a teacher input instead of selecting successful samples by reward.
        teacher_model_source (str): Teacher source. Options: "legacy", "current" or "fixed".
        teacher_model_path (Optional[str]): Fixed teacher model path when teacher_model_source="fixed".
        teacher_image_key (Optional[str]): Dataset column holding teacher-side images for multimodal distillation.
        teacher_input_mode (str): Teacher image composition mode. "crop_only" preserves the original
            Vision-OPD input, while "crop_plus_full" provides each full image followed by its crop.
        distillation_target_mode (str): Distillation target construction. "standard" preserves the existing
            Vision-OPD target, while "crop_black_contrast" applies VCSD-style full-vocabulary contrastive
            shaping from crop and same-size black-image teacher predictions.
        contrastive_alpha (float): Strength of crop-vs-black vocabulary-level contrastive shaping.
        contrastive_zero_termination_delta (bool): Whether to set the contrast value of EOS-like tokens to zero.
        distillation_temperature (Optional[float]): Temperature used for full-logit distillation distributions.
            None preserves the existing behavior by using the rollout temperature.
        contrastive_vocab_chunk_size (int): Vocabulary chunk size used to reduce peak memory while computing
            the exact full-vocabulary crop-black forward KL.
        ranking_margin (float): Margin for positive-vs-negative ranking.
        ranking_loss_weight (float): Weight of the auxiliary ranking loss.
        negative_crop_max_iou (float): Maximum allowed IoU with the positive crop box when metadata is available.
        negative_crop_min_scale (float): Minimum negative crop area fraction.
        negative_crop_max_scale (float): Maximum negative crop area fraction.
        negative_crop_seed (int): Base seed for deterministic per-rollout negative crops.
        negative_forward_chunk_size (int): Deprecated compatibility option; rollout-level negatives use one forward.
        ranking_token_selection (str): Token selection for the auxiliary ranking loss. ``batch_mean_jsd``
            preserves the v1-v3 gate; ``all_valid`` selects every valid response token; v6 uses the batch-mean gate with sampled-token ranking, and v7 disables ranking entirely.
        gate_forward_chunk_size (int): Number of samples per full/positive gate precomputation chunk.
        max_gated_tokens_per_batch (Optional[int]): Optional cap on gated tokens; None keeps all.
        fallback_to_policy_loss_on_missing_teacher (bool): When teacher_always_on=True, fall back to vanilla
            policy loss for samples whose teacher_image_key column is empty.
        log_prob_dump_dir (Optional[str]): Optional directory used to dump student/teacher log-prob tensors for each step.
    """

    full_logit_distillation: bool = True
    alpha: float = 0.0
    gamma: float = 1.0
    success_reward_threshold: float = 1.0
    teacher_regularization: str = "ema"
    teacher_update_rate: float = 0.05
    teacher_update_interval: Optional[int] = None
    distillation_topk: Optional[int] = None
    distillation_add_tail: bool = True
    max_reprompt_len: int = 10240
    reprompt_truncation: str = "right"
    dont_reprompt_on_self_success: bool = False
    remove_thinking_from_demonstration: bool = False
    is_clip: Optional[float] = None
    reprompt_template: str = (
        "{prompt}{solution}{feedback}\n\n"
        "Correctly solve the original question.\n"
    )
    solution_template: str = (
        "\n"
        "Correct solution:\n\n"
        "{successful_previous_attempt}\n\n"
    )
    feedback_template: str = (
        "\n"
        "The following is feedback from your unsuccessful earlier attempt:\n\n"
        "{feedback_raw}\n\n"
    )
    include_environment_feedback: bool = False
    environment_feedback_only_without_solution: bool = False
    teacher_always_on: bool = False
    teacher_model_source: str = "legacy"
    teacher_model_path: Optional[str] = None
    teacher_image_key: Optional[str] = None
    teacher_input_mode: str = "crop_only"
    distillation_target_mode: str = "standard"
    # v2-only image treatment; disabled for all legacy modes.
    crop_red_frame_enabled: bool = False
    crop_red_frame_width: int = 4
    contrastive_alpha: float = 1.0
    contrastive_zero_termination_delta: bool = True
    distillation_temperature: Optional[float] = None
    contrastive_vocab_chunk_size: int = 8192
    ranking_margin: float = 0.1
    ranking_loss_weight: float = 0.1
    negative_crop_max_iou: float = 0.1
    negative_crop_min_scale: float = 0.05
    negative_crop_max_scale: float = 0.5
    negative_crop_seed: int = 20260813
    negative_forward_chunk_size: int = 8
    ranking_token_selection: str = "batch_mean_jsd"
    gate_forward_chunk_size: int = 1
    max_gated_tokens_per_batch: Optional[int] = None
    teacher_prompt_mode: Optional[str] = None
    answer_hint_template: str = (
        "\n\nHere is a reference solution to this problem:\n"
        "{answer}\n\n"
        "After understanding the reference solution, please try to solve this problem using your own approach below:\n"
    )
    fallback_to_policy_loss_on_missing_teacher: bool = False
    log_prob_dump_dir: Optional[str] = None

    # Additional GRPO policy-loss weight for the v8-9/v8-10 distillation variants.
    # The default keeps all existing Vision-OPD modes unchanged.
    grpo_loss_weight: float = 0.0

    # When enabled, apply the additional GRPO term only to samples with positive outcome advantage.
    grpo_positive_advantage_gate: bool = False

    # Temperature for v9-9 group-normalized inverse GRPO-advantage weights.
    inverse_advantage_temperature: float = 1.0

    def __post_init__(self):
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"self_distillation.alpha must be in [0,1], got {self.alpha}")
        if self.gamma < 0.0:
            raise ValueError(f"self_distillation.gamma must be non-negative, got {self.gamma}")
        if self.grpo_loss_weight < 0.0:
            raise ValueError(f"self_distillation.grpo_loss_weight must be non-negative, got {self.grpo_loss_weight}")
        if self.inverse_advantage_temperature <= 0.0:
            raise ValueError(
                "self_distillation.inverse_advantage_temperature must be positive, "
                f"got {self.inverse_advantage_temperature}"
            )
        valid_teacher_regularization = ["ema", "trust-region", "progressive"]
        if self.teacher_regularization not in valid_teacher_regularization:
            raise ValueError(
                "self_distillation.teacher_regularization must be one of "
                f"{valid_teacher_regularization}, got {self.teacher_regularization}"
            )
        if not 0.0 <= self.teacher_update_rate <= 1.0:
            raise ValueError(
                f"self_distillation.teacher_update_rate must be in [0,1], got {self.teacher_update_rate}"
            )
        if self.teacher_update_interval is not None and self.teacher_update_interval <= 0:
            raise ValueError(
                "self_distillation.teacher_update_interval must be a positive integer "
                f"when set, got {self.teacher_update_interval}"
            )
        if self.distillation_topk is not None and self.distillation_topk <= 0:
            raise ValueError(
                f"self_distillation.distillation_topk must be a positive integer, got {self.distillation_topk}"
            )
        if self.is_clip is not None and self.is_clip <= 0:
            raise ValueError(f"self_distillation.is_clip must be positive, got {self.is_clip}")
        if self.teacher_prompt_mode is not None and self.teacher_prompt_mode != "answer_hint":
            raise ValueError(
                f"self_distillation.teacher_prompt_mode must be None or 'answer_hint', got {self.teacher_prompt_mode}"
            )
        valid_teacher_input_modes = ["crop_only", "crop_plus_full"]
        if self.teacher_input_mode not in valid_teacher_input_modes:
            raise ValueError(
                "self_distillation.teacher_input_mode must be one of "
                f"{valid_teacher_input_modes}, got {self.teacher_input_mode}"
            )
        valid_distillation_target_modes = [
            "standard",
            "crop_black_contrast",
            "crop_full_abs_contrast_rank",
            "crop_full_abs_contrast_rank_v2",
            "crop_full_abs_contrast_rank_v3",
            "crop_full_abs_contrast_rank_v4",
            "crop_full_abs_contrast_rank_v5",
            "crop_full_abs_contrast_rank_v6",
            "crop_full_abs_contrast_rank_v7",
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
            "crop_only_v9_12",
            "crop_only_v9_13",
            "crop_only_v9_14",
            "crop_only_v10_1",
            "crop_only_v10_2",
        ]
        if self.distillation_target_mode not in valid_distillation_target_modes:
            raise ValueError(
                "self_distillation.distillation_target_mode must be one of "
                f"{valid_distillation_target_modes}, got {self.distillation_target_mode}"
            )
        if self.crop_red_frame_width < 0:
            raise ValueError("self_distillation.crop_red_frame_width must be non-negative")
        if self.contrastive_alpha < 0.0:
            raise ValueError(
                "self_distillation.contrastive_alpha must be non-negative, "
                f"got {self.contrastive_alpha}"
            )
        if self.distillation_temperature is not None and self.distillation_temperature <= 0.0:
            raise ValueError(
                "self_distillation.distillation_temperature must be positive, "
                f"got {self.distillation_temperature}"
            )
        if self.contrastive_vocab_chunk_size <= 0:
            raise ValueError(
                "self_distillation.contrastive_vocab_chunk_size must be positive, "
                f"got {self.contrastive_vocab_chunk_size}"
            )
        if self.ranking_margin < 0.0:
            raise ValueError(f"self_distillation.ranking_margin must be non-negative, got {self.ranking_margin}")
        if self.ranking_loss_weight < 0.0:
            raise ValueError(f"self_distillation.ranking_loss_weight must be non-negative, got {self.ranking_loss_weight}")
        if not 0.0 <= self.negative_crop_max_iou <= 1.0:
            raise ValueError(f"self_distillation.negative_crop_max_iou must be in [0,1], got {self.negative_crop_max_iou}")
        if not 0.0 < self.negative_crop_min_scale <= self.negative_crop_max_scale <= 1.0:
            raise ValueError("negative crop scales must satisfy 0 < min <= max <= 1")
        if self.negative_forward_chunk_size <= 0:
            raise ValueError("self_distillation.negative_forward_chunk_size must be positive")
        valid_ranking_token_selections = ["batch_mean_jsd", "all_valid", "multi_negative_probability"]
        if self.ranking_token_selection not in valid_ranking_token_selections:
            raise ValueError(
                "self_distillation.ranking_token_selection must be one of "
                f"{valid_ranking_token_selections}, got {self.ranking_token_selection}"
            )
        if self.gate_forward_chunk_size <= 0:
            raise ValueError("self_distillation.gate_forward_chunk_size must be positive")
        if self.max_gated_tokens_per_batch is not None and self.max_gated_tokens_per_batch <= 0:
            raise ValueError("self_distillation.max_gated_tokens_per_batch must be positive when set")
        if self.distillation_target_mode == "crop_black_contrast":
            if not self.teacher_always_on or not self.teacher_image_key:
                raise ValueError(
                    "self_distillation.distillation_target_mode='crop_black_contrast' requires "
                    "teacher_always_on=True and teacher_image_key"
                )
            if self.teacher_prompt_mode is not None:
                raise ValueError(
                    "self_distillation.distillation_target_mode='crop_black_contrast' requires "
                    "teacher_prompt_mode=None"
                )
            if self.teacher_input_mode != "crop_only":
                raise ValueError(
                    "self_distillation.distillation_target_mode='crop_black_contrast' requires "
                    "teacher_input_mode='crop_only'"
                )
            if not self.full_logit_distillation:
                raise ValueError(
                    "self_distillation.distillation_target_mode='crop_black_contrast' requires "
                    "full_logit_distillation=True"
                )
            if self.distillation_topk is not None:
                raise ValueError(
                    "self_distillation.distillation_target_mode='crop_black_contrast' requires "
                    "distillation_topk=None"
                )
            if self.alpha != 0.0:
                raise ValueError(
                    "self_distillation.distillation_target_mode='crop_black_contrast' uses forward KL and "
                    "requires alpha=0.0"
                )
            if self.distillation_temperature is None:
                raise ValueError(
                    "self_distillation.distillation_target_mode='crop_black_contrast' requires "
                    "distillation_temperature to be set"
                )
        if self.distillation_target_mode in {
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
            "crop_only_v9_7",
            "crop_only_v9_8",
            "crop_only_v9_9",
            "crop_only_v9_15",
            "crop_only_v9_10",
            "crop_only_v9_11",
            "crop_only_v9_12",
            "crop_only_v9_13",
            "crop_only_v9_14",
            "crop_only_v10_1",
            "crop_only_v10_2",
        }:
            mode_name = self.distillation_target_mode
            if not self.teacher_always_on or not self.teacher_image_key:
                raise ValueError(f"{mode_name} requires teacher_always_on=True and teacher_image_key")
            if self.teacher_prompt_mode is not None or self.teacher_input_mode != "crop_only":
                raise ValueError(f"{mode_name} requires teacher_prompt_mode=None and teacher_input_mode=crop_only")
            if not self.full_logit_distillation or self.distillation_topk != 100 or not self.distillation_add_tail:
                raise ValueError(f"{mode_name} requires full_logit_distillation=True, distillation_topk=100, and distillation_add_tail=True")
            if self.alpha != 0.5 or self.distillation_temperature != 1.0 or self.is_clip != 2.0:
                raise ValueError(f"{mode_name} preserves v8 main loss and requires alpha=0.5, distillation_temperature=1.0, and is_clip=2.0")
            if self.teacher_model_source != "legacy" or self.teacher_regularization != "ema":
                raise ValueError(f"{mode_name} requires legacy EMA teacher")
            expected_ranking_selection = (
                "multi_negative_probability" if mode_name == "crop_only_v8_8"
                else (
                    "all_valid" if mode_name in {"crop_only_v8_3", "crop_only_v8_4", "crop_only_v8_5", "crop_only_v8_6", "crop_only_v8_7", "crop_only_v8_3_2", "crop_only_v8_4_2", "crop_only_v8_5_2", "crop_only_v8_6_2"} else "batch_mean_jsd"
                )
            )
            if self.ranking_token_selection != expected_ranking_selection:
                raise ValueError(
                    f"{mode_name} requires ranking_token_selection={expected_ranking_selection}"
                )
            if mode_name in {"crop_only_v8_9", "crop_only_v8_10", "crop_only_v9_2"}:
                if self.grpo_loss_weight != 0.5:
                    raise ValueError(f"{mode_name} requires grpo_loss_weight=0.5")
                expected_gate = mode_name == "crop_only_v8_10"
                if self.grpo_positive_advantage_gate != expected_gate:
                    raise ValueError(
                        f"{mode_name} requires grpo_positive_advantage_gate={expected_gate}"
                    )
            if mode_name == "crop_only_v9_8":
                if self.grpo_loss_weight != 0.0 or self.grpo_positive_advantage_gate:
                    raise ValueError(
                        "crop_only_v9_8 only uses reward for trajectory filtering and requires "
                        "grpo_loss_weight=0.0 and grpo_positive_advantage_gate=False"
                    )
            if mode_name == "crop_only_v9_9":
                if self.grpo_loss_weight != 0.0 or self.grpo_positive_advantage_gate:
                    raise ValueError(
                        "crop_only_v9_9 uses GRPO advantage only for main-distillation weighting and requires "
                        "grpo_loss_weight=0.0 and grpo_positive_advantage_gate=False"
                    )
                if self.inverse_advantage_temperature != 1.0:
                    raise ValueError("crop_only_v9_9 requires inverse_advantage_temperature=1.0")
            if mode_name == "crop_only_v9_15":
                if self.grpo_loss_weight != 0.0 or self.grpo_positive_advantage_gate:
                    raise ValueError(
                        "crop_only_v9_15 uses positive GRPO advantage only for main-distillation weighting and requires "
                        "grpo_loss_weight=0.0 and grpo_positive_advantage_gate=False"
                    )
                if self.inverse_advantage_temperature != 1.0:
                    raise ValueError("crop_only_v9_15 requires inverse_advantage_temperature=1.0")
            if mode_name == "crop_only_v9_10":
                if self.grpo_loss_weight != 0.0 or self.grpo_positive_advantage_gate:
                    raise ValueError(
                        "crop_only_v9_10 preserves v8-1 and requires "
                        "grpo_loss_weight=0.0 and grpo_positive_advantage_gate=False"
                    )
            if mode_name == "crop_only_v9_11":
                if self.grpo_loss_weight != 0.0 or self.grpo_positive_advantage_gate:
                    raise ValueError(
                        "crop_only_v9_11 preserves v9-10 and requires "
                        "grpo_loss_weight=0.0 and grpo_positive_advantage_gate=False"
                    )
            if mode_name == "crop_only_v10_2":
                if self.grpo_loss_weight != 0.0 or self.grpo_positive_advantage_gate:
                    raise ValueError(
                        "crop_only_v10_2 preserves v9-11 losses and requires "
                        "grpo_loss_weight=0.0 and grpo_positive_advantage_gate=False"
                    )
            if mode_name == "crop_only_v9_12":
                if self.grpo_loss_weight != 0.0 or self.grpo_positive_advantage_gate:
                    raise ValueError(
                        "crop_only_v9_12 preserves v9-11 with the v8-1 original-teacher gate and requires "
                        "grpo_loss_weight=0.0 and grpo_positive_advantage_gate=False"
                    )
            if mode_name == "crop_only_v9_13":
                if self.grpo_loss_weight != 0.0 or self.grpo_positive_advantage_gate:
                    raise ValueError(
                        "crop_only_v9_13 preserves v9-11 for successful UIDs and uses crop-teacher SFT for all-wrong UIDs; "
                        "requires grpo_loss_weight=0.0 and grpo_positive_advantage_gate=False"
                    )
            if mode_name == "crop_only_v9_14":
                if self.grpo_loss_weight != 0.0 or self.grpo_positive_advantage_gate:
                    raise ValueError(
                        "crop_only_v9_14 preserves v9-13 losses and uses trajectory-level correct distillation / "
                        "wrong crop-teacher SFT; requires grpo_loss_weight=0.0 and "
                        "grpo_positive_advantage_gate=False"
                    )
            if mode_name == "crop_only_v10_1":
                if self.grpo_loss_weight != 0.0 or self.grpo_positive_advantage_gate:
                    raise ValueError(
                        "crop_only_v10_1 preserves v8-1 policy/auxiliary losses and requires "
                        "grpo_loss_weight=0.0 and grpo_positive_advantage_gate=False"
                    )
        if self.distillation_target_mode == "crop_only_v9_6":
            if not self.teacher_always_on or not self.teacher_image_key:
                raise ValueError("crop_only_v9_6 requires teacher_always_on=True and teacher_image_key")
            if self.teacher_prompt_mode is not None or self.teacher_input_mode != "crop_only":
                raise ValueError("crop_only_v9_6 requires teacher_prompt_mode=None and teacher_input_mode=crop_only")
            if not self.full_logit_distillation or self.distillation_topk is not None or self.distillation_add_tail:
                raise ValueError("crop_only_v9_6 requires full_logit_distillation=True, distillation_topk=None, and distillation_add_tail=False")
            if self.alpha != 0.5 or self.distillation_temperature != 1.0 or self.is_clip != 2.0:
                raise ValueError("crop_only_v9_6 preserves v8-1 main loss and requires alpha=0.5, distillation_temperature=1.0, and is_clip=2.0")
            if self.teacher_model_source != "legacy" or self.teacher_regularization != "ema":
                raise ValueError("crop_only_v9_6 requires legacy EMA teacher")
            if self.ranking_token_selection != "batch_mean_jsd":
                raise ValueError("crop_only_v9_6 requires ranking_token_selection=batch_mean_jsd")
        if self.distillation_target_mode in {
            "crop_full_abs_contrast_rank",
            "crop_full_abs_contrast_rank_v2",
            "crop_full_abs_contrast_rank_v3",
            "crop_full_abs_contrast_rank_v4",
            "crop_full_abs_contrast_rank_v5",
            "crop_full_abs_contrast_rank_v6",
            "crop_full_abs_contrast_rank_v7",
        }:
            mode_name = self.distillation_target_mode
            if not self.teacher_always_on or not self.teacher_image_key:
                raise ValueError(f"{mode_name} requires teacher_always_on=True and teacher_image_key")
            if self.teacher_prompt_mode is not None or self.teacher_input_mode != "crop_only":
                raise ValueError(f"{mode_name} requires teacher_prompt_mode=None and teacher_input_mode='crop_only'")
            if not self.full_logit_distillation or self.distillation_topk is not None:
                raise ValueError(f"{mode_name} requires full_logit_distillation=True and distillation_topk=None")
            if self.alpha != 0.0 or self.distillation_temperature is None:
                raise ValueError(f"{mode_name} requires alpha=0.0 and distillation_temperature")
            if mode_name in {"crop_full_abs_contrast_rank", "crop_full_abs_contrast_rank_v2"}:
                if self.teacher_model_source != "fixed":
                    raise ValueError(f"{mode_name} requires teacher_model_source=fixed")
            if mode_name in {"crop_full_abs_contrast_rank_v3", "crop_full_abs_contrast_rank_v4", "crop_full_abs_contrast_rank_v5", "crop_full_abs_contrast_rank_v6", "crop_full_abs_contrast_rank_v7"}:
                if self.teacher_model_source != "legacy":
                    raise ValueError(f"{mode_name} requires teacher_model_source=legacy")
                if self.teacher_regularization != "ema":
                    raise ValueError(f"{mode_name} requires teacher_regularization=ema")
            if mode_name in {"crop_full_abs_contrast_rank_v4", "crop_full_abs_contrast_rank_v5"} and self.ranking_token_selection != "all_valid":
                raise ValueError(f"{mode_name} requires ranking_token_selection=all_valid")
            if mode_name == "crop_full_abs_contrast_rank_v6" and self.ranking_token_selection != "batch_mean_jsd":
                raise ValueError(f"{mode_name} requires ranking_token_selection=batch_mean_jsd")
        if self.teacher_prompt_mode == "answer_hint" and self.teacher_input_mode != "crop_only":
            raise ValueError(
                "self_distillation.teacher_input_mode only applies to image-conditioned teachers; "
                "use crop_only when teacher_prompt_mode='answer_hint'"
            )
        if self.teacher_always_on and not self.teacher_image_key and self.teacher_prompt_mode != "answer_hint":
            raise ValueError(
                "self_distillation.teacher_image_key is required when teacher_always_on=True "
                "(unless teacher_prompt_mode='answer_hint')"
            )
        valid_teacher_model_source = ["legacy", "current", "fixed"]
        if self.teacher_model_source not in valid_teacher_model_source:
            raise ValueError(
                "self_distillation.teacher_model_source must be one of "
                f"{valid_teacher_model_source}, got {self.teacher_model_source}"
            )
        if self.teacher_model_source == "fixed" and not self.teacher_model_path:
            raise ValueError("self_distillation.teacher_model_path is required when teacher_model_source='fixed'")
        if self.teacher_regularization == "progressive":
            if self.teacher_model_source != "legacy":
                raise ValueError(
                    "self_distillation.teacher_regularization='progressive' requires "
                    "teacher_model_source='legacy'"
                )
            if self.teacher_update_interval is None:
                raise ValueError(
                    "self_distillation.teacher_update_interval is required when "
                    "teacher_regularization='progressive'"
                )


@dataclass
class RouterReplayConfig(BaseConfig):
    """Configuration for router replay in MoE models.

    This configuration controls the routing behavior for Mixture of Experts (MoE) models,
    allowing for deterministic training through route recording and replay.

    Args:
        mode (str): Router replay mode. Options: 'disabled', 'R2', 'R3'.
            - 'disabled': No router replay functionality
            - 'R2': Use Router Replay routing strategy
            - 'R3': Use Rollout Router Replay routing strategy
        record_file (Optional[str]): File path to save recorded routing decisions.
            Required when mode is 'record', 'R2', or 'R3'.
        replay_file (Optional[str]): File path to load recorded routing decisions for replay.
            Required when mode is 'replay'.
    """

    mode: str = "disabled"
    record_file: Optional[str] = None
    replay_file: Optional[str] = None

    def __post_init__(self):
        """Validate router replay configuration."""
        valid_modes = ["disabled", "R2", "R3"]
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid router_replay mode: {self.mode}. Must be one of {valid_modes}")


@dataclass
class PolicyLossConfig(BaseConfig):
    """Configuration for policy loss computation.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        loss_mode (str): Loss function mode. Options: 'vanilla', 'clip-cov', 'kl-cov', 'gpg', 'vopd'.
        clip_cov_ratio (float): Ratio of tokens to be clipped for clip-cov loss.
        clip_cov_lb (float): Lower bound for clip-cov loss.
        clip_cov_ub (float): Upper bound for clip-cov loss.
        kl_cov_ratio (float): Ratio of tokens to be applied KL penalty for kl-cov loss.
        ppo_kl_coef (float): KL divergence penalty coefficient.
    """

    loss_mode: str = "vanilla"
    clip_cov_ratio: float = 0.0002
    clip_cov_lb: float = 1.0
    clip_cov_ub: float = 5.0
    kl_cov_ratio: float = 0.0002
    ppo_kl_coef: float = 0.1


@dataclass
class ActorConfig(BaseConfig):
    """Configuration for actor model training.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy. Must be specified.
        ppo_mini_batch_size (int): Mini-batch size for PPO training.
        ppo_micro_batch_size (Optional[int]): Micro-batch size for PPO training.
            If None, uses ppo_micro_batch_size_per_gpu.
        ppo_micro_batch_size_per_gpu (Optional[int]): Micro-batch size per GPU for PPO training.
        use_dynamic_bsz (bool): Whether to use dynamic batch sizing.
        ppo_max_token_len_per_gpu (int): Maximum token length per GPU for PPO training.
        clip_ratio (float): PPO clipping ratio for policy loss.
        clip_ratio_low (float): Lower bound for PPO clipping ratio.
        clip_ratio_high (float): Upper bound for PPO clipping ratio.
        policy_loss (PolicyLossConfig): Configuration for policy loss computation.
        clip_ratio_c (float): Clipping ratio for critic loss.
        loss_agg_mode (str): Loss aggregation mode. Options: 'token-mean', 'sample-mean'.
        loss_scale_factor (Optional[int]): Scale factor for 'seq-mean-token-sum-norm' loss aggregation mode.
            If None, uses response_length. Set to a constant to ensure consistent normalization.
        entropy_coeff (float): Entropy coefficient for regularization.
        tau_pos (float): Positive tau for SAPO smoothing (>= 1.0 keeps rewards stable).
        tau_neg (float): Negative tau for SAPO smoothing (> tau_pos for asymmetry).
        use_kl_loss (bool): Whether to use KL divergence loss.
        use_torch_compile (bool): Whether to use torch.compile for optimization.
        kl_loss_coef (float): KL divergence loss coefficient.
        kl_loss_type (str): Type of KL loss to use.
        ppo_epochs (int): Number of PPO epochs per training step.
        shuffle (bool): Whether to shuffle data during training.
        checkpoint (CheckpointConfig): Configuration for checkpointing.
        optim (OptimizerConfig): Configuration for optimizer.
        use_fused_kernels (bool): Whether to use custom fused kernels (e.g., FlashAttention, fused MLP).
        data_loader_seed (int): Seed for data loader. If None, uses global seed.
        router_replay (RouterReplayConfig): Configuration for router replay in MoE models.
    """

    _mutable_fields = BaseConfig._mutable_fields | {
        "ppo_mini_batch_size",
        "ppo_micro_batch_size",
        "ppo_micro_batch_size_per_gpu",
        "ppo_infer_micro_batch_size_per_gpu",
        "engine",
        "model_config",
    }

    strategy: str = MISSING
    ppo_mini_batch_size: int = 256
    ppo_micro_batch_size: Optional[int] = None  # deprecate
    ppo_micro_batch_size_per_gpu: Optional[int] = None
    ppo_infer_micro_batch_size_per_gpu: Optional[int] = None
    use_dynamic_bsz: bool = False
    ppo_max_token_len_per_gpu: int = 16384
    ppo_infer_max_token_len_per_gpu: int = 16384
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2
    freeze_vision_tower: bool = False
    policy_loss: PolicyLossConfig = field(default_factory=PolicyLossConfig)
    clip_ratio_c: float = 3.0
    loss_agg_mode: str = "token-mean"
    loss_scale_factor: Optional[int] = None
    entropy_coeff: float = 0
    tau_pos: float = 1.0
    tau_neg: float = 1.05
    calculate_entropy: bool = False
    use_kl_loss: bool = False
    # Whether to enable PrefixGrouper-based shared-prefix forward
    use_prefix_grouper: bool = False
    use_torch_compile: bool = True
    kl_loss_coef: float = 0.001
    kl_loss_type: str = "low_var_kl"
    ppo_epochs: int = 1
    shuffle: bool = False
    data_loader_seed: int = 1
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    use_fused_kernels: bool = False
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    engine: BaseConfig = field(default_factory=BaseConfig)
    rollout_n: int = MISSING  # must be override by sampling config
    model_config: HFModelConfig = field(default_factory=BaseConfig)
    router_replay: RouterReplayConfig = field(default_factory=RouterReplayConfig)
    self_distillation: SelfDistillationConfig = field(default_factory=SelfDistillationConfig)

    # Store global batch info for loss aggregation:
    # dp_size: data parallel size
    # batch_num_tokens: number of valid tokens in global batch
    # global_batch_size: global batch size
    global_batch_info: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate actor configuration parameters."""
        assert self.strategy != MISSING
        assert self.rollout_n != MISSING
        if not self.use_dynamic_bsz:
            if self.ppo_micro_batch_size is not None and self.ppo_micro_batch_size_per_gpu is not None:
                raise ValueError(
                    "[actor] You have set both 'actor.ppo_micro_batch_size' AND 'actor.ppo_micro_batch_size_per_gpu'. "
                    "Please remove 'actor.ppo_micro_batch_size' because only '*_ppo_micro_batch_size_per_gpu' is "
                    "supported (the former is deprecated)."
                )
            else:
                assert not (self.ppo_micro_batch_size is None and self.ppo_micro_batch_size_per_gpu is None), (
                    "[actor] Please set at least one of 'actor.ppo_micro_batch_size' or "
                    "'actor.ppo_micro_batch_size_per_gpu' if use_dynamic_bsz is not enabled."
                )

        valid_loss_agg_modes = [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ]
        if self.loss_agg_mode not in valid_loss_agg_modes:
            raise ValueError(f"Invalid loss_agg_mode: {self.loss_agg_mode}")

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        """Validate actor configuration with runtime parameters."""
        if not self.use_dynamic_bsz:
            if train_batch_size < self.ppo_mini_batch_size:
                raise ValueError(
                    f"train_batch_size ({train_batch_size}) must be >= "
                    f"actor.ppo_mini_batch_size ({self.ppo_mini_batch_size})"
                )

            sp_size = getattr(self, "ulysses_sequence_parallel_size", 1)
            if self.ppo_micro_batch_size is not None:
                if self.ppo_mini_batch_size % self.ppo_micro_batch_size != 0:
                    raise ValueError(
                        f"ppo_mini_batch_size ({self.ppo_mini_batch_size}) must be divisible by "
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size})"
                    )
                if self.ppo_micro_batch_size * sp_size < n_gpus:
                    raise ValueError(
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size}) * "
                        f"ulysses_sequence_parallel_size ({sp_size}) must be >= n_gpus ({n_gpus})"
                    )

    @staticmethod
    def _check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        """Validate mutually exclusive micro batch size configuration options."""
        param = "ppo_micro_batch_size"
        param_per_gpu = f"{param}_per_gpu"

        if mbs is None and mbs_per_gpu is None:
            raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

        if mbs is not None and mbs_per_gpu is not None:
            raise ValueError(
                f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
            )


@dataclass
class McoreActorConfig(ActorConfig):
    """Configuration for Megatron actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'megatron' for Megatron parallelism.
        load_weight (bool): Whether to load model weights from checkpoint.
        megatron (dict[str, Any]): Configuration for Megatron parallelism settings.
        profile (dict[str, Any]): Configuration for profiling settings.
    """

    strategy: str = "megatron"
    load_weight: bool = True
    megatron: McoreEngineConfig = field(default_factory=McoreEngineConfig)
    profile: dict[str, Any] = field(default_factory=dict)
    use_rollout_log_probs: bool = False

    def __post_init__(self):
        """Validate FSDP actor configuration parameters."""
        super().__post_init__()
        self.engine = self.megatron


@dataclass
class FSDPActorConfig(ActorConfig):
    """Configuration for FSDP actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'fsdp' for Fully Sharded Data Parallel.
        grad_clip (float): Gradient clipping threshold.
        ulysses_sequence_parallel_size (int): [DEPRECATED] Ulysses sequence parallel size for long sequences.
        entropy_from_logits_with_chunking (bool): Whether to compute entropy from logits
            with chunking for memory efficiency.
        entropy_checkpointing (bool): Whether to use gradient checkpointing for entropy computation.
        fsdp_config (dict[str, Any]): Configuration for FSDP settings.
        use_remove_padding (bool): Whether to remove padding tokens in inputs during training
    """

    strategy: str = "fsdp"
    grad_clip: float = 1.0
    ulysses_sequence_parallel_size: int = 1
    entropy_from_logits_with_chunking: bool = False
    entropy_checkpointing: bool = False
    fsdp_config: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)
    use_remove_padding: bool = False
    use_rollout_log_probs: bool = False
    calculate_sum_pi_squared: bool = False
    sum_pi_squared_checkpointing: bool = False

    def __post_init__(self):
        """Validate FSDP actor configuration parameters."""
        super().__post_init__()
        self.engine = self.fsdp_config

        # backward compatibility
        if self.ulysses_sequence_parallel_size > 1:
            self.fsdp_config.ulysses_sequence_parallel_size = self.ulysses_sequence_parallel_size

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        """Validate FSDP actor configuration with runtime parameters."""
        super().validate(n_gpus, train_batch_size, model_config)

        if self.strategy in {"fsdp", "fsdp2"} and self.ulysses_sequence_parallel_size > 1:
            if model_config and not model_config.get("use_remove_padding", False):
                raise ValueError(
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
                )
