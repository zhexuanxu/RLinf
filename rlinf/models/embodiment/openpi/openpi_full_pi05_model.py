# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Full Pi0.5 model with CoT text generation + action prediction.

Extends the existing OpenPi0ForRLActionPrediction (残血版 pi0.5) to support:
- VLM-only forward: Cross-entropy loss on language tokens (no actions)
- VLA-only forward: Flow matching loss on actions (same as 残血版)
- VLM+VLA forward: Dual-objective (CE + flow matching) for CoT + action training
- Autoregressive language generation with KV cache
- Think-then-act inference: generate reasoning → sample actions

Mode is auto-detected from data:
- actions=None, token_loss_mask.any()=True → VLM-only
- actions≠None, token_loss_mask.any()=False → VLA-only
- actions≠None, token_loss_mask.any()=True → VLM+VLA
"""

import math
from typing import Any, Literal

import torch
import torch.nn.functional as F
from openpi.models import model as _model
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.openpi.openpi_action_model import (
    OpenPi0Config,
    OpenPi0ForRLActionPrediction,
)
from rlinf.models.embodiment.openpi.static_kv_cache import (
    StaticKVCache,
    left_to_right_align,
)
from rlinf.utils.logging import get_logger
from rlinf.utils.nested_dict_process import copy_dict_tensor


class OpenPi05FullForRLActionPrediction(OpenPi0ForRLActionPrediction):
    """Full Pi0.5 with CoT text generation + action prediction.

    Inherits all RL infrastructure from OpenPi0ForRLActionPrediction
    (value head, noise head, DSRL, NFT, flow-SDE/ODE/CPS/noise methods)
    and adds:
    - lm_head for language token prediction
    - _compute_ce_loss for cross-entropy loss
    - _forward_vlm for VLM-only mode
    - _forward_vla_full for dual-objective (CE + flow matching)
    - generate_language for autoregressive generation
    - sample_with_reasoning for think-then-act inference
    """

    def __init__(self, config: OpenPi0Config):
        super().__init__(config)
        self.logger = get_logger()

        # lm_head already exists in PaliGemma — cache reference
        self.lm_head = self.paligemma_with_expert.paligemma.lm_head

        # EOS token id
        self._eos_token_id = self._detect_eos_token_id()

        # Vocab size for CE loss
        self._vocab_size = (
            self.paligemma_with_expert.paligemma.language_model.config.vocab_size
        )

        # Verification flags for one-time logging
        self._verified_kv_cache_mask = False
        self._verified_ce_loss = False

    def _detect_eos_token_id(self) -> int:
        """Get EOS token ID from PaliGemma config, falling back to config."""
        try:
            paligemma_eos = (
                self.paligemma_with_expert.paligemma.config.text_config.eos_token_id
            )
            if paligemma_eos is not None:
                return paligemma_eos
        except AttributeError:
            pass
        return self.config.eos_token_id

    # =========================================================================
    # Observation Processing
    # =========================================================================
    def _preprocess_observation(self, observation, *, train=True):
        """Override to use actual image keys instead of hardcoded defaults.

        The parent's preprocessing expects specific image keys (base_0_rgb, etc.)
        but VLM datasets use generic keys (image_0, image_1). Following openpi's
        pi0_fast.py pattern which uses list(observation.images.keys()).

        Also ensures images are [B, C, H, W] (channels-first) for SigLIP,
        since VLM datasets produce [B, H, W, C] (channels-last).
        """
        from openpi.models_pytorch import preprocessing_pytorch as _preprocessing

        observation = _preprocessing.preprocess_observation_pytorch(
            observation, train=train, image_keys=list(observation.images.keys())
        )
        # Ensure images are [B, C, H, W] for SigLIP vision encoder
        images = []
        for img in observation.images.values():
            if img.dim() == 4 and img.shape[1] != 3 and img.shape[-1] == 3:
                img = img.permute(0, 3, 1, 2)
            images.append(img)
        return (
            images,
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def _preprocess_observation_full(self, observation, *, train=True):
        """Extract observation components including three masks for full pi0.5.

        Returns the standard 5 components from parent plus three masks:
        - token_ar_mask: [B, L] int, 0=bidirectional, 1=causal
        - token_loss_mask: [B, L] bool, True=compute CE loss
        - token_kv_cache_mask: [B, L] bool, True=include in action expert KV cache
        """
        images, img_masks, lang_tokens, lang_masks, state = (
            self._preprocess_observation(observation, train=train)
        )

        batch_size, seq_len = lang_tokens.shape
        device = lang_tokens.device

        # token_ar_mask and token_loss_mask are in Observation dataclass
        token_ar_mask = observation.token_ar_mask
        token_loss_mask = observation.token_loss_mask

        # token_kv_cache_mask is NOT in Observation dataclass — use default
        token_kv_cache_mask = getattr(observation, "token_kv_cache_mask", None)

        # Apply defaults for missing masks
        if token_ar_mask is None:
            token_ar_mask = torch.zeros(
                batch_size, seq_len, dtype=torch.long, device=device
            )
        if token_loss_mask is None:
            token_loss_mask = torch.zeros(
                batch_size, seq_len, dtype=torch.bool, device=device
            )
        if token_kv_cache_mask is None:
            token_kv_cache_mask = observation.tokenized_prompt_mask.clone()

        return (
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            token_ar_mask,
            token_loss_mask,
            token_kv_cache_mask,
        )

    # =========================================================================
    # Embedding Methods
    # =========================================================================
    def embed_prefix_with_ar_mask(
        self, images, img_masks, lang_tokens, lang_masks, token_ar_mask
    ):
        """Embed prefix with per-token ar_mask for prefix-LM attention.

        Unlike parent's embed_prefix which uses all-zero ar_masks (bidirectional),
        this method uses token_ar_mask to support causal attention on CoT tokens.
        """
        embs, pad_masks, ar_masks = [], [], []
        bsize = lang_tokens.shape[0]
        device = lang_tokens.device

        # Embed images (ar_mask=0: bidirectional)
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            num_img_embs = img_emb.shape[1]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            ar_masks.append(
                torch.zeros(bsize, num_img_embs, dtype=torch.long, device=device)
            )

        # Embed language tokens with sqrt(dim) scaling
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            return lang_emb * math.sqrt(lang_emb.shape[-1])

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        ar_masks.append(token_ar_mask)

        return (
            torch.cat(embs, dim=1),
            torch.cat(pad_masks, dim=1),
            torch.cat(ar_masks, dim=1),
        )

    # =========================================================================
    # CE Loss
    # =========================================================================
    def _compute_ce_loss_from_logits(self, logits, lang_tokens, token_loss_mask):
        """Compute cross-entropy loss from pre-computed logits.

        Used by _forward_vlm where logits are computed inside the forward context
        to avoid FSDP FlatParameter view conflicts with lm_head.

        Args:
            logits: Pre-computed logits [B, N-1, vocab]
            lang_tokens: Language token ids [B, N]
            token_loss_mask: Which tokens to compute loss on [B, N]
        """
        target_ids = lang_tokens[:, 1:].long()
        loss_mask = token_loss_mask[:, 1:]
        denom = torch.clamp(loss_mask.sum(dim=-1).float(), min=1)

        ce_per_position = F.cross_entropy(
            logits.reshape(-1, self._vocab_size),
            target_ids.reshape(-1),
            reduction="none",
        ).view(target_ids.shape)
        ce_loss = (ce_per_position * loss_mask).sum(dim=-1) / denom

        pred_ids = logits.argmax(dim=-1)
        correct = (pred_ids == target_ids).float()
        language_acc = (correct * loss_mask).sum(dim=-1) / denom

        if not self._verified_ce_loss:
            self._verified_ce_loss = True
            self.logger.info(
                "[Full Pi0.5] CE loss computed: lang_len=%d, loss_mask_sum=%.1f",
                lang_tokens.shape[1],
                loss_mask.sum().item(),
            )

        return ce_loss, language_acc

    def _compute_ce_loss(
        self, prefix_out, prefix_pad_masks, lang_tokens, token_loss_mask,
        truncated_input=False,
    ):
        """Compute cross-entropy loss for language token prediction.

        For next-token prediction with input [img_0..img_M-1, tok_0..tok_N-1]:
        - Output at position i predicts token at position i+1
        - Need outputs at positions [M, M+1, ..., M+N-2] to predict lang_tokens[1:N]

        Args:
            prefix_out: Model output [B, seq_len, D]
            prefix_pad_masks: Padding masks [B, M+N] (original, not truncated)
            lang_tokens: Language token ids [B, N]
            token_loss_mask: Which tokens to compute loss on [B, N]
            truncated_input: True for VLM mode (last token removed from input)
        """
        num_img_tokens = prefix_pad_masks.shape[1] - lang_tokens.shape[1]

        if truncated_input:
            # VLM mode: input was [M+N-1], output is [M+N-1]
            lang_out = prefix_out[:, num_img_tokens:].clone()
        else:
            # VLA mode: input was [M+N], output is [M+N]
            lang_out = prefix_out[:, num_img_tokens:-1].clone()

        logits = self.lm_head(lang_out)  # [B, N-1, vocab]
        ce_loss, language_acc = self._compute_ce_loss_from_logits(
            logits, lang_tokens, token_loss_mask
        )

        return ce_loss, language_acc, logits

    # =========================================================================
    # Attention Masking
    # =========================================================================
    def _mask_suffix_with_kv_cache_mask(
        self, attn_mask, token_kv_cache_mask, prefix_pad_masks, suffix_pad_masks
    ):
        """Prevent action expert from attending to tokens excluded by kv_cache_mask.

        Uses token_kv_cache_mask (set during tokenization) to block suffix-to-excluded
        attention (e.g., EOS tokens). Ensures train/inference consistency.
        """
        batch_size = attn_mask.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        total_len = attn_mask.shape[1]
        device = attn_mask.device

        # Calculate image token count
        lang_len = token_kv_cache_mask.shape[1]
        num_img_tokens = prefix_len - lang_len

        # Identify excluded positions (where kv_cache_mask is False)
        excluded_in_lang = ~token_kv_cache_mask  # [B, lang_len]
        excluded_positions = torch.zeros(
            batch_size, total_len, dtype=torch.bool, device=device
        )
        excluded_positions[:, num_img_tokens:prefix_len] = excluded_in_lang

        # Identify suffix positions
        suffix_positions = torch.zeros(
            batch_size, total_len, dtype=torch.bool, device=device
        )
        suffix_positions[:, prefix_len:] = True

        # Block suffix-to-excluded attention
        suffix_to_excluded = suffix_positions[:, :, None] & excluded_positions[:, None, :]

        if not self._verified_kv_cache_mask:
            self._verified_kv_cache_mask = True
            n_excluded = excluded_in_lang.sum(dim=1).float()
            self.logger.info(
                "[Full Pi0.5] kv_cache_mask: excluding %.1f tokens on average from action expert",
                n_excluded.mean().item(),
            )

        return attn_mask & ~suffix_to_excluded

    # =========================================================================
    # Forward Methods
    # =========================================================================
    def _compute_velocity_with_prefix_out(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        token_ar_mask,
        token_kv_cache_mask,
        x_t,
        time,
        state,
    ):
        """Joint prefix+suffix forward returning both prefix_out and v_t.

        Unlike the parent's _build_prefix_cache + get_suffix_out pattern which
        caches prefix then runs suffix separately, this does a single combined
        forward to get both outputs for dual-objective training.
        """
        prefix_embs, prefix_pad_masks, prefix_ar_masks = (
            self.embed_prefix_with_ar_mask(
                images, img_masks, lang_tokens, lang_masks, token_ar_mask
            )
        )
        suffix_embs, suffix_pad_masks, suffix_ar_masks, adarms_cond = (
            self.embed_suffix(state, x_t, time)
        )

        # Cast to model dtype if needed
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0]
            .self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs = prefix_embs.to(torch.bfloat16)
            suffix_embs = suffix_embs.to(torch.bfloat16)

        # Build combined attention mask
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        ar_masks = torch.cat([prefix_ar_masks, suffix_ar_masks], dim=1)
        attn_mask = make_att_2d_masks(pad_masks, ar_masks)

        # Apply kv_cache_mask to block suffix-to-EOS attention
        attn_mask = self._mask_suffix_with_kv_cache_mask(
            attn_mask, token_kv_cache_mask, prefix_pad_masks, suffix_pad_masks
        )

        attn_mask_4d = self._prepare_attention_masks_4d(attn_mask)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Forward through model
        def forward_func(
            prefix_embs, suffix_embs, attn_mask_4d, position_ids, adarms_cond
        ):
            (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=attn_mask_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return prefix_out, suffix_out

        prefix_out, suffix_out = self._apply_checkpoint(
            forward_func,
            prefix_embs,
            suffix_embs,
            attn_mask_4d,
            position_ids,
            adarms_cond,
        )

        # Extract velocity
        action_out = suffix_out[:, -self.config.action_horizon :]
        action_out = action_out.to(self.action_out_proj.weight.dtype)
        v_t = self._apply_checkpoint(self.action_out_proj, action_out)

        return v_t, prefix_out, prefix_pad_masks

    def _forward_vlm(
        self, images, img_masks, lang_tokens, lang_masks, token_ar_mask, token_loss_mask
    ):
        """VLM-only forward: CE loss on language tokens, no action expert."""
        prefix_embs, prefix_pad_masks, prefix_ar_masks = (
            self.embed_prefix_with_ar_mask(
                images, img_masks, lang_tokens, lang_masks, token_ar_mask
            )
        )

        # For VLM-only, truncate last token (predicts next token)
        attn_mask = make_att_2d_masks(prefix_pad_masks, prefix_ar_masks)
        attn_mask_4d = self._prepare_attention_masks_4d(attn_mask[:, :-1, :-1])
        position_ids = torch.cumsum(prefix_pad_masks[:, :-1], dim=1) - 1

        # Cast to model dtype
        prefix_embs_input = prefix_embs[:, :-1]
        model_dtype = (
            self.paligemma_with_expert.paligemma.language_model.layers[0]
            .self_attn.q_proj.weight.dtype
        )
        if model_dtype == torch.bfloat16:
            prefix_embs_input = prefix_embs_input.to(torch.bfloat16)
            attn_mask_4d = attn_mask_4d.to(torch.bfloat16)

        # Forward through PaliGemma only (no suffix)
        def forward_func(prefix_embs, attn_mask_4d, position_ids):
            (prefix_out, _), _ = self.paligemma_with_expert.forward(
                attention_mask=attn_mask_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=False,
                adarms_cond=[None, None],
            )
            return prefix_out

        prefix_out = self._apply_checkpoint(
            forward_func, prefix_embs_input, attn_mask_4d, position_ids
        )

        ce_loss, language_acc, logits = self._compute_ce_loss(
            prefix_out, prefix_pad_masks, lang_tokens, token_loss_mask,
            truncated_input=True,
        )

        return {
            "loss": ce_loss.mean(),
            "action_loss": None,
            "language_loss": ce_loss.mean(),
            "language_token_acc": language_acc.mean(),
        }

    def _forward_vla_full(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        token_ar_mask,
        token_loss_mask,
        token_kv_cache_mask,
        actions,
        state,
        compute_ce_loss,
    ):
        """VLA or CoT+VLA forward with optional CE loss.

        Performs a single joint prefix+suffix forward to get both prefix_out
        (for CE loss) and suffix_out (for flow matching loss).
        """
        # Sample noise and time (same as parent flow matching)
        noise = self.sample_noise(actions.shape, actions.device)
        time = self.sample_time(actions.shape[0], actions.device)
        noise, time = noise.to(actions.dtype), time.to(actions.dtype)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Joint forward to get both prefix_out and v_t
        v_t, prefix_out, prefix_pad_masks = self._compute_velocity_with_prefix_out(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            token_ar_mask,
            token_kv_cache_mask,
            x_t,
            time,
            state,
        )

        # Flow matching loss
        flow_loss = F.mse_loss(u_t, v_t, reduction="none").mean(dim=(-1, -2))

        # Optional CE loss
        ce_loss, language_acc = None, None
        if compute_ce_loss:
            ce_loss, language_acc, _ = self._compute_ce_loss(
                prefix_out, prefix_pad_masks, lang_tokens, token_loss_mask
            )

        # Combine losses
        if ce_loss is not None:
            total_loss = (
                self.config.language_loss_weight * ce_loss
                + self.config.action_loss_weight * flow_loss
            )
        else:
            total_loss = flow_loss

        return {
            "loss": total_loss.mean(),
            "action_loss": flow_loss.mean(),
            "language_loss": ce_loss.mean() if ce_loss is not None else None,
            "language_token_acc": (
                language_acc.mean() if language_acc is not None else None
            ),
        }

    # =========================================================================
    # Forward Dispatch
    # =========================================================================
    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        elif forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        elif forward_type == ForwardType.NFT:
            return self.forward_nft(**kwargs)
        elif forward_type == ForwardType.SAC:
            return self.sac_forward(**kwargs)
        elif forward_type == ForwardType.SAC_Q:
            return self.sac_q_forward(**kwargs)
        elif forward_type == ForwardType.GENERATE_LANGUAGE:
            # Route language generation through the outer forward so FSDP's
            # lazy_init fires at the top level (not on a sub-module).
            return self.generate_language(**kwargs)
        else:
            raise NotImplementedError

    def sft_forward(self, data, **kwargs):
        """Unified SFT forward supporting VLM-only, VLA-only, and VLM+VLA modes.

        The config `forward_mode` determines data loading (in the worker):
        - "vlm" → Pi05VLMDataset (Robo2VLM), actions=None
        - "vla" → openpi LeRobot data loader, token_loss_mask all False
        - "vlm_vla" → LeRobot + CoT transform, both actions and loss_mask present

        Mode is then auto-detected from data content:
        - actions=None → _forward_vlm (CE loss only)
        - actions + loss_mask all False → _forward_vla_full (flow matching only)
        - actions + loss_mask has True → _forward_vla_full (CE + flow matching)
        """
        if hasattr(self, "gradient_checkpointing_disable"):
            self.gradient_checkpointing_disable()

        observation = data["observation"]
        actions = data.get("actions", None)

        # Extract token_kv_cache_mask from data dict (not in Observation dataclass)
        token_kv_cache_mask_override = data.get("token_kv_cache_mask", None)

        # Build Observation object and extract components
        if isinstance(observation, dict):
            obs_obj = _model.Observation.from_dict(observation)
        else:
            obs_obj = observation

        (
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            token_ar_mask,
            token_loss_mask,
            token_kv_cache_mask,
        ) = self._preprocess_observation_full(obs_obj)

        # Override kv_cache_mask if explicitly provided
        if token_kv_cache_mask_override is not None:
            device = lang_tokens.device
            token_kv_cache_mask = token_kv_cache_mask_override.to(device)

        has_ce_loss = token_loss_mask.any()
        has_actions = actions is not None

        if not has_actions:
            # VLM-only mode
            return self._forward_vlm(
                images, img_masks, lang_tokens, lang_masks,
                token_ar_mask, token_loss_mask,
            )

        # VLA-only or VLM+VLA mode
        return self._forward_vla_full(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            token_ar_mask,
            token_loss_mask,
            token_kv_cache_mask,
            actions,
            state,
            has_ce_loss,
        )

    # =========================================================================
    # Inference: Autoregressive Language Generation
    # =========================================================================
    @torch.no_grad()
    def generate_language(
        self, observation, max_new_tokens=50, temperature=0.0, eos_token_id=None
    ):
        """Generate language autoregressively (VLM mode).

        Uses StaticKVCache for efficient token-by-token generation.
        EOS token is stored in output but NOT added to KV cache, ensuring
        action expert won't attend to it during sample_with_reasoning.

        Returns:
            output_tokens: [B, actual_len] generated tokens (includes EOS if generated)
            past_kv: KV cache (excludes EOS, for action sampling)
            full_pad_mask: [B, prefix_len + tokens_in_cache] padding mask
            full_ar_mask: [B, prefix_len + tokens_in_cache] AR mask
        """
        if eos_token_id is None:
            eos_token_id = self._eos_token_id

        (
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            token_ar_mask,
            _,
            _,
        ) = self._preprocess_observation_full(observation, train=False)

        batch_size = lang_tokens.shape[0]
        device = lang_tokens.device

        prefix_embs, prefix_pad_masks, prefix_ar_masks = (
            self.embed_prefix_with_ar_mask(
                images, img_masks, lang_tokens, lang_masks, token_ar_mask
            )
        )

        # Left-to-right align for efficient decoding
        prefix_attn_2d = make_att_2d_masks(prefix_pad_masks, prefix_ar_masks)
        prefix_embs, prefix_pad_masks, prefix_attn_2d = left_to_right_align(
            prefix_embs, prefix_pad_masks, prefix_attn_2d
        )

        prefill_size = prefix_embs.shape[1]
        prefill_len = prefix_pad_masks.sum(dim=-1)  # [B]
        prefix_start = prefill_size - prefill_len  # [B]

        # Pad attention mask for decoding steps
        prefix_attn_2d = F.pad(prefix_attn_2d, (0, max_new_tokens, 0, 0))
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Cast to model dtype
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0]
            .self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs = prefix_embs.to(torch.bfloat16)

        # Initialize static KV cache
        paligemma_config = (
            self.paligemma_with_expert.paligemma.config.text_config
        )
        static_cache = StaticKVCache(
            max_batch_size=batch_size,
            max_cache_len=prefill_size + max_new_tokens,
            num_layers=paligemma_config.num_hidden_layers,
            num_key_value_heads=paligemma_config.num_key_value_heads,
            head_dim=paligemma_config.head_dim,
            dtype=prefix_embs.dtype,
            device=device,
        )

        # Prefill
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        (prefix_out, _), past_kv = self.paligemma_with_expert.forward(
            attention_mask=self._prepare_attention_masks_4d(prefix_attn_2d),
            position_ids=position_ids,
            past_key_values=static_cache,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            adarms_cond=[None, None],
        )

        # Decode loop
        last_logits = F.log_softmax(self.lm_head(prefix_out[:, -1:]), dim=-1)
        generated_tokens = []
        tokens_in_cache = 0

        for step in range(max_new_tokens):
            if temperature > 0:
                token = torch.multinomial(
                    torch.exp(last_logits[:, 0] / temperature), num_samples=1
                )
            else:
                token = last_logits[:, 0].argmax(dim=-1, keepdim=True)
            generated_tokens.append(token)

            # Stop before adding EOS to KV cache
            if torch.all(token == eos_token_id):
                break

            # Embed new token and update KV cache (only non-EOS)
            tokens_in_cache += 1
            token_emb = self.paligemma_with_expert.embed_language_tokens(token)
            token_emb = token_emb * math.sqrt(token_emb.shape[-1])
            if token_emb.dtype != prefix_embs.dtype:
                token_emb = token_emb.to(prefix_embs.dtype)
            step_position_ids = (prefill_len + step)[:, None]

            total_len = prefill_size + max_new_tokens
            j_indices = torch.arange(total_len, device=device)[None, None, :]
            mask = (j_indices >= prefix_start[:, None, None]) & (
                j_indices < (prefill_size + step + 1)
            )

            (prefix_out, _), past_kv = self.paligemma_with_expert.forward(
                attention_mask=self._prepare_attention_masks_4d(mask),
                position_ids=step_position_ids,
                past_key_values=past_kv,
                inputs_embeds=[token_emb, None],
                use_cache=True,
                adarms_cond=[None, None],
            )
            last_logits = F.log_softmax(self.lm_head(prefix_out[:, -1:]), dim=-1)

        assert len(generated_tokens) > 0, "No tokens generated"
        output_tokens = torch.cat(generated_tokens, dim=1)

        # Build masks for sequence in KV cache (prefix + non-EOS generated tokens)
        gen_pad_mask = torch.ones(
            (batch_size, tokens_in_cache), dtype=torch.bool, device=device
        )
        gen_ar_mask = torch.ones(
            (batch_size, tokens_in_cache), dtype=torch.long, device=device
        )
        full_pad_mask = torch.cat([prefix_pad_masks, gen_pad_mask], dim=1)
        full_ar_mask = torch.cat([prefix_ar_masks, gen_ar_mask], dim=1)

        return output_tokens, past_kv, full_pad_mask, full_ar_mask

    # =========================================================================
    # Inference: Denoising with KV Cache
    # =========================================================================
    def _euler_sample_with_cache(
        self, prefix_pad_masks, past_key_values, noise, device, num_steps
    ):
        """Run Euler integration for flow matching using pre-built KV cache."""
        bsize = noise.shape[0]
        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)

        while time >= -dt / 2:
            timestep = time.expand(bsize)
            v_t = self._denoise_step_with_cache(
                prefix_pad_masks, past_key_values, x_t, timestep
            )
            x_t = x_t + dt * v_t
            time = time + dt

        return x_t

    def _denoise_step_with_cache(
        self, prefix_pad_masks, past_key_values, x_t, timestep
    ):
        """Single denoising step using pre-built KV cache from language generation."""
        # Use parent's get_suffix_out, which handles suffix embedding and forward
        # with past_key_values. We pass a dummy state since pi0.5 ignores it.
        dummy_state = torch.zeros(
            x_t.shape[0], 1, dtype=x_t.dtype, device=x_t.device
        )
        suffix_out = self.get_suffix_out(
            dummy_state, prefix_pad_masks, past_key_values, x_t, timestep
        )
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        return self.action_out_proj(suffix_out)

    # =========================================================================
    # Inference: Think-then-Act
    # =========================================================================
    @torch.no_grad()
    def sample_with_reasoning(
        self,
        device,
        observation,
        noise=None,
        num_steps=10,
        max_reasoning_tokens=50,
        temperature=0.0,
        eos_token_id=None,
    ):
        """Generate reasoning text, then sample actions (VLM+VLA mode).

        1. Generate reasoning tokens autoregressively until EOS
        2. EOS is NOT in KV cache
        3. Use the EOS-excluded KV cache for action generation

        Returns:
            actions: [B, action_horizon, action_dim]
            generated_tokens: [B, actual_len] generated reasoning tokens
        """
        if eos_token_id is None:
            eos_token_id = self._eos_token_id

        # Stage 1: Generate reasoning tokens
        generated_tokens, past_kv, prefix_pad_masks, _ = self.generate_language(
            observation, max_reasoning_tokens, temperature, eos_token_id
        )

        # Stage 2: Sample actions using KV cache (excludes EOS)
        batch_size = generated_tokens.shape[0]
        if noise is None:
            noise = self.sample_noise(
                (batch_size, self.config.action_horizon, self.config.action_dim),
                device,
            )

        actions = self._euler_sample_with_cache(
            prefix_pad_masks, past_kv, noise, device, num_steps
        )

        return actions, generated_tokens

    # =========================================================================
    # RL: predict_action_batch override
    # =========================================================================
    def predict_action_batch(
        self,
        env_obs,
        mode: Literal["train", "eval"] = "train",
        compute_values=True,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Override for RL inference with CoT support.

        When forward_mode="vla": same as parent (残血版).
        When forward_mode="vlm_vla": generate CoT text then sample actions.
            PPO training data comes from parent's sample_actions (without CoT).
        """
        if self.config.forward_mode == "vla":
            return super().predict_action_batch(
                env_obs, mode, compute_values, **kwargs
            )

        # VLM+VLA: think-then-act
        to_process_obs = self.obs_processor(env_obs)
        processed_obs = self.input_transform(to_process_obs, transpose=False)
        processed_obs = self.precision_processor(processed_obs)
        observation = _model.Observation.from_dict(processed_obs)

        device = next(self.parameters()).device

        # Generate reasoning + sample actions (for env interaction)
        actions_raw, generated_tokens = self.sample_with_reasoning(
            device,
            observation,
            num_steps=self.config.num_steps,
            max_reasoning_tokens=self.config.max_language_len,
            temperature=self.config.language_temperature,
        )

        actions = self.output_transform(
            {"actions": actions_raw, "state": observation.state}
        )["actions"]

        # Build PPO training data using parent's sample_actions (without CoT)
        # CoT is inference-only; PPO trains on action chains without CoT context
        ppo_outputs = super().sample_actions(
            observation, mode=mode, compute_values=compute_values
        )

        forward_inputs = {
            "chains": ppo_outputs["chains"],
            "denoise_inds": ppo_outputs["denoise_inds"],
            "tokenized_prompt": processed_obs["tokenized_prompt"],
            "tokenized_prompt_mask": processed_obs["tokenized_prompt_mask"],
            "action": actions.reshape(actions.shape[0], -1).contiguous(),
            "model_action": actions_raw.reshape(actions_raw.shape[0], -1).contiguous(),
        }

        # Clone observations to avoid cross-step reference issues
        cloned_obs = copy_dict_tensor(
            {k: v for k, v in to_process_obs.items() if k != "prompt"}
        )
        forward_inputs.update(cloned_obs)

        result = {
            "prev_logprobs": ppo_outputs["prev_logprobs"],
            "prev_values": ppo_outputs["prev_values"],
            "forward_inputs": forward_inputs,
            "generated_tokens": generated_tokens,
        }
        return actions, result
