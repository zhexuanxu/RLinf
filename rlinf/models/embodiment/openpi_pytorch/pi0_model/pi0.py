# Copyright (c) 2025, RLinf contributors.
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

"""Pi0 model for PyTorch, aligned with JAX models/pi0.py.

Flow matching model for continuous action generation.
"""

from __future__ import annotations

import logging

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import gemma, model, pointnet, siglip
from .pi0_config import Pi0Config
from .static_kv_cache import StaticKVCache, left_to_right_align
from .utils import _str_to_dtype

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask: torch.Tensor, mask_ar: torch.Tensor) -> torch.Tensor:
    """Create attention mask from input mask and autoregressive mask.

    Tokens can attend to valid input tokens which have a cumulative mask_ar
    smaller or equal to theirs.

    Args:
        input_mask: bool[B, N] - true if token is valid
        mask_ar: bool[N] or bool[B, N] - true where next token starts a new
            autoregressive block
    """
    mask_ar = mask_ar.expand(input_mask.shape[0], -1)
    cumsum = torch.cumsum(mask_ar.int(), dim=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return torch.logical_and(attn_mask, valid_mask)


def block_suffix_from_excluded_prefix(
    attn_mask: torch.Tensor,
    token_kv_cache_mask: torch.Tensor,
    prefix_len: int,
) -> torch.Tensor:
    """Hide KV-excluded text positions (e.g. EOS) from suffix (action) queries.

    The per-token KV-cache mask marks which TEXT tokens the action expert may
    attend to. Text tokens are the last ``token_kv_cache_mask.shape[1]``
    positions of the prefix; suffix queries are the rows at and after
    ``prefix_len``. Prefix self-attention is left untouched, keeping training
    attention identical to the generation-time view where excluded tokens never
    enter the cache.

    Args:
        attn_mask: bool[B, T, S] joint prefix+suffix attention mask.
        token_kv_cache_mask: bool[B, L] per-text-token visibility.
        prefix_len: Number of prefix positions (images + text).

    Returns:
        The masked bool[B, T, S] attention mask.
    """
    batch_size, num_queries, num_keys = attn_mask.shape
    text_len = token_kv_cache_mask.shape[1]
    excluded_keys = torch.zeros(
        batch_size, num_keys, dtype=torch.bool, device=attn_mask.device
    )
    excluded_keys[:, prefix_len - text_len : prefix_len] = ~token_kv_cache_mask
    suffix_queries = torch.zeros(num_queries, dtype=torch.bool, device=attn_mask.device)
    suffix_queries[prefix_len:] = True
    blocked = suffix_queries[None, :, None] & excluded_keys[:, None, :]
    return attn_mask & ~blocked


def posemb_sincos(
    pos: torch.Tensor,
    embedding_dim: int,
    min_period: float = 4e-3,
    max_period: float = 4.0,
) -> torch.Tensor:
    """Sine-cosine positional embedding for scalar positions.

    Args:
        pos: (B,) float positions
        embedding_dim: output dimension (must be even)

    Returns:
        (B, embedding_dim) positional embedding
    """
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = torch.linspace(
        0.0, 1.0, embedding_dim // 2, device=pos.device, dtype=torch.float32
    )
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = torch.einsum("i,j->ij", pos.float(), 1.0 / period * 2 * torch.pi)
    # Match JAX which keeps posemb in float32. However, PT Linear does not support
    # mixed float32/bf16 matmul, so cast back to the model's embed_dtype.
    # The caller should upcast to float32 if needed for high-precision ops.
    return torch.cat([torch.sin(sinusoid_input), torch.cos(sinusoid_input)], dim=-1).to(
        pos.dtype
    )


class Pi0(model.BaseModel):
    """Pi0 flow matching model for continuous action generation."""

    def __init__(self, config: Pi0Config):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.pcd = config.pcd
        self.embed_dtype = _str_to_dtype(config.dtype)
        self._config = config

        # VLM token output (subtask supervision/generation) configuration.
        self.vlm_vla = config.mode == "vlm_vla"
        self.language_loss_weight = config.language_loss_weight
        self.action_loss_weight = config.action_loss_weight
        self.stop_gradient_to_vlm = config.stop_gradient_to_vlm and self.vlm_vla
        self.max_new_tokens = config.max_new_tokens
        self.language_temperature = config.language_temperature

        paligemma_config = gemma.get_config(config.paligemma_variant)
        action_expert_config = gemma.get_config(config.action_expert_variant)

        # Gemma LLM with dual experts
        # Expert 0 (PaliGemma) uses regular RMSNorm; Expert 1 (Action Expert) may use adaRMS
        adarms = [False, config.pi05]
        self.llm = gemma.Module(
            configs=[paligemma_config, action_expert_config],
            embed_dtype=config.dtype,
            adarms=adarms,
            use_gradient_checkpointing=True,
        )
        if self.stop_gradient_to_vlm:
            for layer in self.llm.layers:
                layer.attn.detach_prefix_kv = True

        # SigLIP vision encoder
        self.img = siglip.SigLIPViT(
            variant="So400m/14",
            pool_type="none",
            num_classes=paligemma_config.width,
            remat=True,
            dtype_mm=config.dtype,
        )

        action_expert_width = action_expert_config.width
        self.action_dim = config.action_dim

        # Action input projection
        self.action_in_proj = nn.Linear(config.action_dim, action_expert_width)

        if config.pi05:
            self.time_mlp_in = nn.Linear(action_expert_width, action_expert_width)
            self.time_mlp_out = nn.Linear(action_expert_width, action_expert_width)
        else:
            self.state_proj = nn.Linear(config.action_dim, action_expert_width)
            self.action_time_mlp_in = nn.Linear(
                2 * action_expert_width, action_expert_width
            )
            self.action_time_mlp_out = nn.Linear(
                action_expert_width, action_expert_width
            )

        # Action output projection
        self.action_out_proj = nn.Linear(action_expert_width, config.action_dim)

        # Optional PointNet
        if config.pcd:
            pointnet_config = pointnet.get_config(config.pointnet_variant)
            self.pointnet = pointnet.UncoloredPointNet(
                n_coordinates=pointnet_config.n_coordinates,
                output_dim=pointnet_config.output_dim,
                hidden_dim=pointnet_config.hidden_dim,
                hidden_depth=pointnet_config.hidden_depth,
            )

        self._init_weights()

    def _init_weights(self):
        """Initialize projection weights."""
        nn.init.normal_(self.action_in_proj.weight, std=0.02)
        nn.init.zeros_(self.action_in_proj.bias)
        nn.init.normal_(self.action_out_proj.weight, std=0.02)
        nn.init.zeros_(self.action_out_proj.bias)

        if self.pi05:
            nn.init.normal_(self.time_mlp_in.weight, std=0.02)
            nn.init.zeros_(self.time_mlp_in.bias)
            nn.init.normal_(self.time_mlp_out.weight, std=0.02)
            nn.init.zeros_(self.time_mlp_out.bias)
        else:
            nn.init.normal_(self.state_proj.weight, std=0.02)
            nn.init.zeros_(self.state_proj.bias)
            nn.init.normal_(self.action_time_mlp_in.weight, std=0.02)
            nn.init.zeros_(self.action_time_mlp_in.bias)
            nn.init.normal_(self.action_time_mlp_out.weight, std=0.02)
            nn.init.zeros_(self.action_time_mlp_out.bias)

    def embed_prefix(
        self, obs: model.Observation
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed the prefix (images + language + optional point cloud).

        Returns:
            tokens: (B, S, emb_dim) embedded tokens
            input_mask: (B, S) mask of valid tokens
            ar_mask: (S,) autoregressive mask, all False — or (B, S) when the
                observation carries a per-token ``token_ar_mask`` (VLM token
                output: image tokens stay bidirectional, text tokens follow the
                per-token mask so the response/EOS region is causal)
        """
        tokens = []
        input_mask = []
        ar_mask = []
        # Per-batch causal structure is only needed when the text tokens carry
        # their own autoregressive mask; the action-only path keeps the original
        # shared 1D all-False mask.
        per_batch_ar = obs.token_ar_mask is not None
        ar_chunks = []

        # Embed images through SigLIP
        for name in obs.images:
            image_tokens, _ = self.img(obs.images[name])  # (B, num_patches, width)
            tokens.append(image_tokens)

            # Image tokens use bidirectional attention
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name], "b -> b s", s=image_tokens.shape[1]
                )
            )
            ar_mask += [False] * image_tokens.shape[1]
            if per_batch_ar:
                ar_chunks.append(
                    torch.zeros(
                        image_tokens.shape[:2],
                        dtype=torch.bool,
                        device=image_tokens.device,
                    )
                )

        # Add language tokens
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.llm.embed(obs.tokenized_prompt)
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]
            if per_batch_ar:
                ar_chunks.append(obs.token_ar_mask.bool())

        # Add point cloud tokens
        if self.pcd and obs.pcd_xyz is not None:
            # pcd_xyz: (B, 16, 2025, 3)
            # PointNet expects (B, num_points, 3)
            B = obs.pcd_xyz.shape[0]
            pcd_flat = obs.pcd_xyz.reshape(B, -1, 3)  # (B, 16*2025, 3)
            pcd_tokens = self.pointnet(pcd_flat)  # (B, 16, 2048)
            # Reshape to match expected dimensions
            if pcd_tokens.dim() == 2:
                pcd_tokens = pcd_tokens.unsqueeze(1)  # (B, 1, 2048)

            tokens.append(pcd_tokens)
            input_mask.append(
                torch.ones(
                    pcd_tokens.shape[:2], dtype=torch.bool, device=pcd_tokens.device
                )
            )
            ar_mask += [False] * pcd_tokens.shape[1]
            if per_batch_ar:
                ar_chunks.append(
                    torch.zeros(
                        pcd_tokens.shape[:2],
                        dtype=torch.bool,
                        device=pcd_tokens.device,
                    )
                )

        tokens = torch.cat(tokens, dim=1)
        input_mask = torch.cat(input_mask, dim=1)
        if per_batch_ar:
            ar_mask = torch.cat(ar_chunks, dim=1)
        else:
            ar_mask = torch.tensor(ar_mask, device=tokens.device)
        return tokens, input_mask, ar_mask

    def embed_suffix(
        self,
        obs: model.Observation,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Embed the suffix (state + noisy actions + time embedding).

        Args:
            obs: observation
            noisy_actions: (B, action_horizon, action_dim)
            timestep: (B,) float timestep values

        Returns:
            tokens: (B, S, emb_dim)
            input_mask: (B, S)
            ar_mask: (S,)
            adarms_cond: (B, emb_dim) or None
        """
        input_mask = []
        ar_mask = []
        tokens = []

        B = noisy_actions.shape[0]

        # Cast to embed_dtype to ensure consistency between training and inference
        # (during training FSDP2 casts forward inputs, but inference may pass float32).
        noisy_actions = noisy_actions.to(self.embed_dtype)
        timestep = timestep.to(self.embed_dtype)

        if not self.pi05:
            # Add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(
                torch.ones(B, 1, dtype=torch.bool, device=state_token.device)
            )

        # Embed actions
        action_tokens = self.action_in_proj(noisy_actions)

        # Time embedding
        time_emb = posemb_sincos(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0
        )

        if self.pi05:
            # Time MLP for adaRMS conditioning
            time_emb = self.time_mlp_in(time_emb)
            time_emb = F.silu(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = F.silu(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # Mix timestep + action through MLP
            time_tokens = einops.repeat(
                time_emb, "b emb -> b s emb", s=self.action_horizon
            )
            action_time_tokens = torch.cat([action_tokens, time_tokens], dim=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = F.silu(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None

        tokens.append(action_expert_tokens)
        input_mask.append(
            torch.ones(
                action_expert_tokens.shape[:2],
                dtype=torch.bool,
                device=action_expert_tokens.device,
            )
        )

        tokens = torch.cat(tokens, dim=1)
        input_mask = torch.cat(input_mask, dim=1)

        # Build ar_mask with correct length matching input_mask.shape[1]
        if not self.pi05:
            # state token [True] + action tokens [True, False, ..., False]
            ar_mask += [True] + [True] + [False] * (input_mask.shape[1] - 2)
        else:
            # action tokens only [True, False, ..., False]
            ar_mask += [True] + [False] * (input_mask.shape[1] - 1)
        ar_mask = torch.tensor(ar_mask, device=tokens.device)

        return tokens, input_mask, ar_mask, adarms_cond

    def compute_loss(
        self,
        observation: model.Observation,
        actions: torch.Tensor,
        *,
        train: bool = False,
        rng: torch.Generator | None = None,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Compute the flow matching loss (plus the language CE loss in vlm_vla).

        Returns:
            Action-only mode: (B, action_horizon) per-timestep MSE loss.
            vlm_vla mode: a dict with the combined scalar ``loss``
            (= ``action_loss_weight * flow + language_loss_weight * ce``) and
            detached ``action_loss`` / ``language_loss`` / ``language_acc``
            metrics.
        """
        B = actions.shape[0]
        device = actions.device

        if self.vlm_vla:
            if (
                observation.token_ar_mask is None
                or observation.token_loss_mask is None
                or observation.token_kv_cache_mask is None
            ):
                raise ValueError(
                    "vlm_vla mode requires per-token ar/loss/kv-cache masks on "
                    "the observation (produced by the subtask-supervision "
                    "tokenizer); got an action-only batch."
                )
            if observation.pcd_xyz is not None:
                raise NotImplementedError(
                    "vlm_vla mode assumes the text tokens are the final prefix "
                    "block; point-cloud tokens are not supported."
                )

        # Preprocess first (requries float32 for image ops),
        # then cast to model dtype for FSDP2 mixed precision compatibility.
        observation = model.preprocess_observation(observation, train=train, rng=rng)

        embed_dtype = self.embed_dtype
        observation = model._observation_to_dtype(observation, embed_dtype)
        actions = actions.to(dtype=embed_dtype)
        dtype = actions.dtype

        # Sample noise and time (or use provided values for reproducibility)
        if noise is None:
            noise = torch.randn(
                actions.shape, device=device, dtype=dtype, generator=rng
            )
        else:
            noise = noise.to(dtype=dtype)
        if time is None:
            time = (
                torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.0))
                .sample((B,))
                .to(device=device, dtype=dtype)
            )
            time = time * 0.999 + 0.001
        else:
            time = time.to(dtype=dtype)
        time_expanded = time[:, None, None]

        # Flow matching interpolation
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # One forward pass for prefix + suffix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, time
        )

        input_mask = torch.cat([prefix_mask, suffix_mask], dim=1)
        if prefix_ar_mask.dim() == 2:
            # Per-batch prefix ar (vlm_vla): expand the shared suffix ar to
            # match before concatenating along the sequence dimension.
            suffix_ar_mask = suffix_ar_mask.unsqueeze(0).expand(B, -1)
            ar_mask = torch.cat([prefix_ar_mask, suffix_ar_mask], dim=1)
        else:
            ar_mask = torch.cat([prefix_ar_mask, suffix_ar_mask], dim=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        if self.vlm_vla:
            # The action expert must not attend KV-excluded text tokens (EOS),
            # matching the generation-time view where they never enter the cache.
            attn_mask = block_suffix_from_excluded_prefix(
                attn_mask,
                observation.token_kv_cache_mask,
                prefix_len=prefix_mask.shape[1],
            )
            # Position numbering also matches the generation-time view: tokens
            # the action expert never sees (EOS) do not occupy a position slot,
            # so the suffix RoPE base is identical between training and eval
            # (where the EOS is never written into the cache). This is a
            # deliberate divergence from the reference implementation, whose
            # training positions count the EOS slot while its eval does not.
            # The EOS key keeps a (duplicated) position, but only its own —
            # unused — query ever attends it.
            prefix_len = prefix_mask.shape[1]
            text_len = observation.token_kv_cache_mask.shape[1]
            position_mask = input_mask.clone()
            position_mask[:, prefix_len - text_len : prefix_len] &= (
                observation.token_kv_cache_mask
            )
            positions = torch.cumsum(position_mask.int(), dim=1) - 1
        else:
            positions = torch.cumsum(input_mask.int(), dim=1) - 1

        prefix_out, suffix_out = self.llm(
            [prefix_tokens, suffix_tokens],
            positions=positions,
            mask=attn_mask,
            adarms_cond=[None, adarms_cond],
        )[0]

        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        flow_loss = torch.mean(torch.square(v_t - u_t), dim=-1)
        if not self.vlm_vla:
            return flow_loss

        ce_loss, language_acc = self._compute_language_loss(
            prefix_out,
            observation.tokenized_prompt,
            observation.token_loss_mask,
        )
        action_loss = flow_loss.mean()
        total = (
            self.action_loss_weight * action_loss + self.language_loss_weight * ce_loss
        )
        return {
            "loss": total,
            "action_loss": action_loss.detach(),
            "language_loss": ce_loss.detach(),
            "language_acc": language_acc,
        }

    def _compute_language_loss(
        self,
        prefix_out: torch.Tensor,
        tokens: torch.Tensor,
        token_loss_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Next-token CE over the supervised text positions.

        The text block is the final segment of the prefix, so its offset is
        derived from the tensor shapes rather than a hardcoded image-token
        count. The hidden state at text position ``i`` predicts token ``i+1``;
        logits come from the tied embedding projection (the vendored model has
        no separate LM head). The loss is normalized per sample over its
        supervised positions, then averaged over the batch.

        Returns:
            ``(ce_loss, language_acc)`` — the differentiable scalar loss and a
            detached token-accuracy metric over the supervised positions.
        """
        text_len = tokens.shape[1]
        offset = prefix_out.shape[1] - text_len
        # Text positions 0..L-2 (rows offset..end-2 of the prefix) predict
        # tokens 1..L-1.
        language_out = prefix_out[:, offset:-1]
        logits = self.llm.embedder.decode(language_out)

        targets = tokens[:, 1:]
        loss_mask = token_loss_mask[:, 1:].to(torch.float32)
        denominator = loss_mask.sum(dim=-1).clamp(min=1.0)

        ce_per_position = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        ).view(targets.shape)
        ce_loss = ((ce_per_position * loss_mask).sum(dim=-1) / denominator).mean()

        with torch.no_grad():
            predictions = logits.argmax(dim=-1)
            correct = (predictions == targets).float()
            language_acc = ((correct * loss_mask).sum(dim=-1) / denominator).mean()
        return ce_loss, language_acc

    def sample_actions(
        self,
        observation: model.Observation,
        *,
        num_steps: int = 10,
        noise: torch.Tensor | None = None,
        rng: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample actions using Euler ODE solver.

        Args:
            observation: input observation
            num_steps: number of ODE solver steps
            noise: optional initial noise of shape (B, action_horizon, action_dim)
            rng: random generator

        Returns:
            actions: (B, action_horizon, action_dim)
        """
        observation = model.preprocess_observation(observation, train=False)

        # Pre-fill KV cache with prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = torch.cumsum(prefix_mask.int(), dim=1) - 1

        _, kv_cache = self.llm(
            [prefix_tokens, None],
            positions=positions,
            mask=prefix_attn_mask,
        )

        return self._denoise_actions(
            observation,
            kv_cache,
            prefix_mask,
            num_steps=num_steps,
            noise=noise,
            rng=rng,
        )

    def _denoise_actions(
        self,
        observation: model.Observation,
        kv_cache,
        kv_valid_mask: torch.Tensor,
        *,
        num_steps: int,
        noise: torch.Tensor | None,
        rng: torch.Generator | None,
    ) -> torch.Tensor:
        """Euler ODE denoising loop attending a prefilled KV cache.

        Args:
            observation: preprocessed observation (only ``state`` is read here).
            kv_cache: per-layer tuple caches (action-only path) or a frozen
                :class:`StaticKVCache` (after subtask generation).
            kv_valid_mask: (B, S_cache) validity of the cache columns the
                action expert may attend to — the prefix validity in the
                action-only path; prefix validity plus per-row generated-token
                validity (EOS and post-EOS columns excluded) after generation.
            num_steps: number of ODE solver steps.
            noise: optional initial noise (B, action_horizon, action_dim).
            rng: random generator.

        Returns:
            actions: (B, action_horizon, action_dim)
        """
        dt = -1.0 / num_steps
        B = observation.state.shape[0]
        device = observation.state.device

        if noise is None:
            noise = torch.randn(
                B, self.action_horizon, self.action_dim, device=device, generator=rng
            )

        x_t = noise
        t = 1.0

        # Euler integration
        while t >= -dt / 2:
            t_tensor = torch.full((B,), t, device=device, dtype=torch.float32)
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, t_tensor
            )

            # Build attention mask: suffix attends to the valid cache columns
            # and to itself
            suffix_len = suffix_tokens.shape[1]
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            cache_to_suffix_mask = einops.repeat(
                kv_valid_mask, "b p -> b s p", s=suffix_len
            )
            full_attn_mask = torch.cat([cache_to_suffix_mask, suffix_attn_mask], dim=-1)

            suffix_positions = (
                torch.sum(kv_valid_mask, dim=-1)[:, None]
                + torch.cumsum(suffix_mask.int(), dim=-1)
                - 1
            )

            _, suffix_out = self.llm(
                [None, suffix_tokens],
                positions=suffix_positions,
                mask=full_attn_mask,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )[0]

            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            x_t = x_t + dt * v_t
            t = t + dt

        return x_t

    @torch.no_grad()
    def generate_language(
        self,
        observation: model.Observation,
        *,
        eos_token_id: int,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict:
        """Autoregressively generate the subtask text with the VLM.

        The prefix is right-aligned so all rows share one static-cache write
        column per step. Rows finish independently: once a row emits EOS, its
        subsequent writes are masked dummies (the shared column stays invalid
        for that row), so neither the EOS nor anything after it is ever visible
        to later generation steps or to the action expert.

        Args:
            observation: eval observation carrying the generation-prefix masks.
            eos_token_id: tokenizer EOS id terminating a row.
            max_new_tokens: generation budget (defaults to the config value).
            temperature: 0 = greedy (default), > 0 samples.

        Returns:
            A dict with:
                ``tokens``: (B, T<=max_new_tokens) generated ids — rows that
                finished early are padded with EOS;
                ``eos_steps``: (B,) step at which each row emitted EOS
                (``max_new_tokens`` when it never did);
                ``terminated``: (B,) bool, True when the row emitted EOS;
                ``kv_cache``: the frozen :class:`StaticKVCache`;
                ``cache_valid_mask``: (B, S_cache) per-row column validity for
                the subsequent denoise (prefix + pre-EOS generated columns);
                ``observation``: the preprocessed observation, reusable by the
                denoise loop.
        """
        if not self.vlm_vla:
            raise RuntimeError("generate_language requires mode='vlm_vla'.")
        if observation.token_ar_mask is None:
            raise ValueError(
                "generate_language requires the generation-prefix token masks "
                "(produced by the subtask-supervision tokenizer)."
            )
        max_new = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        temp = self.language_temperature if temperature is None else temperature

        observation = model.preprocess_observation(observation, train=False)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_tokens, prefix_mask, prefix_attn_mask = left_to_right_align(
            prefix_tokens, prefix_mask, prefix_attn_mask
        )

        B, prefill_size = prefix_mask.shape
        device = prefix_tokens.device
        prefill_len = prefix_mask.sum(dim=-1)  # (B,) valid prefix length per row
        paligemma_config = self.llm.configs[0]
        cache = StaticKVCache(
            batch_size=B,
            max_len=prefill_size + max_new,
            num_layers=paligemma_config.depth,
            num_kv_heads=paligemma_config.num_kv_heads,
            head_dim=paligemma_config.head_dim,
            dtype=self.embed_dtype,
            device=device,
        )

        # Prefill: keys live in the (padded) static buffer, so the mask covers
        # the generation columns too (all invalid for now).
        prefill_attn_mask = F.pad(prefix_attn_mask, (0, max_new))
        positions = torch.cumsum(prefix_mask.int(), dim=1) - 1
        cache.set_write_col(0)
        outputs, _ = self.llm(
            [prefix_tokens, None],
            positions=positions,
            mask=prefill_attn_mask,
            kv_cache=cache,
        )
        # Right alignment puts every row's last VALID token in the last column.
        logits = self.llm.embedder.decode(outputs[0][:, -1]).float()

        done = torch.zeros(B, dtype=torch.bool, device=device)
        eos_steps = torch.full((B,), max_new, dtype=torch.long, device=device)
        generated_valid = torch.zeros(B, max_new, dtype=torch.bool, device=device)
        generated_tokens = []

        for step in range(max_new):
            if temp > 0:
                token = torch.multinomial(
                    F.softmax(logits / temp, dim=-1), num_samples=1
                ).squeeze(-1)
            else:
                token = logits.argmax(dim=-1)
            # Rows that already finished keep emitting EOS in the output text.
            token = torch.where(done, torch.full_like(token, eos_token_id), token)
            generated_tokens.append(token)

            is_eos = token == eos_token_id
            newly_done = (~done) & is_eos
            eos_steps[newly_done] = step
            # A column is valid only for rows whose token this step is a real
            # (non-EOS) continuation; EOS and post-EOS columns stay invalid.
            generated_valid[:, step] = (~done) & (~is_eos)
            done = done | is_eos
            if bool(done.all()):
                break

            token_embedding = self.llm.embed(token[:, None])
            step_positions = (prefill_len + step).long()[:, None]
            cache.set_write_col(prefill_size + step)
            keys_valid = torch.cat([prefix_mask, generated_valid], dim=1)
            outputs, _ = self.llm(
                [token_embedding, None],
                positions=step_positions,
                mask=keys_valid[:, None, :],
                kv_cache=cache,
            )
            logits = self.llm.embedder.decode(outputs[0][:, -1]).float()

        cache.freeze()
        return {
            "tokens": torch.stack(generated_tokens, dim=1),
            "eos_steps": eos_steps,
            "terminated": done,
            "kv_cache": cache,
            "cache_valid_mask": torch.cat([prefix_mask, generated_valid], dim=1),
            "observation": observation,
        }

    @torch.no_grad()
    def reason_and_sample_actions(
        self,
        observation: model.Observation,
        *,
        eos_token_id: int,
        num_steps: int = 10,
        noise: torch.Tensor | None = None,
        rng: torch.Generator | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Generate the subtask text, then denoise actions against its cache.

        Returns:
            ``(actions, generation)`` where ``actions`` is
            (B, action_horizon, action_dim) and ``generation`` is the
            :meth:`generate_language` result dict.
        """
        generation = self.generate_language(
            observation,
            eos_token_id=eos_token_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        actions = self._denoise_actions(
            generation["observation"],
            generation["kv_cache"],
            generation["cache_valid_mask"],
            num_steps=num_steps,
            noise=noise,
            rng=rng,
        )
        return actions, generation

    def forward(
        self,
        observation: model.Observation,
        actions: torch.Tensor,
        *,
        train: bool = True,
        rng: torch.Generator | None = None,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Default forward computes loss."""
        return self.compute_loss(
            observation, actions, train=train, rng=rng, noise=noise, time=time
        )

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.llm.gradient_checkpointing = True
        self.img.encoder.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing (used by the eval / no-recompute path)."""
        self.llm.gradient_checkpointing = False
        self.img.encoder.gradient_checkpointing = False
