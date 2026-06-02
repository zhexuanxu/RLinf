# Copyright (c) 2025, RLinf contributors.
#
# Self-contained PyTorch OpenPI 0.5 model package for embodied BEHAVIOR eval.
#
# This package vendors the optimized PyTorch OpenPI 0.5 implementation so that
# the eval / action-generation path is fully self-contained: it does not import
# the externally installed ``openpi`` package and does not patch ``transformers``.
#
# Layout:
#   utils/                vendored model core (pi0, gemma, siglip, ...).
#   policies/             BEHAVIOR input/output transforms (added with the model entry point).
#   dataconfig/           BEHAVIOR normalization / tokenizer config (added with the model entry point).
#   openpi_action_model.py  high-level entry point preserving the old interface.
#
# ``get_model`` (the factory used by ``rlinf.models.get_model``) is wired in once
# the action-model entry point lands; until then this package only exposes the
# vendored core under ``utils``.
