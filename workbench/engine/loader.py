"""Model loading. Thin wrapper so the rest of the code never imports mlx_lm.load directly."""
from mlx_lm import load

TEST_MODEL = "mlx-community/Qwen3-0.6B-4bit"
MAIN_MODEL = "mlx-community/Qwen3-8B-8bit"

# Describe-sidecar VLM for real image understanding (workbench/attachments
# processors.VlmImageProcessor), loaded via mlx-vlm (not mlx_lm.load -- see
# workbench/attachments/processors.py). Same family as MAIN_MODEL (Qwen3)
# for ecosystem/prompt-format consistency. Verified to exist on
# mlx-community as of 2026-07-17: ~5.78 GB total on disk (4-bit,
# 2 safetensors shards), well within the ~5-8 GB budget, comfortably
# resident alongside MAIN_MODEL in 48 GB unified memory.
VLM_MODEL = "mlx-community/Qwen3-VL-8B-Instruct-4bit"


def load_model(repo_id: str):
    """Returns (model, tokenizer). Downloads to the HF cache on first call."""
    return load(repo_id)
