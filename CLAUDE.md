# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**OminiControl** — a minimal universal control framework for Diffusion Transformers (FLUX). It injects control signals (subject-driven, spatial alignment like canny/depth/coloring/inpainting) into a frozen base model via tiny LoRA adapters (~0.1% extra parameters). OminiControl2 adds efficient conditioning (KV-cache, compact tokens, token integration).

Validated stack (see `requirements.txt`): Python 3.12, torch 2.8 (cu12x), `diffusers==0.38.0`, `transformers>=4.55,<5`, `huggingface_hub<1.0`, plus `peft`, `prodigyopt`, `lightning`, `datasets`.

There is no test suite, linter, or formatter wired up — verification is end-to-end via the example notebooks and training scripts.

## Repository layout

```
omini/
  pipeline/flux_omini.py        # Core inference pipeline (Condition, generate, LoRA-aware forward passes)
  train_flux/
    trainer.py                  # LightningModule, TrainingCallback, train() entry point
    train_subject.py            # Subject200K dataset + script entry point
    train_spatial_alignment.py  # canny/depth/coloring/deblurring/fill/depth_pred
    train_multi_condition.py    # Multiple simultaneous conditions per sample
    train_token_integration.py  # Token integration (fill task) for OminiControl2
    train_custom.py             # Template for custom tasks
train/
  config/                       # YAML configs (one per task)
  script/                       # Shell scripts (train_*.sh, data_download*.sh)
  requirements.txt              # Training-only extras (lightning, datasets, prodigyopt, wandb, torchvision)
examples/                       # Jupyter notebooks (subject, spatial, inpainting, style LoRA, etc.)
assets/                         # Test images used by inference & periodic sampling during training
```

## Inference pipeline (`omini/pipeline/flux_omini.py`)

Everything goes through `generate(pipe, prompt, conditions=[...], **kwargs)`. It swaps in custom forward passes for FLUX's transformer blocks so multiple "branches" share attention.

Core building blocks:

- **`Condition`** — a PIL image + adapter name + spatial offsets (`position_delta`, `position_scale`) + optional `latent_mask`. `Condition.encode()` VAE-encodes the image and adjusts the positional ids so the condition is placed in the shared grid next to the generated image.
- **`convert_to_condition(condition_type, raw_img)`** — converts raw PIL images to spatial conditions: `depth` (uses `depth-anything-small-hf`, lazy-loaded on CPU), `canny` (cv2 Canny edges), `coloring` (grayscale), `deblurring` (Gaussian blur).
- **`attn_forward` / `block_forward` / `single_block_forward` / `transformer_forward`** — LoRA-aware FLUX forward passes that run *N* text/image branches through dual-stream + single-stream blocks together. Branches are kept separate via `group_mask` (a bool matrix marking which cross-branch attention is allowed).
- **`lora_forward`** — fast PEFT-aware wrapper that applies a single named adapter at scale 1.0 without mutating `module.scaling` (replaces an older context manager).
- **`generate()`** — the main entry. Notable options:
  - `conditions: List[Condition]` — one per control branch.
  - `main_adapter` / per-condition adapter names — selects which LoRA is active on each branch.
  - `condition_scale` (default 1.0) — additive log-scale bias on attention logits between condition and non-condition branches; `1.0` reproduces original behavior exactly, `<1` weakens, `0` suppresses, `>1` strengthens.
  - `image_guidance_scale` — real CFG against an empty (black) condition; required `> 1.0` on FLUX.1-dev (use ~1.5), default `1.0` for FLUX.1-schnell.
  - `kv_cache=True` — writes keys/values for condition branches on step 0, reads them on subsequent steps (~1.5× speedup). Requires LoRA trained with `model.independent_condition: true`.
  - `latent_mask` + `Condition.is_complement=True` — token integration for the fill task; the generated latents are stitched back with the complement region after the denoising loop.
  - Note `guidance_scale=3.5` is **fixed** for FLUX.1-dev (training-matching embedding/distilled guidance — not a tunable knob).

## Training (`omini/train_flux/`)

All training scripts use the same Lightning-based harness in `trainer.py`:

1. `OminiModel` loads `flux_pipe_id` (e.g. `black-forest-labs/FLUX.1-dev`) at `bfloat16`, freezes everything except a fresh PEFT LoRA adapter (built from `lora_config` in the YAML, default `r=16` for subject, `r=4` for spatial).
2. `training_step` builds *N* condition branches (text, main image, plus each `condition_i`), forms a `group_mask`, and calls `transformer_forward` against a single target. Loss is flow-matching MSE `pred vs (x_1 - x_0)` on the main-image branch.
3. `TrainingCallback` logs to wandb (if `WANDB_API_KEY` set) and prints every `print_every_n_steps`, saves LoRA weights every `save_interval` steps, and runs `test_function(model, save_path, file_name)` every `sample_interval` steps.
4. `train()` is the Lightning `Trainer.fit()` driver. Multi-GPU DDP is auto-enabled when `WORLD_SIZE>1` and uses `ddp_find_unused_parameters_true` (the last single block's `to_q`/`proj_mlp` LoRA params only feed the discarded condition branch output and never get gradients).
5. Config comes from the env var `OMINI_CONFIG=<path to yaml>`.

To launch training, set the env vars in the script and run `accelerate launch`. See `train/script/train_*.sh` for the full set (subject, spatial, multi-condition, feature-reuse, compact-token-representation, token-integration).

### Custom tasks
Use `ominicontrol_art.ipynb` + `examples/combine_with_style_lora.ipynb` as references. For a brand-new task, copy `train_custom.py`, fill in `CustomDataset.__getitem__` (must return `{"image", "description", "condition_0", "condition_type_0", "position_delta_0", ...}` — keep `position_delta` consistent with what `test_function` uses), and implement `test_function()`.

`adapter_names` must include `None` for the text and main-image branches and one entry per condition branch (e.g. `[None, None, "default"]`); only adapters in this list are added to the model via `transformer.add_adapter(LoraConfig(**lora_config), adapter_name=...)`.

### Spatial conditioning tasks
Switch by setting `condition_type` in the config:
`canny`, `coloring`, `deblurring`, `depth`, `depth_pred`, `fill`. Each maps to a `convert_to_condition` branch in `ImageConditionDataset`.

### OminiControl2 efficiency features
- **Feature reuse / KV-cache**: set `model.independent_condition: true` in the YAML and pass `kv_cache=True` to `generate()`. Requires the special config (`train/config/feature_reuse.yaml`).
- **Compact encoding representation**: set `dataset.condition_size` smaller than `target_size` and add `dataset.position_scale: N` (e.g. condition 256 / target 512 / `position_scale=2`).
- **Token integration** (fill task): `train/config/token_integration.yaml` + `train_token_integration.py`. Conditions use `latent_mask` + `is_complement=True`; the post-loop canvas stitching reconciles masked generated latents with the unmasked complement.

## Common development tasks

Setup (per `requirements.txt` and `train/README.md`):
```bash
conda create -n omini python=3.12 && conda activate omini
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt          # for inference + notebooks
pip install -r train/requirements.txt     # adds lightning, datasets, prodigyopt, wandb, torchvision
```

Run an example notebook (from the repo root; the first cell `os.chdir("..")` puts you there):
```bash
jupyter nbconvert --to notebook --execute examples/subject.ipynb
```

Train (multi-GPU via `CUDA_VISIBLE_DEVICES`):
```bash
export OMINI_CONFIG=./train/config/subject.yaml
bash train/script/train_subject.sh        # or train_spatial_alignment.sh, train_multi_condition.sh, etc.
```

Datasets:
```bash
bash train/script/data_download/data_download1.sh   # Subject200K (~200K subject images, HF gated)
bash train/script/data_download/data_download2.sh   # text-to-image-2M shards
```

## LoRA adapter loading — multi-adapter gotcha

When loading more than one adapter via repeated `pipe.load_lora_weights(..., adapter_name=...)`, activation is **not** implicit — call `pipe.set_adapters([...])` after the final `load_lora_weights`. Otherwise only the last-loaded adapter stays active (see top-of-`examples/spatial.ipynb` for the pattern).

## Subject LoRAs on FLUX.1-dev vs FLUX.1-schnell

`schnell` (the default in `subject.ipynb`): `image_guidance_scale=1.0`, ~8 steps.
`dev` (see `subject_dev.ipynb`): `image_guidance_scale≈1.5` (the *only* tunable CFG), `guidance_scale=3.5` **must stay fixed** (matches training), ~20–28 steps. Preprocess inputs with center-crop + resize to 512×512 (or 1024×1024 for `subject_1024_beta`); the pipeline does not do this automatically. The `subject_512` LoRA uses `position_delta=(0, 32)`; `subject_1024_beta` uses `(0, -32)` (sign is arbitrary — both separate condition from target).
