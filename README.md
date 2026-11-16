---
title: Diffusers Comparison Playground
emoji: 🎛️
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 6.32.0
app_file: app.py
pinned: false
license: apache-2.0
---

# Diffusers Model Comparison Playground

A template Hugging Face Space for comparing any number of `diffusers` checkpoints —
text-to-image **and** image-to-image — side by side with the same prompt and seed.

## Usage

1. Tick one or more checkpoints in the **Models to compare** list.
2. Type a prompt (and an init image when using *Image → Image*).
3. Hit **Generate** — every selected model runs with the same seed and settings, and
   the results land in one gallery for easy comparison.

## Adding checkpoints

Two ways, both take a raw repo id (`author/model`) or a full
`https://huggingface.co/author/model` URL:

- **In the UI** — open the *Add any diffusers checkpoint* accordion, paste the id,
  and press **Add model**. The repo is validated against the Hub and becomes
  selectable immediately (in-memory, for the current run).
- **In the repo** — append it to `models.yaml` and restart the Space. That makes it
  part of the default lineup for everyone.

Only repos whose Hub metadata says `library_name: diffusers` are accepted, so
ComfyUI single-file checkpoints and GGUF quantizations are rejected up front.

## Configuration

`models.yaml` holds the default lineup and fallback generation settings:

```yaml
defaults:
  steps: 25
  guidance: 7.0
  width: 1024
  height: 1024

models:
  - repo: Tongyi-MAI/Z-Image-Turbo
  - repo: stabilityai/sdxl-turbo
  ...
```

Per-pipeline defaults (steps, guidance scale, base resolution) are auto-detected from
the loaded pipeline class, with per-repo overrides in `app.py` (`REPO_OVERRIDES`) for
turbo/schnell-style few-step models.

## Hardware & notes

- Runs on CPU but is slow; a T4/L4/A10G GPU Space (or ZeroGPU) is recommended.
- Gated checkpoints (e.g. `black-forest-labs/FLUX.1-dev`): accept the license on the
  model page, then add a `HF_TOKEN` secret in the Space settings.
- At most `MAX_LOADED_PIPELINES` checkpoints stay resident; the least recently used
  one is evicted to keep VRAM in check.
- Pipelines keep their stock safety checker; no filters are disabled.
