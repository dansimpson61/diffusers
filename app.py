"""Diffusers model comparison playground.

Compare any number of diffusers checkpoints (text-to-image and image-to-image)
by pointing the UI at a repo id or a huggingface.co URL.
"""

import gc
import inspect
import os
import random
import re
from pathlib import Path

try:
    import spaces
except ImportError:
    spaces = None

import gradio as gr
import torch
import yaml
from diffusers import AutoPipelineForImage2Image, DiffusionPipeline
from huggingface_hub import model_info


def gpu_function(fn):
    if spaces is not None:
        return spaces.GPU(duration=120)(fn)
    return fn

CONFIG_PATH = Path(__file__).parent / "models.yaml"
MAX_LOADED_PIPELINES = 2
TEXT2IMG = "text2img"
IMG2IMG = "img2img"

CLASS_DEFAULTS = {
    "StableDiffusionPipeline": {"steps": 30, "guidance": 7.5, "size": 512},
    "StableDiffusionXLPipeline": {"steps": 30, "guidance": 7.0, "size": 1024},
    "StableDiffusion3Pipeline": {"steps": 28, "guidance": 7.0, "size": 1024},
    "StableDiffusionImg2ImgPipeline": {"steps": 30, "guidance": 7.5, "size": 512},
    "StableDiffusionXLImg2ImgPipeline": {"steps": 30, "guidance": 7.0, "size": 1024},
    "FluxPipeline": {"steps": 28, "guidance": 3.5, "size": 1024},
    "FluxImg2ImgPipeline": {"steps": 28, "guidance": 3.5, "size": 1024},
    "ZImagePipeline": {"steps": 8, "guidance": 3.0, "size": 1024},
    "QwenImagePipeline": {"steps": 30, "guidance": 4.0, "size": 1024},
}
FALLBACK_DEFAULTS = {"steps": 25, "guidance": 7.0, "size": 1024}

REPO_OVERRIDES = {
    "stabilityai/sdxl-turbo": {"steps": 4, "guidance": 0.0},
    "black-forest-labs/FLUX.1-schnell": {"steps": 4, "guidance": 0.0},
    "Tongyi-MAI/Z-Image-Turbo": {"steps": 8, "guidance": 3.0},
}


def hub_token():
    return os.getenv("HF_TOKEN") or None


def load_config():
    if not CONFIG_PATH.exists():
        return {}, []
    config = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    repos = []
    for entry in config.get("models") or []:
        repo = entry["repo"] if isinstance(entry, dict) else str(entry)
        if repo and repo not in repos:
            repos.append(repo)
    return config.get("defaults") or {}, repos


CONFIG_DEFAULTS, REGISTRY = load_config()
FALLBACK_DEFAULTS = {
    "steps": 25,
    "guidance": 7.0,
    "size": 1024,
    **{k: v for k, v in CONFIG_DEFAULTS.items() if k in ("steps", "guidance")},
    **({"size": CONFIG_DEFAULTS["width"]} if "width" in CONFIG_DEFAULTS else {}),
}


class PipelineCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self._pipelines = {}
        self._order = []

    def get(self, repo):
        if repo in self._pipelines:
            self._order.remove(repo)
            self._order.append(repo)
            return self._pipelines[repo]
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        pipe = DiffusionPipeline.from_pretrained(
            repo, torch_dtype=dtype, token=hub_token()
        )
        pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        self._pipelines[repo] = pipe
        self._order.append(repo)
        while len(self._order) > self.capacity:
            self._evict(self._order[0])
        return pipe

    def _evict(self, repo):
        self._order.remove(repo)
        del self._pipelines[repo]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


CACHE = PipelineCache(MAX_LOADED_PIPELINES)


def resolve_repo(value):
    value = (value or "").strip()
    if not value:
        return None, "Enter a repo id or a huggingface.co URL."
    match = re.search(r"huggingface\.co/(?:spaces/)?([\w.-]+/[\w.-]+)", value)
    repo = match.group(1) if match else value
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        return None, f"`{value}` is not a valid repo id."
    try:
        info = model_info(repo, token=hub_token())
    except Exception as exc:
        return None, f"Could not reach the Hub for `{repo}`: {exc}"
    if info.library_name not in (None, "diffusers"):
        return None, (
            f"`{repo}` is a `{info.library_name}` repo. Only native diffusers "
            "checkpoints are supported (no ComfyUI single-file or GGUF repos)."
        )
    return repo, f"Added **{repo}**."


def defaults_for(repo, pipeline_class):
    defaults = dict(FALLBACK_DEFAULTS)
    if pipeline_class in CLASS_DEFAULTS:
        defaults.update(CLASS_DEFAULTS[pipeline_class])
    if repo in REPO_OVERRIDES:
        defaults.update(REPO_OVERRIDES[repo])
    return defaults


def image2image_variant(pipe):
    return AutoPipelineForImage2Image.from_pipe(pipe)


def accepts(pipe, arg):
    return arg in inspect.signature(type(pipe).__call__).parameters


def round64(value):
    return max(64, int(value) // 64 * 64)


@gpu_function
def generate(selected, prompt, negative_prompt, mode, init_image, steps, guidance,
             width, height, strength, seed, randomize_seed):
    if not selected:
        return [], "Select at least one model.", seed
    if not prompt.strip():
        return [], "Enter a prompt.", seed
    if mode == IMG2IMG and init_image is None:
        return [], "Image → Image needs an init image.", seed
    if randomize_seed or seed is None:
        seed = random.randint(0, 2**32 - 1)

    results, errors = [], []
    for repo in selected:
        try:
            pipe = CACHE.get(repo)
            generator = torch.Generator(device=pipe.device).manual_seed(int(seed))
            kwargs = {
                "prompt": prompt,
                "num_inference_steps": int(steps),
                "guidance_scale": float(guidance),
                "generator": generator,
            }
            target = pipe
            if mode == IMG2IMG:
                target = image2image_variant(pipe)
                kwargs["image"] = init_image
                kwargs["strength"] = float(strength)
            else:
                kwargs["width"] = round64(width)
                kwargs["height"] = round64(height)
            if negative_prompt.strip() and accepts(target, "negative_prompt"):
                kwargs["negative_prompt"] = negative_prompt
            result = target(**kwargs)
            results.append((result.images[0], repo))
        except Exception as exc:
            errors.append(f"**{repo}** failed: `{exc}`")

    summary = f"Seed: `{seed}` — {len(results)} image(s) generated."
    if errors:
        summary += "\n\n" + "\n\n".join(errors)
    return results, summary, seed


def add_model(value, current):
    repo, message = resolve_repo(value)
    if repo is None:
        return gr.update(), message
    if repo not in REGISTRY:
        REGISTRY.append(repo)
    return gr.update(choices=REGISTRY, value=list(dict.fromkeys(list(current) + [repo]))), message


def toggle_mode(mode):
    is_i2i = mode == IMG2IMG
    return gr.update(visible=is_i2i), gr.update(visible=is_i2i)


def pipeline_class_from_tags(repo):
    try:
        info = model_info(repo, token=hub_token())
    except Exception as exc:
        return None, f"Could not fetch Hub info for `{repo}`: {exc}"
    for tag in info.tags or []:
        if tag.startswith("diffusers:"):
            return tag.split(":", 1)[1], ""
    return None, f"`{repo}` does not advertise a diffusers pipeline class; adjust settings manually."


def sync_defaults(selected):
    if len(selected) != 1:
        return gr.update(), gr.update(), gr.update(), ""
    class_name, message = pipeline_class_from_tags(selected[0])
    if class_name is None:
        return gr.update(), gr.update(), gr.update(), message
    defaults = defaults_for(selected[0], class_name)
    return (
        gr.update(value=defaults["steps"]),
        gr.update(value=defaults["guidance"]),
        gr.update(value=defaults["size"]),
        "",
    )


with gr.Blocks(title="Diffusers Comparison Playground") as demo:
    gr.Markdown(
        "# 🎛️ Diffusers Comparison Playground\n"
        "Run the same prompt and seed through several diffusers checkpoints and "
        "compare the results. Add any checkpoint by repo id or huggingface.co URL."
    )
    with gr.Row():
        with gr.Column(scale=1):
            model_select = gr.CheckboxGroup(
                choices=REGISTRY,
                value=REGISTRY[:1],
                label="Models to compare",
                info="Pick one or more; results share the same seed.",
            )
            with gr.Accordion("Add any diffusers checkpoint", open=False):
                new_repo = gr.Textbox(
                    label="Repo id or URL",
                    placeholder="author/model or https://huggingface.co/author/model",
                )
                add_btn = gr.Button("Add model")
            add_status = gr.Markdown()
            mode = gr.Radio(
                choices=[(label, value) for label, value in
                         (("Text → Image", TEXT2IMG), ("Image → Image", IMG2IMG))],
                value=TEXT2IMG,
                label="Mode",
            )
            init_image = gr.Image(type="pil", label="Init image", visible=False)
            strength = gr.Slider(0.1, 1.0, value=0.65, step=0.05,
                                 label="Strength (image-to-image)", visible=False)
            prompt = gr.Textbox(lines=3, label="Prompt",
                                placeholder="A serene mountain lake at dawn, cinematic light")
            negative_prompt = gr.Textbox(lines=1, label="Negative prompt",
                                         placeholder="blurry, low quality")
            with gr.Row():
                steps = gr.Slider(1, 100, value=25, step=1, label="Steps")
                guidance = gr.Slider(0.0, 15.0, value=7.0, step=0.1, label="Guidance scale")
            with gr.Row():
                width = gr.Slider(256, 2048, value=1024, step=64, label="Width")
                height = gr.Slider(256, 2048, value=1024, step=64, label="Height")
            with gr.Row():
                seed = gr.Number(value=None, precision=0, label="Seed")
                randomize_seed = gr.Checkbox(value=True, label="Randomize seed")
            generate_btn = gr.Button("Generate", variant="primary")
        with gr.Column(scale=2):
            gallery = gr.Gallery(columns=2, label="Results")
            status = gr.Markdown()

    mode.change(toggle_mode, mode, [init_image, strength])
    add_btn.click(add_model, [new_repo, model_select], [model_select, add_status])
    model_select.change(sync_defaults, model_select, [steps, guidance, width, add_status])
    generate_btn.click(
        generate,
        [model_select, prompt, negative_prompt, mode, init_image, steps, guidance,
         width, height, strength, seed, randomize_seed],
        [gallery, status, seed],
    )

if __name__ == "__main__":
    demo.queue(max_size=16).launch()
