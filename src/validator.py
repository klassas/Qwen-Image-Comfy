"""Input validation for the Qwen-Image ComfyUI worker.

The field set is what the RunPod Media Studio app actually sends for its
Qwen-Image model (see app RunPodClient.kt): flat fields prompt /
negative_prompt / image_url / image_base64 / width / height, plus a
`workflow` object this worker accepts and ignores — it builds the graph
itself in src/workflow.py, same contract as the MiniMax-H3 worker.
"""

import math

from src import config
from src.errors import ValidationError


def _validate_image_source(inp: dict, url_key: str, b64_key: str) -> bool:
    has_url = bool(inp.get(url_key))
    has_b64 = bool(inp.get(b64_key))
    if has_url and has_b64:
        raise ValidationError(
            f"Provide either {url_key} or {b64_key}, not both.",
            hint="Remove one of the image source fields.",
        )
    return has_url or has_b64


def validate_input(inp: dict) -> dict:
    """Validate and normalize job input. Returns a cleaned dict with defaults applied."""
    if not isinstance(inp, dict):
        raise ValidationError(
            "Input must be a JSON object.",
            hint="Send a JSON object with at least a 'prompt' field.",
        )

    has_image = _validate_image_source(inp, "image_url", "image_base64")
    mode = "edit" if has_image else "t2i"

    prompt = inp.get("prompt", "")
    if not isinstance(prompt, str):
        raise ValidationError("prompt must be a string.")
    prompt = prompt.strip()
    if not prompt:
        raise ValidationError(
            "prompt is required.",
            hint="Provide a text description of the desired image.",
        )
    if len(prompt) > config.MAX_PROMPT_LENGTH:
        raise ValidationError(
            f"prompt exceeds max length ({config.MAX_PROMPT_LENGTH} chars).",
            hint="Shorten your prompt.",
        )

    negative_prompt = inp.get("negative_prompt", "")
    if not isinstance(negative_prompt, str):
        raise ValidationError("negative_prompt must be a string.")
    if len(negative_prompt) > config.MAX_PROMPT_LENGTH:
        raise ValidationError(
            f"negative_prompt exceeds max length ({config.MAX_PROMPT_LENGTH} chars).",
        )

    steps = inp.get("steps", config.DEFAULT_STEPS)
    if not isinstance(steps, int) or isinstance(steps, bool) \
            or steps < 1 or steps > config.MAX_STEPS:
        raise ValidationError(
            f"steps must be an integer between 1 and {config.MAX_STEPS}.",
            hint=f"You sent steps={steps!r}.",
        )

    cfg_scale = inp.get("cfg_scale", config.DEFAULT_CFG_SCALE)
    if not isinstance(cfg_scale, (int, float)) or isinstance(cfg_scale, bool) \
            or cfg_scale < 0 or cfg_scale > 20:
        raise ValidationError(
            "cfg_scale must be a number between 0 and 20.",
            hint=f"You sent cfg_scale={cfg_scale!r}.",
        )

    seed = inp.get("seed", config.DEFAULT_SEED)
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValidationError("seed must be an integer.")

    # Size: both-or-neither. The app only ever sends the pair (with an
    # image, to keep its aspect ratio); a plain t2i request sends neither
    # and gets the defaults.
    has_w, has_h = "width" in inp, "height" in inp
    if has_w != has_h:
        raise ValidationError(
            "width and height must be sent together.",
            hint="Send both, or neither to use the defaults.",
        )
    width = inp.get("width", config.DEFAULT_WIDTH)
    height = inp.get("height", config.DEFAULT_HEIGHT)
    if not isinstance(width, int) or not isinstance(height, int) \
            or isinstance(width, bool) or isinstance(height, bool):
        raise ValidationError("width and height must be integers.")
    if width < config.MIN_DIMENSION or height < config.MIN_DIMENSION:
        raise ValidationError(
            f"width and height must be at least {config.MIN_DIMENSION}.",
            hint=f"You sent {width}x{height}.",
        )
    if width % config.MOD_VALUE != 0 or height % config.MOD_VALUE != 0:
        snapped_w = width // config.MOD_VALUE * config.MOD_VALUE
        snapped_h = height // config.MOD_VALUE * config.MOD_VALUE
        raise ValidationError(
            f"width and height must be divisible by {config.MOD_VALUE}.",
            hint=f"You sent {width}x{height}. Try {snapped_w}x{snapped_h}.",
        )
    if width * height > config.MAX_RESOLUTION_AREA:
        raise ValidationError(
            f"Total pixels ({width}x{height}={width * height}) exceeds max ({config.MAX_RESOLUTION_AREA}).",
            hint="Use a smaller size or increase MAX_RESOLUTION_AREA.",
        )

    # Edit-mode reference size for TextEncodeQwenImage21: an explicit
    # `resolution` wins; otherwise, when the app sent width/height with
    # its image, aim at the same pixel budget; otherwise the default.
    edit_resolution = inp.get("resolution", None)
    if edit_resolution is None:
        if has_w and has_h:
            edit_resolution = int(round(math.sqrt(width * height) / 32.0) * 32)
        else:
            edit_resolution = config.DEFAULT_EDIT_RESOLUTION
    if not isinstance(edit_resolution, int) or isinstance(edit_resolution, bool) \
            or edit_resolution < 0 or edit_resolution > 4096:
        raise ValidationError(
            "resolution must be an integer between 0 and 4096.",
            hint=f"You sent resolution={edit_resolution!r}.",
        )

    output_cfg = inp.get("output", {})
    if not isinstance(output_cfg, dict):
        raise ValidationError("output must be a JSON object.")

    ignored = [p for p in config.IGNORED_PARAMS if p in inp]

    return {
        "mode": mode,
        "image_url": inp.get("image_url"),
        "image_base64": inp.get("image_base64"),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "cfg_scale": float(cfg_scale),
        "seed": seed,
        "width": width,
        "height": height,
        "edit_resolution": edit_resolution,
        "ignored_params": ignored,
        "output": {
            "bucket": output_cfg.get("bucket", config.S3_BUCKET_DEFAULT),
            "prefix": output_cfg.get("prefix", "qwen-image-comfy/"),
            "public": output_cfg.get("public", True),
            "ttl_seconds": output_cfg.get("ttl_seconds", 0),
        },
    }
