"""Configuration and defaults for the Qwen-Image ComfyUI worker.

Mirrors the MiniMax-H3 worker's config conventions (env names, S3 block,
Pod on-demand mode) wherever the two workers share a concept, so the
endpoint setup steps and the app's expectations carry over unchanged.
Qwen-Image-specific fields (model filenames, sampler defaults) live in
src/workflow.py, same split as the H3 worker.
"""

import os


def _env_bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


# -- Model / ComfyUI process ──────────────────────────────────────────────────
# Lowercase deliberately — see src/model_link.py's REPO_ID comment. Must match
# the literal string configured in the RunPod endpoint's Model field, not the
# Hub's mixed-case canonical id. Point this at a slim fork repo holding only
# the three needed files (see README.md) or at the full Comfy-Org repo.
HF_REPO = os.getenv("HF_REPO", "comfy-org/qwen-image-2.1")
# Where the Dockerfile installs ComfyUI itself (core only, no models baked
# in — weights come from Model Cache via src/model_link.py at runtime).
COMFY_DIR = os.getenv("COMFY_DIR", "/opt/ComfyUI")
COMFY_HOST = os.getenv("COMFY_HOST", "127.0.0.1")
COMFY_PORT = int(os.getenv("COMFY_PORT", "8188"))
# Cold start: ComfyUI's own process boot (before any model weights load —
# those load lazily per-prompt) plus this worker's model_link.py symlinking.
COMFY_BOOT_TIMEOUT_S = int(os.getenv("COMFY_BOOT_TIMEOUT_S", "120"))
# Per-job ceiling: the first prompt after boot pays weight-load time
# (~17 GB of int8 weights) on top of inference, so this is generous
# relative to steady-state generation time (tens of seconds at 1024x1024).
COMFY_INFERENCE_TIMEOUT_S = int(os.getenv("COMFY_INFERENCE_TIMEOUT_S", "900"))
# A known ComfyUI regression around pinned-memory allocation has been
# reported fixed by --disable-pinned-memory. Off by default; flip via env
# var without a rebuild if cold starts show it.
COMFY_DISABLE_PINNED_MEMORY = _env_bool("COMFY_DISABLE_PINNED_MEMORY", "0")

MODEL_TAG = os.getenv("MODEL_TAG", "qwen-image-comfy")


def model_tag() -> str:
    return MODEL_TAG


# -- Inference defaults ───────────────────────────────────────────────────────
# The app sends no size for a plain text-to-image request, so these are the
# output dimensions then. 1024x1024 is Qwen-Image 2.1's own 1-megapixel
# default in the official ComfyUI template.
DEFAULT_WIDTH = int(os.getenv("DEFAULT_WIDTH", "1024"))
DEFAULT_HEIGHT = int(os.getenv("DEFAULT_HEIGHT", "1024"))
# Official template values: 25 steps, cfg 1 (the model is guidance-tuned;
# the negative prompt only bites when cfg is raised), euler / simple.
DEFAULT_STEPS = int(os.getenv("DEFAULT_STEPS", "25"))
DEFAULT_CFG_SCALE = float(os.getenv("DEFAULT_CFG_SCALE", "1.0"))
DEFAULT_SEED = -1
# TextEncodeQwenImage21's `resolution`: in edit mode the reference image is
# resized to about resolution x resolution pixels (aspect preserved,
# multiples of 32). When the app sends width/height with its image, the
# validator derives this from their area instead.
DEFAULT_EDIT_RESOLUTION = int(os.getenv("DEFAULT_EDIT_RESOLUTION", "1024"))

MOD_VALUE = 32

# -- Limits ───────────────────────────────────────────────────────────────────
# The app composes long tile-based prompts (character/scene blocks), so the
# prompt ceiling is much higher than the H3 worker's 2000.
MAX_PROMPT_LENGTH = 8000
MAX_STEPS = 100
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 8192 * 8192
# Reference images per job (the ref_images field). Qwen-Image 2.1's
# encode node accepts up to 16, but every reference costs vision tokens
# and the app's use case (character + outfit + pose references) is
# served by 4, so the worker caps it there.
MAX_REF_IMAGES = 4
# Qwen-Image 2.1 supports direct output up to 2048x2048. On a 24 GB GPU
# 1024x1024 is the comfortable size; the ceiling only guards against
# absurd requests, it is not a recommendation.
MAX_RESOLUTION_AREA = int(os.getenv("MAX_RESOLUTION_AREA", str(2048 * 2048)))
MIN_DIMENSION = 128
IMAGE_DOWNLOAD_TIMEOUT = 10
UPLOAD_TIMEOUT = 60

# -- Extra LoRAs (optional, up to 3 slots) ────────────────────────────────────
# Same mechanism as the MiniMax H3 / H3 Image workers: slots are
# configured by env vars, files are fetched at boot, and the app steers
# each slot per request with lora_strength{,_2,_3} (0 = slot skipped for
# that job). EXTRA_LORA_KEY{,_2,_3} are object keys in the default S3
# bucket; EXTRA_LORA_URL{,_2,_3} is the alternative source for a slot (a
# set URL wins over that slot's key). Changing LoRAs needs only an env
# edit and a worker restart, never a rebuild.
EXTRA_LORA_KEY = os.getenv("EXTRA_LORA_KEY", "")
EXTRA_LORA_STRENGTH = float(os.getenv("EXTRA_LORA_STRENGTH", "0.5"))
EXTRA_LORA_KEY_2 = os.getenv("EXTRA_LORA_KEY_2", "")
EXTRA_LORA_STRENGTH_2 = float(os.getenv("EXTRA_LORA_STRENGTH_2", "0.5"))
EXTRA_LORA_KEY_3 = os.getenv("EXTRA_LORA_KEY_3", "")
EXTRA_LORA_STRENGTH_3 = float(os.getenv("EXTRA_LORA_STRENGTH_3", "0.5"))
EXTRA_LORAS = [
    (EXTRA_LORA_KEY, EXTRA_LORA_STRENGTH),
    (EXTRA_LORA_KEY_2, EXTRA_LORA_STRENGTH_2),
    (EXTRA_LORA_KEY_3, EXTRA_LORA_STRENGTH_3),
]
EXTRA_LORA_URL = os.getenv("EXTRA_LORA_URL", "")
EXTRA_LORA_URL_2 = os.getenv("EXTRA_LORA_URL_2", "")
EXTRA_LORA_URL_3 = os.getenv("EXTRA_LORA_URL_3", "")
EXTRA_LORA_URLS = [EXTRA_LORA_URL, EXTRA_LORA_URL_2, EXTRA_LORA_URL_3]

# Fields the app sends that this worker accepts and drops: `workflow` is
# the placeholder asset the app still ships for the old third-party Qwen
# endpoint — this worker builds its graph in code (src/workflow.py), same
# as the H3 worker does for its model.
IGNORED_PARAMS = ("workflow",)

# -- S3 / object storage ──────────────────────────────────────────────────────
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "")
S3_REGION = os.getenv("S3_REGION", "auto")
S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID", "")
S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY", "")
S3_BUCKET_DEFAULT = os.getenv("S3_BUCKET_DEFAULT", "")
S3_PUBLIC_BASE_URL = os.getenv("S3_PUBLIC_BASE_URL", "")

# -- On-demand Pod mode (SERVE_MODE=api, see src/watchdog.py) ─────────────────
SERVE_MODE = os.getenv("SERVE_MODE", "serverless")
POD_API_PORT = os.getenv("POD_API_PORT", "8000")
POD_IDLE_TIMEOUT_S = int(os.getenv("POD_IDLE_TIMEOUT_S", "180"))
POD_MAX_LIFETIME_S = int(os.getenv("POD_MAX_LIFETIME_S", "10800"))
RUNPOD_POD_ID = os.getenv("RUNPOD_POD_ID", "")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
