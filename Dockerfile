FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# Model Cache mounts the endpoint's Model repo under
# /runpod-volume/huggingface-cache/hub/ but does NOT set HF_HOME for us —
# confirmed on a live worker (both HF_HOME and HF_HUB_CACHE unset). The
# Flux/Krea/Wan2.2 workers set it here themselves; same here.
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    HF_HOME=/runpod-volume/huggingface-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ComfyUI itself, via comfy-cli (its torch is replaced with a cu130 build in
# the next step).
#
# Version floor: TextEncodeQwenImage21 / QwenImage21Cache (Qwen-Image 2.1
# support) are absent from 0.36.0 — the pin the MiniMax-H3 worker uses —
# and present since at least 0.38.0 (checked against Comfy-Org/ComfyUI
# tags). Pinned to a specific release rather than "latest" so a future
# ComfyUI release cannot change node behavior under this worker without a
# deliberate version bump here. Bump only after re-verifying
# src/workflow.py's node graph against the new release's node schemas.
ARG COMFYUI_VERSION=0.46.2
ARG COMFY_DIR=/opt/ComfyUI
ENV COMFY_DIR=${COMFY_DIR}
RUN pip install --no-cache-dir comfy-cli && \
    comfy --workspace ${COMFY_DIR} --skip-prompt install \
        --version ${COMFYUI_VERSION} --cuda-version 12.8 --nvidia && \
    test -f ${COMFY_DIR}/main.py

# Swap torch for a cu130 build. On cu128 torch, ComfyUI's comfy_kitchen
# disables its CUDA and Triton kernel backends ("You need pytorch with cu130
# or higher to use optimized CUDA operations") and runs int8_convrot
# weights — both of this worker's big files are int8_convrot — through the
# eager fallback. torchaudio's newest cu130 build is 2.11.0, so the trio is
# pinned to the matching 2.11 set. cu130 wheels bundle their own CUDA
# runtime but need a host driver that supports CUDA 13.0 (>= 580) —
# restrict the endpoint's allowed CUDA versions accordingly.
ARG TORCH_VERSION=2.11.0
ARG TORCHAUDIO_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0
# Explicit +cu130 local versions (PyPI also publishes torch==2.11.0, as a
# different CUDA build); no --force-reinstall, so ComfyUI's other pinned deps
# are only touched if torch actually requires it.
RUN python -m pip install --no-cache-dir --timeout 300 --retries 5 \
        --index-url https://download.pytorch.org/whl/cu130 \
        --extra-index-url https://pypi.org/simple \
        torch==${TORCH_VERSION}+cu130 torchaudio==${TORCHAUDIO_VERSION}+cu130 \
        torchvision==${TORCHVISION_VERSION}+cu130 && \
    python -c "import torch, torchaudio, torchvision; \
print(f'torch={torch.__version__} cuda={torch.version.cuda} torchaudio={torchaudio.__version__} torchvision={torchvision.__version__}'); \
assert torch.version.cuda and torch.version.cuda.startswith('13'), torch.version.cuda"

# This worker's own deps (runpod, requests, websocket-client for
# src/comfy_client.py, boto3 for src/storage.py) — separate from whatever
# comfy-cli installed into the same environment for ComfyUI itself.
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --timeout 180 --retries 5 \
    -r requirements.txt

# Application code
COPY src/ src/

# Verify imports at build time. handler.py is excluded — importing it starts
# the ComfyUI subprocess and blocks on model linking. Everything else is
# pure Python with no GPU/process dependency, so any dependency mismatch or
# typo fails the build here instead of at the next worker cold start.
RUN python -c "\
from src.config import MODEL_TAG, HF_REPO, COMFY_DIR, model_tag; \
from src.validator import validate_input; \
from src.workflow import build_workflow, OUTPUT_NODE_ID, UNET_NAME, CLIP_NAME, VAE_NAME; \
from src.comfy_client import ComfyClient, ComfyError; \
from src.model_link import link_model_cache, ModelLinkError; \
from src.image import fetch_image; \
from src.storage import upload_image; \
from src.errors import format_error; \
from src.security import validate_url; \
import runpod, boto3, websocket; \
wf = build_workflow(prompt='a cat', width=1024, height=1024, seed=1); \
assert OUTPUT_NODE_ID in wf, wf.keys(); \
assert wf['sampler']['inputs']['latent_image'] == ['empty_latent', 0], wf['sampler']; \
assert wf['encode']['class_type'] == 'TextEncodeQwenImage21', wf['encode']; \
wf = build_workflow(prompt='make it blue', seed=1, image_name='in.png'); \
assert wf['encode']['inputs']['images.image_1'] == ['load_image', 0], wf['encode']; \
assert wf['sampler']['inputs']['latent_image'] == ['encode', 2], wf['sampler']; \
wf = build_workflow(prompt='combine refs', seed=1, image_names=['a.png', 'b.png', 'c.png']); \
assert wf['encode']['inputs']['images.image_1'] == ['load_image_1', 0], wf['encode']; \
assert wf['encode']['inputs']['images.image_3'] == ['load_image_3', 0], wf['encode']; \
assert wf['sampler']['inputs']['latent_image'] == ['encode', 2], wf['sampler']; \
wf = build_workflow(prompt='lora', extra_loras=[('a.safetensors', 0.8), None, ('b.safetensors', 1.0)]); \
assert wf['lora_extra']['inputs']['model'] == ['unet_loader', 0], wf['lora_extra']; \
assert wf['model_cache']['inputs']['model'] == ['lora_extra_3', 0], wf['model_cache']; \
assert wf['sampler']['inputs']['model'] == ['model_cache', 0], wf['sampler']; \
assert 'lora_extra_2' not in wf, wf.keys(); \
p = validate_input({'prompt': 'x', 'workflow': {'1': {}}, 'image_base64': 'AAAA'}); \
assert p['mode'] == 'edit' and p['ignored_params'] == ['workflow'], p; \
assert p['lora_strength'] == 0.5 and p['lora_strength_2'] == 0.5, p; \
assert len(p['ref_images']) == 1 and p['ref_images'][0].get('base64') == 'AAAA', p; \
p = validate_input({'prompt': 'x', 'ref_images': [{'base64': 'AAAA'}, {'url': 'https://example.com/a.png'}, 'BBBB']}); \
assert p['mode'] == 'edit' and len(p['ref_images']) == 3, p; \
print(f'All imports OK (tag={MODEL_TAG} model={model_tag()} repo={HF_REPO} comfy_dir={COMFY_DIR})')"

# Only relevant when SERVE_MODE=api (the on-demand Pod deploy path, see
# src/watchdog.py). Serverless (the default) never
# binds this — EXPOSE is documentation for `docker run -p`/Pod port
# mapping, not a switch.
EXPOSE 8000

CMD ["python", "-u", "src/handler.py"]
