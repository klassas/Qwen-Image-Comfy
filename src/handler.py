"""RunPod Serverless handler for the Qwen-Image ComfyUI worker.

Same architecture as the MiniMax-H3 worker: this process loads no model
weights itself. It launches ComfyUI as a subprocess and talks to it over
HTTP/WebSocket (src/comfy_client.py); ComfyUI resolves and loads weights
lazily, per node, the first time a prompt graph references them. What this
handler owns is the RunPod job contract: validating input
(src/validator.py), fetching the optional edit target (src/image.py),
building the node graph (src/workflow.py), running it, and uploading the
resulting PNG (src/storage.py).

Mode is implicit: image_url/image_base64 or ref_images present -> edit
(the 2.1 model's instruction-edit graph, up to config.MAX_REF_IMAGES
references, the first being the edit base); otherwise -> text-to-image.
"""

import io
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback

import runpod

from src import config, watchdog
from src.comfy_client import ComfyClient, ComfyError, new_client_id
from src.errors import InferenceError, ValidationError, format_error
from src.image import fetch_images
from src.model_link import ModelLinkError, link_model_cache
from src.storage import upload_image
from src.validator import validate_input
from src.workflow import CLIP_NAME, OUTPUT_NODE_ID, UNET_NAME, VAE_NAME, build_workflow

logging.basicConfig(
    level=logging.INFO,
    format="%(filename)-20s:%(lineno)-4d %(asctime)s %(message)s",
)
logger = logging.getLogger("qwen-image-comfy")

_COMFY_READY = threading.Event()
_COMFY_ERROR: Exception | None = None
_comfy_process: subprocess.Popen | None = None
_client = ComfyClient(host=config.COMFY_HOST, port=config.COMFY_PORT)


# -- ComfyUI subprocess boot ───────────────────────────────────────────────────

_CUDNN_OPTOUT_NODE = '''"""Worker-installed: keep SDPA away from the cuDNN backend.

On this stack (torch cu130 + cuDNN 9) the cuDNN SDPA backend fails inside
Qwen-Image 2.1 attention with "cuDNN Frontend error: No valid execution
plans built" (seen on RTX A5000 and PRO 6000 MIG). Setting
torch.backends.cuda.enable_cudnn_sdp(False) alone is NOT enough:
ComfyUI 0.46's comfy/ops.py wraps every SDPA call in
sdpa_kernel(SDPA_BACKEND_PRIORITY, set_priority=True), and that priority
list re-enables the cuDNN backend inside the context. So this module
also removes CUDNN_ATTENTION from comfy.ops.SDPA_BACKEND_PRIORITY,
leaving FLASH -> EFFICIENT -> MATH. Convolutions keep using cuDNN.
Written by the worker's handler.py at boot; do not edit here.
"""

try:
    import torch

    torch.backends.cuda.enable_cudnn_sdp(False)
except Exception:
    pass

try:
    import comfy.ops
    from torch.nn.attention import SDPBackend

    comfy.ops.SDPA_BACKEND_PRIORITY = [
        backend
        for backend in comfy.ops.SDPA_BACKEND_PRIORITY
        if backend != SDPBackend.CUDNN_ATTENTION
    ]
except Exception:
    pass

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
'''


def _install_cudnn_optout_node() -> None:
    """Drop the cuDNN SDPA opt-out into ComfyUI's custom_nodes/ so ComfyUI
    imports it during its own startup (after torch is loaded, before any
    generation). Best-effort: a failure here must not block the boot."""
    try:
        node_dir = os.path.join(
            config.COMFY_DIR, "custom_nodes", "zz_cudnn_sdp_optout"
        )
        os.makedirs(node_dir, exist_ok=True)
        with open(os.path.join(node_dir, "__init__.py"), "w") as f:
            f.write(_CUDNN_OPTOUT_NODE)
        logger.info("Installed cuDNN SDPA opt-out custom node in %s", node_dir)
    except Exception:
        logger.exception("Could not install cuDNN SDPA opt-out node; continuing")


def _boot_comfy_bg() -> None:
    """Symlink the Model Cache snapshot into ComfyUI's models/ tree, launch
    ComfyUI, and wait for its HTTP server to answer. Runs on a background
    thread so request validation can overlap the boot, same pattern as the
    MiniMax-H3 worker."""
    global _COMFY_ERROR, _comfy_process
    try:
        logger.info("Linking Model Cache snapshot into ComfyUI models/...")
        snapshot_root = link_model_cache(os.path.join(config.COMFY_DIR, "models"))
        logger.info("Model weights linked from %s", snapshot_root)

        logger.info("Starting ComfyUI subprocess in %s", config.COMFY_DIR)
        comfy_args = [
            sys.executable, "main.py",
            "--listen", config.COMFY_HOST,
            "--port", str(config.COMFY_PORT),
        ]
        if config.COMFY_DISABLE_PINNED_MEMORY:
            comfy_args.append("--disable-pinned-memory")
        # The cuDNN SDPA opt-out must NOT be applied by importing torch
        # before ComfyUI starts (the old sitecustomize.py route): ComfyUI
        # 0.46's cuda_malloc module sets
        # PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync before torch is
        # imported, and PyTorch asserts at CUDA init when the allocator
        # backend parsed at load time (native, because torch was already
        # imported) differs from the runtime one (cudaMallocAsync) —
        # ComfyUI dies instantly at startup. A custom node is imported by
        # ComfyUI itself during startup, after torch, so the flag lands
        # before any sampling call instead.
        _install_cudnn_optout_node()
        _comfy_process = subprocess.Popen(
            comfy_args,
            cwd=config.COMFY_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        threading.Thread(target=_pipe_comfy_logs, daemon=True).start()

        _client.wait_until_alive(config.COMFY_BOOT_TIMEOUT_S)
        logger.info("ComfyUI is up.")
    except (ModelLinkError, ComfyError) as e:
        _COMFY_ERROR = e
        logger.exception("ComfyUI boot failed")
    except BaseException as e:
        _COMFY_ERROR = e
        logger.exception("ComfyUI boot failed (unexpected)")
    finally:
        _COMFY_READY.set()


def _pipe_comfy_logs() -> None:
    """Forward the ComfyUI subprocess's own stdout/stderr into this worker's
    log stream — otherwise its cold-start progress and any node errors are
    invisible in RunPod's worker log."""
    if _comfy_process is None or _comfy_process.stdout is None:
        return
    for line in _comfy_process.stdout:
        logger.info("[comfyui] %s", line.rstrip())


def _wait_for_comfy() -> None:
    _COMFY_READY.wait()
    if _COMFY_ERROR is not None:
        raise InferenceError(f"ComfyUI failed to start: {_COMFY_ERROR}")
    if _comfy_process is not None and _comfy_process.poll() is not None:
        raise InferenceError(
            f"ComfyUI process exited early (code {_comfy_process.returncode})."
        )


# -- Single job ────────────────────────────────────────────────────────────────

def _run_job(job_id: str, params: dict) -> dict:
    timings = {}

    # Fail before any GPU work if the Model Cache snapshot does not carry
    # the three files this graph loads: ComfyUI would only reject the
    # graph after the client has waited on the whole queue round-trip.
    models_dir = os.path.join(config.COMFY_DIR, "models")
    for subdir, filename in (
        ("diffusion_models", UNET_NAME),
        ("text_encoders", CLIP_NAME),
        ("vae", VAE_NAME),
    ):
        if not os.path.exists(os.path.join(models_dir, subdir, filename)):
            raise ValidationError(
                f"Model file {filename} is not available on this worker.",
                hint=(
                    "The endpoint's Model Cache snapshot lacks it. Check the "
                    "endpoint's Model field points at a repo containing "
                    f"{subdir}/{filename} (see worker-qwen/README.md)."
                ),
            )

    t0 = time.perf_counter()
    image_names: list[str] = []
    for i, img in enumerate(fetch_images(params)):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_names.append(
            _client.upload_image(buf.getvalue(), f"{job_id}_input_{i + 1}.png")
        )
    timings["preprocess_s"] = round(time.perf_counter() - t0, 3)

    seed = params["seed"]
    if seed < 0:
        import random
        seed = random.randint(0, 2**32 - 1)

    workflow = build_workflow(
        prompt=params["prompt"],
        negative_prompt=params["negative_prompt"],
        width=params["width"],
        height=params["height"],
        seed=seed,
        steps=params["steps"],
        cfg=params["cfg_scale"],
        image_names=image_names or None,
        edit_resolution=params["edit_resolution"],
        filename_prefix=f"qwen_image/{job_id}",
    )

    client_id = new_client_id()
    t0 = time.perf_counter()
    try:
        prompt_id = _client.queue(workflow, client_id)
        _client.wait_for_completion(prompt_id, client_id, config.COMFY_INFERENCE_TIMEOUT_S)
        image_bytes = _client.fetch_output(prompt_id, OUTPUT_NODE_ID)
    except ComfyError as e:
        raise InferenceError(str(e))
    timings["inference_s"] = round(time.perf_counter() - t0, 3)

    t0 = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(image_bytes)
        png_path = f.name
    try:
        out_cfg = params["output"]
        image_url = upload_image(
            local_path=png_path,
            job_id=job_id,
            bucket=out_cfg["bucket"],
            prefix=out_cfg["prefix"],
            public=out_cfg["public"],
            ttl_seconds=out_cfg["ttl_seconds"],
        )
    finally:
        try:
            os.remove(png_path)
        except OSError:
            pass
    timings["upload_s"] = round(time.perf_counter() - t0, 3)
    timings["total_s"] = round(sum(timings.values()), 3)

    meta = {
        "mode": params["mode"],
        "ref_images": len(image_names),
        "width": params["width"],
        "height": params["height"],
        "edit_resolution": params["edit_resolution"] if params["mode"] == "edit" else None,
        "seed": seed,
        "steps": params["steps"],
        "cfg_scale": params["cfg_scale"],
        "model": config.model_tag(),
        "ignored_params": params["ignored_params"],
    }

    return {
        "status": "completed",
        "image_url": image_url,
        "meta": meta,
        "timing": timings,
    }


# -- Entry point ──────────────────────────────────────────────────────────────

def handler(job: dict) -> dict:
    """RunPod serverless handler.

    Text-to-image: {"input": {"prompt": "..."}}
    Edit:          {"input": {"prompt": "...", "image_base64": "..."}}
    Multi-ref:     {"input": {"prompt": "...", "ref_images": [{"base64": "..."}]}}
    Health:        {"input": {"op": "health"}}  (SERVE_MODE=api / Pod deploy path)
    """
    job_id = job.get("id", "unknown")
    inp = job.get("input", {})

    watchdog.touch()

    if inp.get("op") == "health":
        if _COMFY_ERROR is not None:
            return {"status": "error", "error": str(_COMFY_ERROR)}
        if not _COMFY_READY.is_set():
            return {"status": "loading"}
        return {"status": "ok"}

    logger.info("Job %s started", job_id)

    try:
        _wait_for_comfy()
        params = validate_input(inp)
        result = _run_job(job_id, params)
        logger.info("Job %s completed", job_id)
        return result
    except Exception as exc:
        logger.exception("Job %s failed: %s", job_id, exc)
        return format_error(exc)


# -- Startup ───────────────────────────────────────────────────────────────────

def _probe(msg: str) -> None:
    sys.stderr.write(f"[startup] {msg}\n")
    sys.stderr.flush()


try:
    _probe("entrypoint reached")
    threading.Thread(target=_boot_comfy_bg, name="comfy-boot", daemon=True).start()

    # No-op unless SERVE_MODE=api and RUNPOD_POD_ID/RUNPOD_API_KEY are present.
    watchdog.start()

    if config.SERVE_MODE == "api":
        sys.argv += [
            "--rp_serve_api",
            "--rp_api_host", "0.0.0.0",
            "--rp_api_port", config.POD_API_PORT,
        ]
        _probe(f"SERVE_MODE=api: rp_serve_api on port {config.POD_API_PORT}")
except BaseException as _e:
    sys.stderr.write(f"[startup] FATAL during warm-up: {type(_e).__name__}: {_e}\n")
    traceback.print_exc(file=sys.stderr)
    sys.stderr.flush()
    raise

# KEEP THIS CALL AT MODULE LEVEL, UNINDENTED — RunPod's GitHub integration
# statically scans the repo for it and refuses to deploy ("handler not
# found") when it is wrapped in a try/except or a function.
runpod.serverless.start({"handler": handler})
