# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  LTX-2.3 PRO — Modified Infinite Flow Engine v1.0                          ║
# ║  Integrates: LTX 2.3 models + Infinite Flow Engine + PRO features          ║
# ║  Base engine: LTX-2.3 22B Dev GGUF Q4_K_M + Distilled LoRA (Colab T4 safe)║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# CELL ORDER:
#   Cell 1 — Environment setup & custom nodes   (run once per session)
#   Cell 2 — Model downloads                    (run once, skip if cached)
#   Cell 3 — Imports, helpers & engine classes   (run every session)
#   Cell 4 — EasyPrompt & Vision configuration  (edit to taste)
#   Cell 5 — Character & LoRA configuration     (edit per character)
#   Cell 6 — Video generation configuration     (edit per video)
#   Cell 7 — Define generate_clip()             (run once per session)
#   Cell 8 — InfiniteFlowEngine + Storyboard    (optional)
#   Cell 9 — Run                                (re-run for each clip)


# ══════════════════════════════════════════════════════════════════════════════
# CELL 1  ─  ENVIRONMENT SETUP
# ══════════════════════════════════════════════════════════════════════════════

# @title  { "single-column": true }
# @markdown ## 1. Prepare Environment & Install Custom Nodes
# @markdown Clones current ComfyUI (comfyanonymous) plus required custom nodes.

# ── Base Python packages ──────────────────────────────────────────────────────
get_ipython().system("pip install torch torchvision torchaudio")

os.chdir("/content")
from IPython.display import clear_output
clear_output()

get_ipython().system("pip install -q torchsde einops diffusers accelerate nest_asyncio")
get_ipython().system("pip install -q av spandrel albumentations onnx opencv-python onnxruntime")
get_ipython().system("pip install -q imageio imageio-ffmpeg")
get_ipython().system("pip install -q 'transformers>=4.43.0' accelerate qwen-vl-utils huggingface_hub")

# ── ComfyUI (current mainline — NOT pinned Isi-dev branch) ───────────────────
get_ipython().system("git clone https://github.com/comfyanonymous/ComfyUI")
get_ipython().system("pip install -r /content/ComfyUI/requirements.txt -q")
clear_output()

# ── Custom nodes ──────────────────────────────────────────────────────────────
os.chdir("/content/ComfyUI/custom_nodes")

get_ipython().system("git clone https://github.com/kijai/ComfyUI-KJNodes")
get_ipython().system("git clone https://github.com/city96/ComfyUI-GGUF")
get_ipython().system("git clone https://github.com/Lightricks/ComfyUI-LTXVideo")

# Install node requirements
os.chdir("/content/ComfyUI/custom_nodes/ComfyUI-KJNodes")
get_ipython().system("pip install -r requirements.txt -q")

os.chdir("/content/ComfyUI/custom_nodes/ComfyUI-GGUF")
get_ipython().system("pip install -r requirements.txt -q")

os.chdir("/content/ComfyUI/custom_nodes/ComfyUI-LTXVideo")
get_ipython().system("pip install -r requirements.txt -q 2>/dev/null || true")

# ── System tools ──────────────────────────────────────────────────────────────
import subprocess
import os

def install_apt_packages():
    packages = ["aria2", "ffmpeg"]
    try:
        subprocess.run(["apt-get", "-y", "install", "-qq"] + packages,
                       check=True, capture_output=True)
        print("apt packages installed")
    except subprocess.CalledProcessError as e:
        print(f"apt error: {e.stderr.decode().strip() or 'unknown'}")

print("Installing apt packages...")
install_apt_packages()

# ── Final setup ───────────────────────────────────────────────────────────────
os.chdir("/content/ComfyUI")
import os, sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, "/content/ComfyUI")

clear_output()
print("Environment setup complete.")
print("   Custom nodes installed:")
print("   - ComfyUI-KJNodes    (kijai)")
print("   - ComfyUI-GGUF       (city96)")
print("   - ComfyUI-LTXVideo   (Lightricks)")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 2  ─  MODEL DOWNLOADS
# ══════════════════════════════════════════════════════════════════════════════

# @title  { "single-column": true }
# @markdown ## 2. Download All Model Weights (LTX 2.3)
# @markdown Uses aria2c (fast parallel download). Skips files already cached.

import os, subprocess
from pathlib import Path

def model_download(url: str, dest_dir: str, filename: str = None,
                   silent: bool = True) -> str:
    """aria2c download with skip-if-cached logic. Returns local filename."""
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = url.split("/")[-1].split("?")[0]
    dest = os.path.join(dest_dir, filename)
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        print(f"  cached: {filename}")
        return filename
    cmd = ["aria2c", "--console-log-level=error",
           "-c", "-x", "16", "-s", "16", "-k", "1M",
           "-d", dest_dir, "-o", filename]
    if silent:
        cmd += ["--summary-interval=0", "--quiet"]
        print(f"  downloading {filename}...", end=" ", flush=True)
    cmd.append(url)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\n  FAILED: {result.stderr.strip()}")
        return False
    if silent:
        print("done.")
    return filename


def download_civitai_model(civitai_link, civitai_token, folder="/content/ComfyUI/models/loras"):
    """Download a model from Civitai with token authentication."""
    import time
    os.makedirs(folder, exist_ok=True)
    try:
        model_id = civitai_link.split("/models/")[1].split("?")[0]
    except IndexError:
        raise ValueError("Invalid Civitai URL format.")
    civitai_url = f"https://civitai.com/api/download/models/{model_id}?type=Model&format=SafeTensor"
    if civitai_token:
        civitai_url += f"&token={civitai_token}"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"model_{timestamp}.safetensors"
    full_path = os.path.join(folder, filename)
    download_command = f'wget --max-redirect=10 --show-progress "{civitai_url}" -O "{full_path}"'
    print("Downloading from Civitai...")
    os.system(download_command)
    if os.path.exists(full_path) and os.path.getsize(full_path) > 0:
        print(f"LoRA downloaded: {full_path}")
    else:
        print(f"LoRA download failed: {full_path}")
    return filename


def download_lora(link, folder="/content/ComfyUI/models/loras", civitai_token=None):
    """Download a LoRA file, auto-detecting Civitai vs huggingface URLs."""
    if "civitai.com" in link.lower():
        if not civitai_token:
            raise ValueError("Civitai token is required for Civitai downloads")
        return download_civitai_model(link, civitai_token, folder)
    else:
        return model_download(link, folder)


# ── Source base URLs ──────────────────────────────────────────────────────────
UNSLOTH  = "https://huggingface.co/unsloth"
COMFYORG = "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files"
LIGHTRIX = "https://huggingface.co/Lightricks"
KIJAI23  = "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main"

UNET_D   = "/content/ComfyUI/models/unet"
TE_D     = "/content/ComfyUI/models/text_encoders"
VAE_D    = "/content/ComfyUI/models/vae"
UPD      = "/content/ComfyUI/models/latent_upscale_models"
LORA_DIR = "/content/ComfyUI/models/loras"

print("-- Core LTX 2.3 model downloads --")

# ── UNet: LTX 2.3 22B Dev GGUF Q4_K_M ────────────────────────────────────────
dit_model = model_download(
    f"{UNSLOTH}/LTX-2.3-GGUF/resolve/main/ltx-2.3-22b-dev-Q4_K_M.gguf",
    UNET_D)

# ── Text encoders ─────────────────────────────────────────────────────────────
# Primary: Gemma fp8 + LTX 2.3 embeddings connector
text_encoder_model = model_download(
    f"{COMFYORG}/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors",
    TE_D)

# LTX 2.3 embeddings connector (pairs with safetensors Gemma)
text_encoder_connector = model_download(
    f"{UNSLOTH}/LTX-2.3-GGUF/resolve/main/text_encoders/ltx-2.3-22b-dev_embeddings_connectors.safetensors",
    TE_D)

# GGUF Gemma fallback pair
text_encoder_gguf_model = model_download(
    f"{UNSLOTH}/gemma-3-12b-it-qat-GGUF/resolve/main/gemma-3-12b-it-qat-UD-Q4_K_XL.gguf",
    TE_D)
text_encoder_mmproj = model_download(
    f"{UNSLOTH}/gemma-3-12b-it-qat-GGUF/resolve/main/mmproj-BF16.gguf",
    TE_D)

# ── VAEs ──────────────────────────────────────────────────────────────────────
vae_model = model_download(
    f"{UNSLOTH}/LTX-2.3-GGUF/resolve/main/vae/ltx-2.3-22b-dev_video_vae.safetensors",
    VAE_D)

vae_audio_model = model_download(
    f"{UNSLOTH}/LTX-2.3-GGUF/resolve/main/vae/ltx-2.3-22b-dev_audio_vae.safetensors",
    VAE_D)

# TaeVAE preview
taeltx2_model = model_download(
    f"{KIJAI23}/vae/taeltx2_3.safetensors",
    VAE_D)

# ── Spatial upscaler ──────────────────────────────────────────────────────────
upscaler_model = model_download(
    f"{LIGHTRIX}/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
    UPD)

# ── Distilled LoRA (mandatory - always applied first) ─────────────────────────
distilled_lora_model = model_download(
    f"{LIGHTRIX}/LTX-2.3/resolve/main/ltx-2.3-22b-distilled-lora-384.safetensors",
    LORA_DIR)

# ── IC LoRAs + Camera Control LoRAs ──────────────────────────────────────────
LORA_URLS = {
    "Detailer":    f"{LIGHTRIX}/LTX-2-19b-IC-LoRA-Detailer/resolve/main/ltx-2-19b-ic-lora-detailer.safetensors",
    "Canny":       f"{LIGHTRIX}/LTX-2-19b-IC-LoRA-Canny-Control/resolve/main/ltx-2-19b-ic-lora-canny-control.safetensors",
    "Depth":       f"{LIGHTRIX}/LTX-2-19b-IC-LoRA-Depth-Control/resolve/main/ltx-2-19b-ic-lora-depth-control.safetensors",
    "Pose":        f"{LIGHTRIX}/LTX-2-19b-IC-LoRA-Pose-Control/resolve/main/ltx-2-19b-ic-lora-pose-control.safetensors",
    "Dolly-In":    f"{LIGHTRIX}/LTX-2-19b-LoRA-Camera-Control-Dolly-In/resolve/main/ltx-2-19b-lora-camera-control-dolly-in.safetensors",
    "Dolly-Out":   f"{LIGHTRIX}/LTX-2-19b-LoRA-Camera-Control-Dolly-Out/resolve/main/ltx-2-19b-lora-camera-control-dolly-out.safetensors",
    "Dolly-Left":  f"{LIGHTRIX}/LTX-2-19b-LoRA-Camera-Control-Dolly-Left/resolve/main/ltx-2-19b-lora-camera-control-dolly-left.safetensors",
    "Dolly-Right": f"{LIGHTRIX}/LTX-2-19b-LoRA-Camera-Control-Dolly-Right/resolve/main/ltx-2-19b-lora-camera-control-dolly-right.safetensors",
    "Jib-Up":      f"{LIGHTRIX}/LTX-2-19b-LoRA-Camera-Control-Jib-Up/resolve/main/ltx-2-19b-lora-camera-control-jib-up.safetensors",
    "Jib-Down":    f"{LIGHTRIX}/LTX-2-19b-LoRA-Camera-Control-Jib-Down/resolve/main/ltx-2-19b-lora-camera-control-jib-down.safetensors",
    "Static":      f"{LIGHTRIX}/LTX-2-19b-LoRA-Camera-Control-Static/resolve/main/ltx-2-19b-lora-camera-control-static.safetensors",
}

os.makedirs(LORA_DIR, exist_ok=True)
print(f"\n-- LoRA batch download ({len(LORA_URLS)} files) --")
for name, url in LORA_URLS.items():
    r = model_download(url, LORA_DIR)
    print(f"   {'OK' if r else 'FAIL'}  {name}")

# ── User LoRAs (3 slots with civitai support) ─────────────────────────────────
# @markdown ### User LoRA Downloads
download_loRA_1 = False  # @param {type:"boolean"}
lora_1_download_url = ""  # @param {"type":"string"}
download_loRA_2 = False  # @param {type:"boolean"}
lora_2_download_url = ""  # @param {"type":"string"}
download_loRA_3 = False  # @param {type:"boolean"}
lora_3_download_url = ""  # @param {"type":"string"}
token_if_civitai_url = ""  # @param {"type":"string"}

user_lora_1 = None
user_lora_2 = None
user_lora_3 = None

valid_extensions = {'.safetensors', '.ckpt', '.pt', '.pth', '.sft'}

if download_loRA_1 and lora_1_download_url:
    user_lora_1 = download_lora(lora_1_download_url, civitai_token=token_if_civitai_url)
    if user_lora_1 and not any(user_lora_1.lower().endswith(ext) for ext in valid_extensions):
        user_lora_1 = None

if download_loRA_2 and lora_2_download_url:
    user_lora_2 = download_lora(lora_2_download_url, civitai_token=token_if_civitai_url)
    if user_lora_2 and not any(user_lora_2.lower().endswith(ext) for ext in valid_extensions):
        user_lora_2 = None

if download_loRA_3 and lora_3_download_url:
    user_lora_3 = download_lora(lora_3_download_url, civitai_token=token_if_civitai_url)
    if user_lora_3 and not any(user_lora_3.lower().endswith(ext) for ext in valid_extensions):
        user_lora_3 = None

print("\nAll models ready.")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 3  ─  IMPORTS, HELPERS & ENGINE CLASSES
# ══════════════════════════════════════════════════════════════════════════════

# @title  { "single-column": true }
# @markdown ## 3. Imports, Helpers & Engine Classes
# @markdown All utility functions, VisionDescribeEngine, EasyPromptEngine,
# @markdown CharacterBible, and PRO feature functions.

import os, sys, gc, re, json, time, shutil, warnings, subprocess, asyncio
import numpy as np
import torch
import cv2
from PIL import Image
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any, Union, Sequence, Mapping
from base64 import b64encode
from IPython.display import display, HTML, Image as IPImage, clear_output
from google.colab import files

warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, "/content/ComfyUI")

from nodes import NODE_CLASS_MAPPINGS, LoraLoaderModelOnly
import folder_paths


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping."""
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def tensor_width_height(image):
    """Return (width, height) for a ComfyUI image tensor in NHWC or HWC format.

    Newer ComfyUI/LTXVideo builds may not register the old GetImageSize node, so
    use tensor shape directly instead of depending on that optional custom node.
    """
    if isinstance(image, (tuple, list)):
        image = get_value_at_index(image, 0)
    if image.ndim == 4:      # (N, H, W, C)
        return int(image.shape[2]), int(image.shape[1])
    if image.ndim == 3:      # (H, W, C)
        return int(image.shape[1]), int(image.shape[0])
    raise ValueError(f"Unsupported image tensor shape: {getattr(image, 'shape', None)}")


def load_audio_vae_compat(vae_name):
    """Load the LTX audio VAE across ComfyUI/KJNodes versions."""
    if "VAELoaderKJ" in NODE_CLASS_MAPPINGS:
        print("Loading audio VAE with VAELoaderKJ...")
        loader = NODE_CLASS_MAPPINGS["VAELoaderKJ"]()
        return loader.load_vae(vae_name=vae_name, device="main_device", weight_dtype="fp16")
    if "VAELoader" in NODE_CLASS_MAPPINGS:
        print("VAELoaderKJ not found; loading audio VAE with built-in VAELoader...")
        loader = NODE_CLASS_MAPPINGS["VAELoader"]()
        return loader.load_vae(vae_name=vae_name)
    candidates = sorted(k for k in NODE_CLASS_MAPPINGS.keys() if "vae" in k.lower() and "load" in k.lower())
    raise KeyError("No compatible VAE loader found. Available: " + ", ".join(candidates))


# ── VRAM management ───────────────────────────────────────────────────────────
def cleanup_memory(verbose: bool = False):
    """Enhanced memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
    gc.collect()
    if verbose:
        _print_vram()


def purge_vram(label: str = ""):
    """Purge VRAM after model loading phases."""
    tag = f" [{label}]" if label else ""
    if "LayerUtility: PurgeVRAM V2" in NODE_CLASS_MAPPINGS:
        try:
            node = NODE_CLASS_MAPPINGS["LayerUtility: PurgeVRAM V2"]()
            fn = getattr(node, node.FUNCTION)
            fn()
            print(f"   VRAM purged via PurgeVRAM V2{tag}")
            return
        except Exception:
            pass
    cleanup_memory()
    print(f"   VRAM cleared via torch.cuda.empty_cache{tag}")


def _print_vram():
    if not torch.cuda.is_available():
        return
    used = torch.cuda.memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    filled = int(20 * used / total) if total > 0 else 0
    bar = "#" * filled + "." * (20 - filled)
    print(f"   VRAM [{bar}] {used:.1f}/{total:.1f} GB")


# ── Tensor / image conversion ─────────────────────────────────────────────────
def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL -> ComfyUI NHWC float tensor."""
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """ComfyUI NHWC tensor -> PIL."""
    if t.ndim == 4:
        t = t[0]
    return Image.fromarray((t.cpu().numpy() * 255).clip(0, 255).astype(np.uint8), "RGB")


def load_image_tensor(path: str) -> Optional[torch.Tensor]:
    """Load an image file as a ComfyUI NHWC tensor. Returns None if missing."""
    if not path or not os.path.exists(path):
        return None
    return pil_to_tensor(Image.open(path).convert("RGB"))


def get_last_frame_tensor(video_path: str) -> Optional[torch.Tensor]:
    """Extract last frame of a video as NHWC float tensor shape (1,H,W,3)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n == 0:
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, n - 1)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(frame).float().unsqueeze(0) / 255.0


# ── Video helpers ─────────────────────────────────────────────────────────────
def display_video(path: str):
    """Display video inline in Colab."""
    if not path or not os.path.exists(path):
        print(f"   Not found: {path}")
        return
    data = b64encode(open(path, "rb").read()).decode()
    display(HTML(
        '<video width=800 controls autoplay loop muted>'
        f'<source src="data:video/mp4;base64,{data}" type="video/mp4">'
        '</video>'
    ))


def save_video_from_components(video_obj, prefix="LTX-2.3-PRO") -> str:
    """Save a ComfyUI video object and return the output path."""
    from comfy_api.latest import Types
    w, h = video_obj.get_dimensions()
    folder, fname, ctr, _, _ = folder_paths.get_save_image_path(
        prefix, folder_paths.get_output_directory(), w, h)
    ext = Types.VideoContainer.get_extension("auto")
    path = os.path.join(folder, f"{fname}_{ctr:05}_.{ext}")
    video_obj.save_to(path, format=Types.VideoContainer("auto"),
                      codec="auto", metadata=None)
    return path


def concatenate_clips(clip_paths: List[str], output_path: str) -> str:
    """ffmpeg concat all clips into one final video."""
    list_file = "/tmp/concat_list.txt"
    with open(list_file, "w") as f:
        for p in clip_paths:
            f.write(f"file '{p}'\n")
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file,
                    "-c", "copy", output_path], check=True, capture_output=True)
    print(f"   Concatenated -> {output_path}")
    return output_path


# ── ComfyUI async node loader ─────────────────────────────────────────────────
_NODES_LOADED = False


def import_custom_nodes() -> None:
    """Load all built-in and external custom nodes in a Jupyter/Colab-safe way."""
    global _NODES_LOADED
    if _NODES_LOADED:
        return
    import nest_asyncio
    from nodes import init_builtin_extra_nodes, init_external_custom_nodes

    async def _load():
        failed = await init_builtin_extra_nodes()
        await init_external_custom_nodes()
        if failed:
            print(f"   Node failures: {failed}")

    try:
        asyncio.run(_load())
    except RuntimeError:
        nest_asyncio.apply()
        asyncio.get_event_loop().run_until_complete(_load())
    _NODES_LOADED = True


# ── Performance patches ───────────────────────────────────────────────────────
def apply_sage_attention(unet):
    """Apply PathchSageAttentionKJ if available."""
    if "PathchSageAttentionKJ" not in NODE_CLASS_MAPPINGS:
        return unet
    try:
        node = NODE_CLASS_MAPPINGS["PathchSageAttentionKJ"]()
        fn = getattr(node, node.FUNCTION)
        unet = get_value_at_index(fn(model=unet), 0)
        print("   SageAttention patch applied")
    except Exception as e:
        print(f"   SageAttention failed ({e})")
    return unet


def apply_chunk_ff(unet):
    """Apply LTXVChunkFeedForward if available."""
    if "LTXVChunkFeedForward" not in NODE_CLASS_MAPPINGS:
        return unet
    try:
        node = NODE_CLASS_MAPPINGS["LTXVChunkFeedForward"]()
        fn = getattr(node, node.FUNCTION)
        unet = get_value_at_index(fn(model=unet), 0)
        print("   ChunkFeedForward patch applied")
    except Exception as e:
        print(f"   ChunkFeedForward failed ({e})")
    return unet


# ══════════════════════════════════════════════════════════════════════════════
# VISION DESCRIBE ENGINE
# Standalone Qwen2.5-VL wrapper. Loads -> describes -> unloads to free VRAM.
# ══════════════════════════════════════════════════════════════════════════════

class VisionDescribeEngine:
    """Analyses an image and returns a 100-130 word scene description."""

    MODEL_OPTIONS = {
        "3B-fast": "huihui-ai/Qwen2.5-VL-3B-Instruct-abliterated",
        "7B-nsfw": "prithivMLmods/Qwen2.5-VL-7B-Abliterated-Caption-it",
    }

    PROMPT = (
        "Describe this image in one paragraph of plain sentences, 100-130 words. "
        "Start with 'Style: photorealistic' or 'Style: anime' or 'Style: 3D animation' etc. "
        "The FIRST sentence about any person MUST explicitly state ethnicity and skin tone "
        "using plain terms: 'a Black man', 'a white woman', 'a South Asian man'. "
        "Include age, hair colour and style, body type, clothing or nude state, pose, "
        "camera framing, angle, lighting, time of day, and setting. "
        "One flowing paragraph, no bullets, no labels. "
        "If no person, describe environment, objects, lighting, mood."
    )

    def __init__(self, model_key: str = "3B-fast", offline: bool = False):
        self.model_key = model_key
        self.offline = offline

    def describe(self, image: Union[Image.Image, torch.Tensor]) -> str:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from huggingface_hub import snapshot_download
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            raise ImportError("[VisionDescribe] pip install qwen-vl-utils")

        if isinstance(image, torch.Tensor):
            image = tensor_to_pil(image)

        hf_id = self.MODEL_OPTIONS[self.model_key]
        if not self.offline:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            try:
                source = snapshot_download(hf_id)
            except Exception:
                source = hf_id
        else:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            source = hf_id

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"   [VisionDescribe] Loading {self.model_key} ({image.size}) ...")
        processor = AutoProcessor.from_pretrained(source, local_files_only=self.offline)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            source, device_map="auto", torch_dtype=dtype,
            local_files_only=self.offline)
        model.eval()

        messages = [
            {"role": "system", "content":
             "You are an image analysis tool. Describe exactly what you see in plain prose."},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": self.PROMPT},
            ]},
        ]
        text_in = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        img_in, vid_in = process_vision_info(messages)
        inputs = processor(text=[text_in], images=img_in, videos=vid_in,
                           padding=True, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        tok = processor.tokenizer
        stop_ids = [i for i in [tok.eos_token_id] if i is not None]
        for s in ["<|im_end|>", "<|endoftext|>"]:
            ids = tok.encode(s, add_special_tokens=False)
            if len(ids) == 1 and ids[0] not in stop_ids:
                stop_ids.append(ids[0])

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=210, temperature=0.3,
                                 do_sample=True, top_p=0.9,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id,
                                 eos_token_id=stop_ids)
        desc = tok.decode(out[0][input_len:], skip_special_tokens=True).strip()
        del out, inputs, model, processor
        cleanup_memory()
        print(f"   [VisionDescribe] Done. {len(desc.split())} words.")
        return desc


# ══════════════════════════════════════════════════════════════════════════════
# EASY PROMPT ENGINE
# Full cinematic LLM expansion with pacing, bible lock, dialogue control.
# ══════════════════════════════════════════════════════════════════════════════

_NEG_BASE = (
    "blurry, out of focus, low quality, worst quality, jpeg artifacts, "
    "static, no motion, frozen, duplicate, watermark, text, signature, "
    "poorly drawn, bad anatomy, deformed, disfigured, extra limbs, "
    "missing limbs, overexposed, underexposed, grainy, noise, flickering"
)


def _build_neg(result: str, user_input: str) -> str:
    c = (result + " " + user_input).lower()
    extras = []
    if any(w in c for w in ["indoor", "room", "interior", "bedroom", "kitchen", "office"]):
        extras.append("harsh outdoor lighting, direct sunlight")
    elif any(w in c for w in ["outdoor", "street", "beach", "forest", "park"]):
        extras.append("studio background, indoor lighting")
    if any(w in c for w in ["close-up", "close up", "portrait", "headshot"]):
        extras.append("wide angle distortion, fish eye")
    elif any(w in c for w in ["wide shot", "wide angle", "aerial"]):
        extras.append("close-up, portrait crop")
    if any(w in c for w in ["night", "dark", "moonlight", "dimly lit", "candlelight"]):
        extras.append("overexposed, bright daylight, blown highlights")
    elif any(w in c for w in ["daylight", "sunny", "golden hour", "bright"]):
        extras.append("underexposed, dark shadows, black crush")
    if any(w in c for w in ["two women", "two men", "two people", "couple", "both"]):
        extras.append("merged bodies, fused figures, incorrect number of people")
    return ", ".join([_NEG_BASE] + extras)


class EasyPromptEngine:
    """
    Expands a simple story beat into a dense cinematic LTX-2 prompt.
    Loads the LLM, generates, cleans output, then unloads to free VRAM.
    """

    MODELS = {
        "8B":  "mlabonne/NeuralDaredevil-8B-abliterated",
        "3B":  "huihui-ai/Llama-3.2-3B-Instruct-abliterated",
        "14B": "huihui-ai/Huihui-Qwen3-14B-abliterated-v2",
    }

    SYSTEM_PROMPT = """You are a cinematic prompt writer for LTX-2, an AI video generation model. Expand the user's idea into a rich, video-ready prompt.

PRIORITY ORDER:
1. Video style & genre (slow-burn thriller, documentary, editorial, action blockbuster)
2. Camera angle & shot type (low-angle close-up, bird's-eye wide, Dutch angle medium)
3. Character description - age MUST be a specific number (e.g. "a 28-year-old woman"), body type, hair, skin, clothing. Use exact words from the user.
4. Scene & environment (location, time of day, lighting, colour palette, atmosphere)
5. Action & motion - continuous present-tense sequence.
6. Camera movement - prose only, no screenplay brackets like (HOLD) or (DOWN 10).
7. Audio - max 2 ambient sounds active at once, woven as prose. Dialogue as inline prose with attribution, never as [DIALOGUE:] tags.

RULES:
- Present tense throughout.
- 8-12 sentences of dense flowing prose - no bullet lists.
- Fill the full token budget. Do not stop early.
- Output ONLY the expanded prompt. No preamble. No trailing notes. No commentary."""

    _CLEAN_RE = [
        (re.compile(r"<think>.*?</think>", re.DOTALL), ""),
        (re.compile(r"^(Sure!?|Certainly!?|Here(?:'s| is).*?:)[^\n]*\n?", re.IGNORECASE), ""),
        (re.compile(r"\s*(assistant|user|system|<\|[^|>]*\|>)\s*$", re.IGNORECASE), ""),
        (re.compile(r"\s*\n+Note:.*$", re.DOTALL), ""),
        (re.compile(r"\s*\(Note:.*$", re.DOTALL | re.IGNORECASE), ""),
        (re.compile(r"\s*(\([^)]{5,120}\)\s*){2,}$", re.DOTALL), ""),
        (re.compile(r"\s*\n+(Please let me know|Let me revise|Confirmed\.|Output ends|"
                    r"Done\.|I hope|Thank you|No further).*$",
                    re.DOTALL | re.IGNORECASE), ""),
        (re.compile(r"\n{3,}"), "\n\n"),
    ]

    def __init__(self, model_size: str = "8B", offline: bool = False,
                 keep_loaded: bool = False):
        self.model_size = model_size
        self.offline = offline
        self.keep_loaded = keep_loaded
        self._tok = None
        self._model = None
        self._loaded_key = None

    def _load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from huggingface_hub import snapshot_download
        key = self.model_size
        if self._model is not None and self._loaded_key == key:
            return
        if self._model is not None:
            self._unload()
        hf_id = self.MODELS[key]
        if not self.offline:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            try:
                source = snapshot_download(hf_id)
            except Exception:
                source = hf_id
        else:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            source = hf_id
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"   [EasyPrompt] Loading {key} ...")
        self._tok = AutoTokenizer.from_pretrained(source, local_files_only=self.offline)
        self._model = AutoModelForCausalLM.from_pretrained(
            source, device_map="auto", torch_dtype=dtype,
            trust_remote_code=True, local_files_only=self.offline)
        self._model.config.use_cache = True
        self._model.eval()
        self._loaded_key = key
        print(f"   [EasyPrompt] Loaded.")

    def _unload(self):
        if self._model is not None:
            try:
                self._model.to("cpu")
            except Exception:
                pass
        self._model = None
        self._tok = None
        self._loaded_key = None
        cleanup_memory()
        print("   [EasyPrompt] VRAM cleared.")

    def _stop_ids(self) -> List[int]:
        delims = ["assistant", "user", "system", "<|eot_id|>", "<|end_of_turn|>",
                  "<|im_end|>", "<end_of_turn>", "[/INST]", "### Human", "### Assistant"]
        ids = [self._tok.eos_token_id]
        for s in delims:
            enc = self._tok.encode(s, add_special_tokens=False)
            if enc and enc[0] not in ids:
                ids.append(enc[0])
        return [i for i in dict.fromkeys(ids) if i is not None]

    @staticmethod
    def _clean(text: str) -> str:
        text = text.strip()
        for pattern, repl in EasyPromptEngine._CLEAN_RE:
            text = pattern.sub(repl, text)
        text = re.sub(r"\s*[\(\[]\s*$", "", text)
        return text.strip()

    def generate(
        self,
        user_input: str,
        frame_count: int = 121,
        creativity: float = 0.9,
        seed: int = -1,
        scene_context: str = "",
        lora_triggers: str = "",
        character_bible: str = "",
    ) -> tuple:
        """
        Returns (positive_prompt, negative_prompt).

        character_bible - injected as a hard [CHARACTER BIBLE - NON-NEGOTIABLE]
        block so the LLM cannot alter hair, age, clothing, or any locked attribute.
        """
        self._load()

        real_seconds = frame_count / 25.0
        action_count = max(1, min(10, round(real_seconds / 4)))
        token_budget = max(256, min(1200, action_count * 120))
        max_tokens = int(token_budget * 1.05)
        min_tokens = int(token_budget * 0.75)

        if seed != -1:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        ordinal = {2: "2nd", 3: "3rd"}.get(action_count, f"{action_count}th")
        pacing = (
            f"This clip is {real_seconds:.0f}s. Write EXACTLY {action_count} "
            f"distinct action{'s' if action_count > 1 else ''}. "
            f"HARD STOP after the {ordinal} action. "
            f"Write ~{token_budget} tokens."
        ) if action_count > 1 else (
            f"This clip is {real_seconds:.0f}s. Write EXACTLY 1 action. "
            f"HARD STOP after it. ~{token_budget} tokens."
        )

        bible_clause = ""
        if character_bible.strip():
            bible_clause = (
                f"\n[CHARACTER BIBLE - NON-NEGOTIABLE: Every character attribute below "
                f"MUST remain exactly as described. Do NOT alter hair, age, skin, "
                f"clothing, or any other attribute. This overrides any inference:\n"
                f"{character_bible.strip()}\n]"
            )

        if scene_context.strip():
            effective = (
                f"[SCENE CONTEXT FROM IMAGE - authoritative, do not contradict]\n"
                f"{scene_context.strip()}\n\n"
                f"[USER DIRECTION - action, style, mood]\n{user_input.strip()}"
            )
        else:
            effective = user_input.strip()

        lora_clause = (f"\n[LORA: Begin prompt with: {lora_triggers.strip()}]"
                       if lora_triggers.strip() else "")

        user_content = (effective + bible_clause + lora_clause
                        + f"\n[PACING: {pacing}]")

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        is_qwen3 = "Qwen3" in self.MODELS.get(self.model_size, "")
        raw = self._tok.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True,
            **({"enable_thinking": False} if is_qwen3 else {}))

        if hasattr(raw, "input_ids"):
            input_ids = raw.input_ids.to(self._model.device)
        elif isinstance(raw, dict):
            input_ids = raw["input_ids"].to(self._model.device)
        elif isinstance(raw, list):
            input_ids = torch.tensor([raw], dtype=torch.long).to(self._model.device)
        else:
            input_ids = raw.to(self._model.device)

        input_len = input_ids.shape[1]

        with torch.no_grad():
            out = self._model.generate(
                input_ids, min_new_tokens=min_tokens, max_new_tokens=max_tokens,
                temperature=creativity, do_sample=True, top_k=40, top_p=0.9,
                repetition_penalty=1.07, use_cache=True,
                pad_token_id=self._tok.eos_token_id,
                eos_token_id=self._stop_ids())

        result = self._tok.decode(out[0][input_len:], skip_special_tokens=True).strip()
        result = self._clean(result)
        result = re.sub(r'\s*[\(\[]\s*$', '', result).strip()
        del out, input_ids
        neg = _build_neg(result, user_input)

        if not self.keep_loaded:
            self._unload()

        print(f"   [EasyPrompt] Done. {len(result.split())} words generated.")
        return result, neg


# ══════════════════════════════════════════════════════════════════════════════
# CHARACTER BIBLE
# Serialisable cross-scene character consistency record.
# ══════════════════════════════════════════════════════════════════════════════

class CharacterBible:
    """
    Records named character attributes and serialises them as a prompt-injection
    block. The block is passed to EasyPromptEngine as character_bible so the
    LLM receives a hard [NON-NEGOTIABLE] constraint preventing attribute drift.

    Auto-populated from Vision Describe output on the seed image (recommended),
    or filled manually with add().
    """

    def __init__(self):
        self._chars: dict = {}

    def add(self, name: str, **attributes):
        """Manually define a character."""
        self._chars[name] = dict(attributes)

    def extract_from_description(self, name: str, description: str):
        """Store a raw Vision Describe output under a character name."""
        self._chars[name] = {"_raw": description.strip()}

    def to_prompt_block(self) -> str:
        """Returns the injection string for EasyPromptEngine."""
        if not self._chars:
            return ""
        lines = []
        for name, attrs in self._chars.items():
            if "_raw" in attrs:
                lines.append(f"CHARACTER - {name}:\n{attrs['_raw']}")
            else:
                attr_str = "; ".join(f"{k}: {v}" for k, v in attrs.items())
                lines.append(f"CHARACTER - {name}: {attr_str}")
        return "\n\n".join(lines)

    def has_characters(self) -> bool:
        return bool(self._chars)

    def names(self) -> List[str]:
        return list(self._chars.keys())

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self._chars, f, indent=2)
        print(f"   [Bible] Saved -> {path}")

    def load(self, path: str):
        with open(path) as f:
            self._chars = json.load(f)
        print(f"   [Bible] Loaded from {path}")

    def __repr__(self):
        return f"CharacterBible({self.names()})"


# ══════════════════════════════════════════════════════════════════════════════
# PRO FEATURES - Camera LoRA Mapping, Motion Guidance, Adaptive Strength
# ══════════════════════════════════════════════════════════════════════════════

CAMERA_LORA_MAPPING = {
    "dolly_forward": "ltx-2-19b-lora-camera-control-dolly-in.safetensors",
    "dolly_backward": "ltx-2-19b-lora-camera-control-dolly-out.safetensors",
    "dolly_in": "ltx-2-19b-lora-camera-control-dolly-in.safetensors",
    "dolly_out": "ltx-2-19b-lora-camera-control-dolly-out.safetensors",
    "dolly_left": "ltx-2-19b-lora-camera-control-dolly-left.safetensors",
    "dolly_right": "ltx-2-19b-lora-camera-control-dolly-right.safetensors",
    "pan_left": "ltx-2-19b-lora-camera-control-dolly-left.safetensors",
    "pan_right": "ltx-2-19b-lora-camera-control-dolly-right.safetensors",
    "tilt_up": "ltx-2-19b-lora-camera-control-jib-up.safetensors",
    "tilt_down": "ltx-2-19b-lora-camera-control-jib-down.safetensors",
    "jib_up": "ltx-2-19b-lora-camera-control-jib-up.safetensors",
    "jib_down": "ltx-2-19b-lora-camera-control-jib-down.safetensors",
    "zoom_in": "ltx-2-19b-lora-camera-control-dolly-in.safetensors",
    "zoom_out": "ltx-2-19b-lora-camera-control-dolly-out.safetensors",
    "static": "ltx-2-19b-lora-camera-control-static.safetensors",
}


def build_character_prompt_detailed(character_data):
    """Build highly detailed character description for consistency."""
    char_name = character_data["name"]
    appearance = character_data.get("detailed_appearance", {})
    if not appearance:
        return f"{char_name}: {character_data.get('desc', '')}"
    prompt = f"{char_name}: "
    prompt += f"{appearance.get('face', '')}, "
    prompt += f"{appearance.get('hair', '')}, "
    prompt += f"wearing {appearance.get('clothing', '')}, "
    prompt += f"{appearance.get('build', '')}, {appearance.get('skin_tone', '')}, "
    prompt += f"{appearance.get('accessories', '')}. "
    prompt += (f"ALWAYS MAINTAIN: {char_name} has "
               f"{appearance.get('face', '').split(',')[0]}, "
               f"{appearance.get('hair', '').split(',')[0]}, "
               f"{appearance.get('clothing', '').split(',')[0]}. ")
    return prompt


def get_character_consistency_prefix(scene_json):
    """Generate character consistency prefix for ALL prompts."""
    char_prompts = []
    for char in scene_json.get("main_characters", []):
        char_prompt = build_character_prompt_detailed(char)
        char_prompts.append(char_prompt)
    if not char_prompts:
        return ""
    consistency_prompt = "CHARACTER CONSISTENCY CRITICAL: " + " | ".join(char_prompts)
    consistency_prompt += " | MAINTAIN EXACT SAME CHARACTER APPEARANCE THROUGHOUT."
    return consistency_prompt


def get_motion_guidance_prompt(shot):
    """Build motion-specific guidance prompt."""
    motion_intensity = shot.get("motion_intensity", 0.5)
    camera_movement = shot.get("camera_movement", "static")
    if motion_intensity < 0.3:
        motion_desc = "minimal motion, subtle movements, mostly static"
    elif motion_intensity < 0.6:
        motion_desc = "moderate motion, natural movements, steady pace"
    else:
        motion_desc = "dynamic motion, pronounced movements, energetic action"
    camera_desc = camera_movement.replace("_", " ")
    prompt = f"MOTION GUIDANCE: {motion_desc}. CAMERA: {camera_desc}. "
    if motion_intensity > 0.6:
        prompt += "Fast-paced action, clear motion trails. "
    else:
        prompt += "Smooth controlled movement, clean frames. "
    return prompt


def get_camera_lora_for_shot(shot):
    """Determine which camera LoRA to use for this shot."""
    camera_movement = shot.get("camera_movement", "static")
    for key in CAMERA_LORA_MAPPING.keys():
        if key in camera_movement:
            return key, CAMERA_LORA_MAPPING[key]
    return None, None


def get_dialogue_for_shot_enhanced(start_time, end_time, dialogue_list):
    """Enhanced dialogue injection with voice sync guidance."""
    lines = []
    for entry in dialogue_list:
        if start_time <= entry.get("time", 0) < end_time:
            char = entry.get("character", "")
            text = entry.get("dialogue", "")
            emotion = entry.get("emotion", "neutral")
            voice_direction = entry.get("voice_direction", "")
            lip_sync = entry.get("lip_sync_emphasis", "medium")
            if char == "The Cave":
                lines.append(f"AUDIO EFFECT: Eerie whisper '{text}' with hollow reverb")
            else:
                if lip_sync == "high":
                    lines.append(
                        f"LIP SYNC CRITICAL: {char} speaks '{text}' with {emotion} emotion. "
                        f"{voice_direction}. Mouth movements MUST match dialogue.")
                else:
                    lines.append(f"{char} says '{text}' with {emotion} emotion. {voice_direction}.")
    return " | ".join(lines) if lines else ""


def build_audio_atmosphere_prompt(shot, scene_json, start_s, end_s):
    """Build comprehensive audio prompt including dialogue and SFX."""
    audio_config = scene_json.get("audio", {})
    atmosphere = f"AUDIO ATMOSPHERE: {audio_config.get('background_music', '')}. "
    atmosphere += f"SOUND EFFECTS: {audio_config.get('environment_sfx', '')}. "
    atmosphere += f"VOICE: {audio_config.get('voice_processing', '')}. "
    dialogue_text = get_dialogue_for_shot_enhanced(
        start_s, end_s, scene_json.get("dialogue_with_timing", []))
    if dialogue_text:
        atmosphere += dialogue_text
    return atmosphere


def build_shot_prompt_pro(shot, json_data, shot_index, prev_shot_success=True):
    """
    PRO version of prompt builder with character consistency enforcement,
    motion guidance, voice sync, and adaptive strength.
    """
    try:
        times = shot["time"].replace("s", "").split("-")
        start_s = int(times[0])
        end_s = int(times[1])
    except Exception:
        start_s, end_s = 0, 5

    character_prompt = get_character_consistency_prefix(json_data)
    action_prompt = f"SHOT {shot_index + 1}: {shot.get('action', '')}. "
    camera_prompt = f"CAMERA: {shot.get('camera', '')}. "
    motion_prompt = get_motion_guidance_prompt(shot)

    env = json_data.get("environment", {})
    env_prompt = (f"ENVIRONMENT: {env.get('location', '')}. "
                  f"LIGHTING: {env.get('lighting', '')}. "
                  f"TIME: {env.get('time', '')}. "
                  f"WEATHER: {env.get('weather', '')}. "
                  f"MOOD: {env.get('mood', '')}. "
                  f"COLOR PALETTE: {env.get('color_palette', '')}. ")

    style_prompt = f"STYLE: {json_data.get('video_style', '')}. "
    audio_prompt = build_audio_atmosphere_prompt(shot, json_data, start_s, end_s)
    vfx_prompt = f"VISUAL EFFECTS: {shot.get('visual_effects', 'natural')}. "
    emotion_prompt = (f"EMOTION: {shot.get('emotion', 'neutral')}. "
                      f"FOCUS: {shot.get('character_focus', 'scene')}. ")

    final_prompt = (
        character_prompt + " " +
        action_prompt +
        camera_prompt +
        motion_prompt +
        emotion_prompt +
        env_prompt +
        vfx_prompt +
        style_prompt +
        audio_prompt
    )
    return final_prompt


def calculate_adaptive_strength(shot, prev_shot, prev_shot_success,
                                anchor_high=0.85, anchor_low=0.70):
    """
    Calculate anchor strength based on motion intensity change,
    character focus change, and previous shot success rate.
    """
    strength = anchor_high

    if prev_shot:
        motion_change = abs(
            shot.get("motion_intensity", 0.5) -
            prev_shot.get("motion_intensity", 0.5)
        )
        if motion_change > 0.4:
            strength -= 0.10
        elif motion_change < 0.2:
            strength += 0.05

    if prev_shot:
        if shot.get("character_focus") != prev_shot.get("character_focus"):
            strength -= 0.05

    if not prev_shot_success:
        strength -= 0.10

    strength = max(anchor_low, min(anchor_high, strength))
    return strength


def build_negative_prompt_enhanced(base_neg=None):
    """Build comprehensive negative prompt."""
    if base_neg is None:
        base_neg = _NEG_BASE
    char_neg = ("character morphing, face changing, inconsistent character design, "
                "different clothing in same scene, style shift, ")
    motion_neg = ("motion blur artifacts, jittery movement, unnatural animation, "
                  "robotic motion, floating characters, ")
    audio_neg = ("desynchronized lips, mouth not moving during speech, "
                 "frozen face during dialogue, mismatched audio, ")
    quality_neg = ("compression artifacts, pixelation, banding, color shifts, "
                   "lighting inconsistency, flickering, ")
    return char_neg + motion_neg + audio_neg + quality_neg + base_neg


print("Imports & engine classes ready.")
print("   VisionDescribeEngine, EasyPromptEngine, CharacterBible")
print("   PRO: CAMERA_LORA_MAPPING, build_shot_prompt_pro, calculate_adaptive_strength")
print("   PRO: build_negative_prompt_enhanced, get_motion_guidance_prompt")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 4  ─  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# @title  { "single-column": true }
# @markdown ## 4. EasyPrompt & Vision Configuration

# ── LLM prompt expander ───────────────────────────────────────────────────────
LLM_MODEL = "8B"  # @param ["8B", "3B", "14B"]
CREATIVITY = 0.9  # @param {type:"number"}
INVENT_DIALOGUE = True  # @param {type:"boolean"}
BYPASS_EASY_PROMPT = False  # @param {type:"boolean"}
LORA_TRIGGERS = ""  # @param {type:"string"}

# ── Vision image describer ────────────────────────────────────────────────────
USE_VISION = True  # @param {type:"boolean"}
VISION_MODEL = "3B-fast"  # @param ["3B-fast", "7B-nsfw"]

# ── Display & output ──────────────────────────────────────────────────────────
SHOW_PREVIEWS = True  # @param {type:"boolean"}
DOWNLOAD_AFTER_GENERATE = False  # @param {type:"boolean"}

# ── PRO settings ──────────────────────────────────────────────────────────────
USE_CHARACTER_LORAS = True  # @param {type:"boolean"}
USE_MOTION_LORAS = True  # @param {type:"boolean"}
USE_VOICE_SYNC = True  # @param {type:"boolean"}
USE_ADAPTIVE_STRENGTH = True  # @param {type:"boolean"}
ANCHOR_STRENGTH_HIGH = 0.85  # @param {type:"number"}
ANCHOR_STRENGTH_LOW = 0.70  # @param {type:"number"}
USE_NEGATIVE_PROMPT_EXPANSION = True  # @param {type:"boolean"}
USE_PROMPT_WEIGHTING = True  # @param {type:"boolean"}
INJECT_CHARACTER_EVERY_SHOT = True  # @param {type:"boolean"}

# ── Example SCENE_JSON schema (edit for your project) ─────────────────────────
SCENE_JSON = {
    "scene_id": "scene_01_example",
    "project_name": "MyProject_PRO",
    "duration_seconds": 24,
    "video_style": "cinematic, professional quality, consistent character design",
    "environment": {
        "location": "Urban city street at night",
        "time": "Late evening, artificial lighting",
        "weather": "Light rain, wet surfaces",
        "mood": "Mysterious, atmospheric",
        "lighting": "Neon reflections, volumetric haze",
        "color_palette": "Deep blues, warm amber highlights, purple neon"
    },
    "main_characters": [
        {
            "name": "Character",
            "desc": "Main character",
            "detailed_appearance": {
                "face": "Defined features, expressive eyes",
                "hair": "Dark hair, natural style",
                "clothing": "Dark jacket, casual wear",
                "build": "Average build",
                "skin_tone": "Natural skin tone",
                "accessories": "None"
            },
            "lora_path": None,
            "personality_traits": "Determined, cautious",
            "voice_characteristics": "Clear voice"
        }
    ],
    "story_action": {
        "shots": [
            {
                "time": "0-4s",
                "camera": "Wide establishing shot",
                "camera_movement": "static",
                "motion_intensity": 0.3,
                "action": "Character walks through the rain-soaked street",
                "character_focus": "main",
                "emotion": "determined",
                "visual_effects": "Rain drops, neon reflections"
            },
            {
                "time": "4-8s",
                "camera": "Medium tracking shot",
                "camera_movement": "dolly_forward",
                "motion_intensity": 0.5,
                "action": "Character looks over shoulder, speeds up pace",
                "character_focus": "main",
                "emotion": "alert",
                "visual_effects": "Motion blur background, focused subject"
            }
        ]
    },
    "dialogue_with_timing": [],
    "audio": {
        "background_music": "Low atmospheric ambient",
        "environment_sfx": "Rain, distant traffic, footsteps",
        "voice_processing": "Natural reverb"
    }
}

print("Configuration ready.")
print(f"   LLM: {LLM_MODEL} | Vision: {VISION_MODEL} | Creativity: {CREATIVITY}")
print(f"   PRO: Characters={USE_CHARACTER_LORAS} Motion={USE_MOTION_LORAS} Voice={USE_VOICE_SYNC}")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 5  ─  CHARACTER & LORA CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# @title  { "single-column": true }
# @markdown ## 5. Character & LoRA Configuration

# ── Character Consistency ─────────────────────────────────────────────────────
CHARACTER_IMAGE_PATH = None  # @param {type:"string"}
CHARACTER_STRENGTH = 1.0  # @param {type:"number"}
CHARACTER_CONSISTENCY_MODE = "i2v"  # @param ["i2v", "anchor", "both", "none"]
CHARACTER_NAME = "Character"  # @param {type:"string"}
CHARACTER_DESCRIPTION = ""  # @param {type:"string"}

# ── IC LoRA (Image Conditioning / Detailer) ───────────────────────────────────
IC_LORA = "detailer"  # @param ["none", "detailer", "canny", "depth", "pose"]
IC_LORA_STRENGTH = 0.4  # @param {type:"number"}

_IC_LORA_FILES = {
    "none": "None",
    "detailer": "ltx-2-19b-ic-lora-detailer.safetensors",
    "canny": "ltx-2-19b-ic-lora-canny-control.safetensors",
    "depth": "ltx-2-19b-ic-lora-depth-control.safetensors",
    "pose": "ltx-2-19b-ic-lora-pose-control.safetensors",
}

# ── Camera LoRA ───────────────────────────────────────────────────────────────
CAMERA_LORA = "none"  # @param ["none", "dolly-in", "dolly-out", "dolly-left", "dolly-right", "jib-up", "jib-down", "static"]
CAMERA_LORA_STRENGTH = 1.0  # @param {type:"number"}

_CAMERA_LORA_FILES = {
    "none": "None",
    "dolly-in": "ltx-2-19b-lora-camera-control-dolly-in.safetensors",
    "dolly-out": "ltx-2-19b-lora-camera-control-dolly-out.safetensors",
    "dolly-left": "ltx-2-19b-lora-camera-control-dolly-left.safetensors",
    "dolly-right": "ltx-2-19b-lora-camera-control-dolly-right.safetensors",
    "jib-up": "ltx-2-19b-lora-camera-control-jib-up.safetensors",
    "jib-down": "ltx-2-19b-lora-camera-control-jib-down.safetensors",
    "static": "ltx-2-19b-lora-camera-control-static.safetensors",
}

# ── User LoRA slots (strength) ────────────────────────────────────────────────
# Note: distilled LoRA (ltx-2.3-22b-distilled-lora-384.safetensors) is mandatory
# and always applied FIRST - not user-configurable.
USER_LORA_1_STRENGTH = 1.0  # @param {type:"number"}
USER_LORA_2_STRENGTH = 1.0  # @param {type:"number"}
USER_LORA_3_STRENGTH = 1.0  # @param {type:"number"}

# ── Performance flags ─────────────────────────────────────────────────────────
USE_SAGE_ATTENTION = False  # @param {type:"boolean"}
USE_CHUNK_FF = False  # @param {type:"boolean"}
PURGE_VRAM_AFTER_MODELS = True  # @param {type:"boolean"}

# ── PRO Mode settings ─────────────────────────────────────────────────────────
PRO_MODE = False  # @param {type:"boolean"}
MOTION_STRENGTH = 0.75  # @param {type:"number"}
USE_VOICE_SYNC_STRENGTH = 0.95  # @param {type:"number"}

print("Character & LoRA configuration ready.")
print(f"   Character mode: {CHARACTER_CONSISTENCY_MODE} | strength: {CHARACTER_STRENGTH}")
print(f"   IC LoRA: {IC_LORA} @ {IC_LORA_STRENGTH}")
print(f"   Camera LoRA: {CAMERA_LORA} @ {CAMERA_LORA_STRENGTH}")
print(f"   SageAttn: {USE_SAGE_ATTENTION} | ChunkFF: {USE_CHUNK_FF}")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 6  ─  VIDEO GENERATION CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# @title  { "single-column": true }
# @markdown ## 6. Video Generation Configuration

# ── Simple input (expanded by EasyPromptEngine) ──────────────────────────────
USER_INPUT = (  # @param {type:"string"}
    "a woman walks through a rain-soaked city street at night, "
    "neon reflections on the wet pavement, looking over her shoulder"
)

# ── Reference / seed image (optional) ────────────────────────────────────────
IMAGE_PATH = None  # @param {type:"string"}
IMAGE_STRENGTH = 1.0  # @param {type:"number"}

# ── Manual prompts (BYPASS_EASY_PROMPT=True only) ─────────────────────────────
POSITIVE_PROMPT = (  # @param {type:"string"}
    "Busy city street at night, cinematic, neon reflections on wet pavement, "
    "woman walking, bokeh streetlights, moody atmosphere, ultra detailed"
)
NEGATIVE_PROMPT = (  # @param {type:"string"}
    "blurry, distorted, low quality, watermark, text, bad anatomy, deformed, "
    "grainy, overexposed, underexposed, flickering, motion artifacts"
)

# ── Resolution & length ───────────────────────────────────────────────────────
WIDTH = 832  # @param {type:"integer"}
HEIGHT = 480  # @param {type:"integer"}
FRAMES = 121  # @param {type:"integer"}
FPS = 25  # @param {type:"integer"}

# ── Seed ──────────────────────────────────────────────────────────────────────
SEED = 42  # @param {type:"integer"}
AUTO_INCREMENT_SEED = True  # @param {type:"boolean"}

# ── Model filenames (LTX 2.3 versions) ───────────────────────────────────────
UNET_MODEL = "ltx-2.3-22b-dev-Q4_K_M.gguf"
CLIP_NAME1 = "gemma_3_12B_it_fp8_scaled.safetensors"
CLIP_NAME2 = "ltx-2.3-22b-dev_embeddings_connectors.safetensors"
CLIP_GGUF_NAME1 = "gemma-3-12b-it-qat-UD-Q4_K_XL.gguf"
CLIP_GGUF_NAME2 = "mmproj-BF16.gguf"
VAE_VIDEO_MODEL = "ltx-2.3-22b-dev_video_vae.safetensors"
VAE_AUDIO_MODEL = "ltx-2.3-22b-dev_audio_vae.safetensors"
UPSCALER_MODEL = "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
DISTILLED_LORA = "ltx-2.3-22b-distilled-lora-384.safetensors"

# ── Sigma schedules ───────────────────────────────────────────────────────────
PASS1_SIGMAS = "1., 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
PASS1_SAMPLER = "euler"  # @param {type:"string"}
PASS1_CFG = 1.0  # @param {type:"number"}

PASS2_SIGMAS = "0.909375, 0.725, 0.421875, 0.0"
PASS2_SAMPLER = "gradient_estimation"  # @param {type:"string"}
PASS2_CFG = 1.0  # @param {type:"number"}
PASS2_SEED = 0  # @param {type:"integer"}

# ── Tiled VAE decode ──────────────────────────────────────────────────────────
USE_TILED_VAE = True  # @param {type:"boolean"}
TILED_SPATIAL_TILES = 2  # @param {type:"integer"}
TILED_SPATIAL_OVERLAP = 8  # @param {type:"integer"}
TILED_TEMPORAL_LEN = 48  # @param {type:"integer"}
TILED_TEMPORAL_OVERLAP = 4  # @param {type:"integer"}

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_PREFIX = "LTX-2.3-PRO"  # @param {type:"string"}
USE_SCENE_CONTINUITY = True  # @param {type:"boolean"}

print("Video configuration set.")
print(f"   Resolution: {WIDTH}x{HEIGHT} | Frames: {FRAMES} ({FRAMES/FPS:.1f}s @ {FPS}fps)")
print(f"   UNet: {UNET_MODEL}")
print(f"   Distilled LoRA: {DISTILLED_LORA} (mandatory, applied first)")
print(f"   Seed: {SEED} (auto-increment: {AUTO_INCREMENT_SEED})")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 7  ─  DEFINE generate_clip()
# ══════════════════════════════════════════════════════════════════════════════

# @title  { "single-column": true }
# @markdown ## 7. Define generate_clip()
# @markdown Two-pass LTX-2.3 pipeline with CLIP-delete-before-UNet VRAM strategy.
# @markdown Distilled LoRA is applied FIRST before any other LoRAs.

def generate_clip(
    image_tensor: Optional[torch.Tensor] = None,
    prompt: str = "",
    neg_prompt: str = "",
    width: int = 832,
    height: int = 480,
    frames: int = 121,
    fps: int = 25,
    seed: int = 42,
    image_strength: float = 1.0,
    pass1_sigmas: str = PASS1_SIGMAS,
    pass1_sampler: str = PASS1_SAMPLER,
    pass1_cfg: float = PASS1_CFG,
    pass2_sigmas: str = PASS2_SIGMAS,
    pass2_sampler: str = PASS2_SAMPLER,
    pass2_cfg: float = PASS2_CFG,
    pass2_seed: int = PASS2_SEED,
    use_tiled_vae: bool = USE_TILED_VAE,
    tiled_stiles: int = TILED_SPATIAL_TILES,
    tiled_soverlap: int = TILED_SPATIAL_OVERLAP,
    tiled_tlen: int = TILED_TEMPORAL_LEN,
    tiled_toverlap: int = TILED_TEMPORAL_OVERLAP,
    ic_lora: str = None,
    ic_lora_strength: float = 0.4,
    camera_lora_file: str = None,
    camera_lora_strength: float = 1.0,
    output_prefix: str = OUTPUT_PREFIX,
) -> str:
    """
    Two-pass LTX-2.3 22B GGUF generation for one clip.
    T4-safe VRAM strategy: CLIP is deleted before UNet loads.

    VRAM sequence:
      1. Load CLIP (DualCLIPLoader) - safetensors Gemma primary + GGUF fallback
      2. Encode text (CLIPTextEncode + ConditioningZeroOut + LTXVConditioning.EXECUTE_NORMALIZED)
      3. DELETE CLIP (frees 6-8 GB)
      4. Load UNet (UnetLoaderGGUF - ltx-2.3-22b-dev-Q4_K_M.gguf)
      5. Apply distilled LoRA FIRST (ltx-2.3-22b-distilled-lora-384.safetensors @ 1.0)
      6. Apply IC LoRA, camera LoRA, user LoRAs
      7. Load VAEs + upscaler
      8. Prepare latents using tensor_width_height()
      9. I2V conditioning if image provided
      10. Audio latent + concat AV
      11. Pass 1 (euler + ManualSigmas)
      12. Separate AV + CropGuides + CFGGuider for Pass 2
      13. LTXVLatentUpsampler (spatial x2)
      14. Pass 2 (gradient_estimation + ManualSigmas)
      15. DELETE UNet
      16. Decode video (tiled or standard VAEDecode)
      17. Decode audio (LTXVAudioVAEDecode)
      18. Delete VAEs
      19. CreateVideo + save

    Returns: output video file path (str)
    """
    import_custom_nodes()

    img_bypass = image_tensor is None
    img_str = image_strength if not img_bypass else 0.0

    print(f"   [Clip] {'I2V' if not img_bypass else 'T2V'}  "
          f"{width}x{height}  {frames}f  seed={seed}")

    with torch.inference_mode():

        # ══════════════════════════════════════════════════════════════════
        # STEP 1: Load CLIP (DualCLIPLoader)
        # Primary: gemma_3_12B_it_fp8_scaled + ltx-2.3 embeddings connector
        # Fallback: GGUF Gemma + mmproj pair
        # ══════════════════════════════════════════════════════════════════
        print("   [Clip] Loading CLIP...")
        dualcliploader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
        try:
            dualcliploader_result = dualcliploader.load_clip(
                clip_name1=CLIP_NAME1,
                clip_name2=CLIP_NAME2,
                type="ltxv",
                device="default",
            )
        except Exception as clip_error:
            print(f"   Primary CLIP load failed, trying GGUF fallback: {clip_error}")
            dualcliploader_result = dualcliploader.load_clip(
                clip_name1=CLIP_GGUF_NAME1,
                clip_name2=CLIP_GGUF_NAME2,
                type="ltxv",
                device="default",
            )

        clip_model = get_value_at_index(dualcliploader_result, 0)

        # ══════════════════════════════════════════════════════════════════
        # STEP 2: Text encoding
        # CLIPTextEncode + ConditioningZeroOut + LTXVConditioning.EXECUTE_NORMALIZED
        # ══════════════════════════════════════════════════════════════════
        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cliptextencode_pos = cliptextencode.encode(
            text=prompt,
            clip=clip_model,
        )

        conditioningzeroout = NODE_CLASS_MAPPINGS["ConditioningZeroOut"]()
        conditioningzeroout_neg = conditioningzeroout.zero_out(
            conditioning=get_value_at_index(cliptextencode_pos, 0)
        )

        ltxvconditioning = NODE_CLASS_MAPPINGS["LTXVConditioning"]()
        ltxvconditioning_result = ltxvconditioning.EXECUTE_NORMALIZED(
            frame_rate=fps,
            positive=get_value_at_index(cliptextencode_pos, 0),
            negative=get_value_at_index(conditioningzeroout_neg, 0),
        )

        # ══════════════════════════════════════════════════════════════════
        # STEP 3: DELETE CLIP (frees 6-8 GB for UNet)
        # ══════════════════════════════════════════════════════════════════
        del clip_model, dualcliploader_result
        torch.cuda.empty_cache()
        gc.collect()
        print("   [Clip] CLIP deleted, VRAM freed")

        # ══════════════════════════════════════════════════════════════════
        # STEP 4: Load UNet (UnetLoaderGGUF)
        # ══════════════════════════════════════════════════════════════════
        print("   [Clip] Loading UNet (GGUF Q4_K_M)...")
        unetloadergguf = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
        unetloadergguf_result = unetloadergguf.load_unet(unet_name=UNET_MODEL)
        unet = get_value_at_index(unetloadergguf_result, 0)

        # ══════════════════════════════════════════════════════════════════
        # STEP 5: Apply distilled LoRA FIRST (mandatory)
        # ltx-2.3-22b-distilled-lora-384.safetensors @ 1.0
        # ══════════════════════════════════════════════════════════════════
        print("   [Clip] Applying distilled LoRA (mandatory, first)...")
        load_lora = LoraLoaderModelOnly()
        unet = load_lora.load_lora_model_only(unet, DISTILLED_LORA, 1.0)[0]

        # ══════════════════════════════════════════════════════════════════
        # STEP 6: Apply IC LoRA, camera LoRA, user LoRAs
        # ══════════════════════════════════════════════════════════════════
        # IC LoRA
        _ic_file = _IC_LORA_FILES.get((ic_lora or IC_LORA).lower(), "None")
        if _ic_file != "None":
            _ic_str = ic_lora_strength if ic_lora else IC_LORA_STRENGTH
            try:
                ll = LoraLoaderModelOnly()
                unet = ll.load_lora_model_only(unet, _ic_file, _ic_str)[0]
                print(f"   [Clip] IC LoRA: {_ic_file} @ {_ic_str}")
            except Exception as e:
                print(f"   [Clip] IC LoRA failed: {e}")

        # Camera LoRA
        _cam_file = camera_lora_file or _CAMERA_LORA_FILES.get(CAMERA_LORA.lower(), "None")
        _cam_str = camera_lora_strength if camera_lora_file else CAMERA_LORA_STRENGTH
        if _cam_file != "None" and _cam_file is not None:
            try:
                ll = LoraLoaderModelOnly()
                unet = ll.load_lora_model_only(unet, _cam_file, _cam_str)[0]
                print(f"   [Clip] Camera LoRA: {_cam_file} @ {_cam_str}")
            except Exception as e:
                print(f"   [Clip] Camera LoRA failed: {e}")

        # User LoRAs (downloaded in Cell 2)
        if user_lora_1:
            try:
                ll = LoraLoaderModelOnly()
                unet = ll.load_lora_model_only(unet, user_lora_1, USER_LORA_1_STRENGTH)[0]
                print(f"   [Clip] User LoRA 1: {user_lora_1}")
            except Exception as e:
                print(f"   [Clip] User LoRA 1 failed: {e}")

        if user_lora_2:
            try:
                ll = LoraLoaderModelOnly()
                unet = ll.load_lora_model_only(unet, user_lora_2, USER_LORA_2_STRENGTH)[0]
                print(f"   [Clip] User LoRA 2: {user_lora_2}")
            except Exception as e:
                print(f"   [Clip] User LoRA 2 failed: {e}")

        if user_lora_3:
            try:
                ll = LoraLoaderModelOnly()
                unet = ll.load_lora_model_only(unet, user_lora_3, USER_LORA_3_STRENGTH)[0]
                print(f"   [Clip] User LoRA 3: {user_lora_3}")
            except Exception as e:
                print(f"   [Clip] User LoRA 3 failed: {e}")

        # Optional performance patches
        if USE_SAGE_ATTENTION:
            unet = apply_sage_attention(unet)
        if USE_CHUNK_FF:
            unet = apply_chunk_ff(unet)

        # ══════════════════════════════════════════════════════════════════
        # STEP 7: Load VAEs + upscaler
        # ══════════════════════════════════════════════════════════════════
        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        vae_video = get_value_at_index(vaeloader.load_vae(vae_name=VAE_VIDEO_MODEL), 0)
        vae_audio = get_value_at_index(load_audio_vae_compat(VAE_AUDIO_MODEL), 0)

        latentupscalemodelloader = NODE_CLASS_MAPPINGS["LatentUpscaleModelLoader"]()
        upscale_mdl = get_value_at_index(
            latentupscalemodelloader.EXECUTE_NORMALIZED(model_name=UPSCALER_MODEL), 0)

        # ══════════════════════════════════════════════════════════════════
        # STEP 8: Prepare latents using tensor_width_height()
        # latent_w = max(1, resized_w // 2) (NOT GetImageSize node)
        # ══════════════════════════════════════════════════════════════════
        resizeimagemasknode = NODE_CLASS_MAPPINGS["ResizeImageMaskNode"]()
        resizeimagesbylongeredge = NODE_CLASS_MAPPINGS["ResizeImagesByLongerEdge"]()
        ltxvpreprocess = NODE_CLASS_MAPPINGS["LTXVPreprocess"]()
        emptyltxvlatentvideo = NODE_CLASS_MAPPINGS["EmptyLTXVLatentVideo"]()

        # Prepare image for I2V (or create noise placeholder for T2V)
        if not img_bypass:
            # Resize provided image to target dimensions
            resized_img = resizeimagemasknode.EXECUTE_NORMALIZED(
                input=image_tensor,
                scale_method="lanczos",
                resize_type={
                    "resize_type": "scale dimensions",
                    "width": width,
                    "height": height,
                    "crop": "center",
                }
            )
            resized_for_longer = resizeimagesbylongeredge.EXECUTE_NORMALIZED(
                longer_edge=848,
                images=get_value_at_index(resized_img, 0)
            )
            pp_img = get_value_at_index(ltxvpreprocess.EXECUTE_NORMALIZED(
                img_compression=33,
                image=get_value_at_index(resized_for_longer, 0),
            ), 0)

            # Get dimensions using tensor_width_height (NOT GetImageSize node)
            resized_w, resized_h = tensor_width_height(get_value_at_index(resized_img, 0))
        else:
            # T2V: use configured dimensions directly
            resized_w, resized_h = width, height
            pp_img = None

        # Calculate latent dimensions: half resolution
        latent_w = max(1, resized_w // 2)
        latent_h = max(1, resized_h // 2)

        # Create empty video latent at half resolution
        emptyltxvlatentvideo_result = emptyltxvlatentvideo.EXECUTE_NORMALIZED(
            width=latent_w,
            height=latent_h,
            length=frames,
            batch_size=1,
        )

        # ══════════════════════════════════════════════════════════════════
        # STEP 9: I2V conditioning (LTXVImgToVideoInplace) if image provided
        # ══════════════════════════════════════════════════════════════════
        ltxvimgtovideoinplace = NODE_CLASS_MAPPINGS["LTXVImgToVideoInplace"]()

        if not img_bypass:
            vid_lat_conditioned = ltxvimgtovideoinplace.EXECUTE_NORMALIZED(
                strength=img_str,
                bypass=False,
                vae=vae_video,
                image=pp_img,
                latent=get_value_at_index(emptyltxvlatentvideo_result, 0),
            )
            vid_lat = get_value_at_index(vid_lat_conditioned, 0)
        else:
            vid_lat = get_value_at_index(emptyltxvlatentvideo_result, 0)

        # ══════════════════════════════════════════════════════════════════
        # STEP 10: Audio latent + concat AV
        # ══════════════════════════════════════════════════════════════════
        ltxvemptylatentaudio = NODE_CLASS_MAPPINGS["LTXVEmptyLatentAudio"]()
        ltxvemptylatentaudio_result = ltxvemptylatentaudio.EXECUTE_NORMALIZED(
            frames_number=frames,
            frame_rate=fps,
            batch_size=1,
            audio_vae=vae_audio,
        )

        ltxvconcatavlatent = NODE_CLASS_MAPPINGS["LTXVConcatAVLatent"]()
        av_lat = get_value_at_index(ltxvconcatavlatent.EXECUTE_NORMALIZED(
            video_latent=vid_lat,
            audio_latent=get_value_at_index(ltxvemptylatentaudio_result, 0),
        ), 0)

        # ══════════════════════════════════════════════════════════════════
        # STEP 11: Pass 1 (euler + ManualSigmas)
        # ══════════════════════════════════════════════════════════════════
        print("   [Clip] Pass 1...")
        manualsigmas = NODE_CLASS_MAPPINGS["ManualSigmas"]()
        ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
        randomnoise = NODE_CLASS_MAPPINGS["RandomNoise"]()
        cfgguider = NODE_CLASS_MAPPINGS["CFGGuider"]()
        samplercustomadvanced = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()

        cfgguider_p1 = cfgguider.EXECUTE_NORMALIZED(
            cfg=pass1_cfg,
            model=unet,
            positive=get_value_at_index(ltxvconditioning_result, 0),
            negative=get_value_at_index(ltxvconditioning_result, 1),
        )

        samplercustomadvanced_p1 = samplercustomadvanced.EXECUTE_NORMALIZED(
            noise=get_value_at_index(randomnoise.EXECUTE_NORMALIZED(noise_seed=seed), 0),
            guider=get_value_at_index(cfgguider_p1, 0),
            sampler=get_value_at_index(ksamplerselect.EXECUTE_NORMALIZED(sampler_name=pass1_sampler), 0),
            sigmas=get_value_at_index(manualsigmas.EXECUTE_NORMALIZED(sigmas=pass1_sigmas), 0),
            latent_image=av_lat,
        )

        del cfgguider_p1
        torch.cuda.empty_cache()
        gc.collect()
        print("   [Clip] Pass 1 done")

        # ══════════════════════════════════════════════════════════════════
        # STEP 12: Separate AV + CropGuides + CFGGuider for Pass 2
        # ══════════════════════════════════════════════════════════════════
        ltxvseparateavlatent = NODE_CLASS_MAPPINGS["LTXVSeparateAVLatent"]()
        ltxvseparateavlatent_p1 = ltxvseparateavlatent.EXECUTE_NORMALIZED(
            av_latent=get_value_at_index(samplercustomadvanced_p1, 0)
        )

        ltxvcropguides = NODE_CLASS_MAPPINGS["LTXVCropGuides"]()
        ltxvcropguides_result = ltxvcropguides.EXECUTE_NORMALIZED(
            positive=get_value_at_index(ltxvconditioning_result, 0),
            negative=get_value_at_index(ltxvconditioning_result, 1),
            latent=get_value_at_index(ltxvseparateavlatent_p1, 0),
        )

        cfgguider_p2 = cfgguider.EXECUTE_NORMALIZED(
            cfg=pass2_cfg,
            model=unet,
            positive=get_value_at_index(ltxvcropguides_result, 0),
            negative=get_value_at_index(ltxvcropguides_result, 1),
        )

        # ══════════════════════════════════════════════════════════════════
        # STEP 13: LTXVLatentUpsampler (spatial x2)
        # ══════════════════════════════════════════════════════════════════
        ltxvlatentupsampler = NODE_CLASS_MAPPINGS["LTXVLatentUpsampler"]()
        ltxvlatentupsampler_result = ltxvlatentupsampler.upsample_latent(
            samples=get_value_at_index(ltxvcropguides_result, 2),
            upscale_model=upscale_mdl,
            vae=vae_video,
        )

        del upscale_mdl
        torch.cuda.empty_cache()
        gc.collect()

        # I2V re-apply on upsampled latent
        if not img_bypass:
            upsampled_conditioned = ltxvimgtovideoinplace.EXECUTE_NORMALIZED(
                strength=img_str,
                bypass=False,
                vae=vae_video,
                image=pp_img,
                latent=get_value_at_index(ltxvlatentupsampler_result, 0),
            )
            av_lat2 = get_value_at_index(ltxvconcatavlatent.EXECUTE_NORMALIZED(
                video_latent=get_value_at_index(upsampled_conditioned, 0),
                audio_latent=get_value_at_index(ltxvseparateavlatent_p1, 1),
            ), 0)
        else:
            av_lat2 = get_value_at_index(ltxvconcatavlatent.EXECUTE_NORMALIZED(
                video_latent=get_value_at_index(ltxvlatentupsampler_result, 0),
                audio_latent=get_value_at_index(ltxvseparateavlatent_p1, 1),
            ), 0)

        # ══════════════════════════════════════════════════════════════════
        # STEP 14: Pass 2 (gradient_estimation + ManualSigmas)
        # ══════════════════════════════════════════════════════════════════
        print("   [Clip] Pass 2...")
        samplercustomadvanced_p2 = samplercustomadvanced.EXECUTE_NORMALIZED(
            noise=get_value_at_index(randomnoise.EXECUTE_NORMALIZED(noise_seed=pass2_seed), 0),
            guider=get_value_at_index(cfgguider_p2, 0),
            sampler=get_value_at_index(ksamplerselect.EXECUTE_NORMALIZED(sampler_name=pass2_sampler), 0),
            sigmas=get_value_at_index(manualsigmas.EXECUTE_NORMALIZED(sigmas=pass2_sigmas), 0),
            latent_image=av_lat2,
        )

        # ══════════════════════════════════════════════════════════════════
        # STEP 15: DELETE UNet (free VRAM for decode)
        # ══════════════════════════════════════════════════════════════════
        del cfgguider_p2, unet
        torch.cuda.empty_cache()
        gc.collect()
        print("   [Clip] UNet deleted, Pass 2 done")

        # ══════════════════════════════════════════════════════════════════
        # STEP 16: Decode video (tiled or standard VAEDecode)
        # ══════════════════════════════════════════════════════════════════
        ltxvseparateavlatent_p2 = ltxvseparateavlatent.EXECUTE_NORMALIZED(
            av_latent=get_value_at_index(samplercustomadvanced_p2, 1)
        )

        video_latent_final = get_value_at_index(ltxvseparateavlatent_p2, 0)
        audio_latent_final = get_value_at_index(ltxvseparateavlatent_p2, 1)

        decoded_video = None
        if use_tiled_vae:
            try:
                tiled_decode = NODE_CLASS_MAPPINGS["LTXVSpatioTemporalTiledVAEDecode"]()
                decoded_video = get_value_at_index(tiled_decode.EXECUTE_NORMALIZED(
                    vae=vae_video, latents=video_latent_final,
                    spatial_tiles=tiled_stiles, spatial_overlap=tiled_soverlap,
                    temporal_tile_length=tiled_tlen, temporal_overlap=tiled_toverlap,
                    last_frame_fix=False, working_device="auto", working_dtype="auto"), 0)
                print("   [Clip] Tiled VAE decode done")
            except (KeyError, Exception) as e:
                print(f"   [Clip] Tiled VAE skipped ({type(e).__name__}) - standard decode")
                use_tiled_vae = False

        if not use_tiled_vae or decoded_video is None:
            vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
            decoded_video = get_value_at_index(
                vaedecode.decode(samples=video_latent_final, vae=vae_video), 0)

        del vae_video
        torch.cuda.empty_cache()
        gc.collect()

        # ══════════════════════════════════════════════════════════════════
        # STEP 17: Decode audio (LTXVAudioVAEDecode)
        # ══════════════════════════════════════════════════════════════════
        ltxvaudiovaedecode = NODE_CLASS_MAPPINGS["LTXVAudioVAEDecode"]()
        decoded_audio = ltxvaudiovaedecode.EXECUTE_NORMALIZED(
            samples=audio_latent_final,
            audio_vae=vae_audio,
        )

        # ══════════════════════════════════════════════════════════════════
        # STEP 18: Delete VAEs
        # ══════════════════════════════════════════════════════════════════
        del vae_audio
        torch.cuda.empty_cache()
        gc.collect()

        # ══════════════════════════════════════════════════════════════════
        # STEP 19: CreateVideo + save
        # ══════════════════════════════════════════════════════════════════
        print("   [Clip] Creating video...")
        createvideo = NODE_CLASS_MAPPINGS["CreateVideo"]()
        createvideo_result = createvideo.EXECUTE_NORMALIZED(
            fps=fps,
            images=decoded_video,
            audio=get_value_at_index(decoded_audio, 0),
        )

        video_obj = get_value_at_index(createvideo_result, 0)
        output_path = save_video_from_components(video_obj, prefix=output_prefix)
        print(f"   [Clip] Saved: {output_path}")
        return output_path


print("generate_clip() defined.")
print("   VRAM strategy: CLIP -> encode -> DELETE CLIP -> load UNet -> LoRAs -> generate -> DELETE UNet -> decode")
print("   Distilled LoRA (ltx-2.3-22b-distilled-lora-384.safetensors) applied FIRST, always.")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 8  ─  INFINITE FLOW ENGINE + STORYBOARD
# ══════════════════════════════════════════════════════════════════════════════

# @title  { "single-column": true }
# @markdown ## 8. InfiniteFlowEngine + Storyboard Runner
# @markdown 12-scene orchestrator with last-frame chaining + rolling context.

class InfiniteFlowEngine:
    """
    Generates a sequence of clips where:
      1. VisionDescribeEngine analyses the current seed/last frame
      2. EasyPromptEngine expands the story beat into a cinematic prompt,
         injecting the CharacterBible as a hard consistency lock
      3. generate_clip() renders the clip with the two-pass LTX-2.3 pipeline
      4. get_last_frame_tensor() extracts the last frame for the next beat
      5. All clips are saved and can be concatenated

    Every heavyweight model is loaded -> used -> unloaded in strict sequence.
    """

    def __init__(
        self,
        character_bible: Optional[CharacterBible] = None,
        llm_model: str = "8B",
        vision_model: str = "3B-fast",
        width: int = 832,
        height: int = 480,
        frames: int = 121,
        fps: int = 25,
        image_strength: float = 1.0,
        creativity: float = 0.9,
        lora_triggers: str = "",
        use_vision: bool = True,
        use_tiled_vae: bool = True,
        offline: bool = False,
        output_dir: str = "/content/ComfyUI/output/infinite_flow",
        show_previews: bool = True,
    ):
        self.bible = character_bible or CharacterBible()
        self.llm_model = llm_model
        self.vision_model = vision_model
        self.width = width
        self.height = height
        self.frames = frames
        self.fps = fps
        self.img_strength = image_strength
        self.creativity = creativity
        self.lora_triggers = lora_triggers
        self.use_vision = use_vision
        self.use_tiled_vae = use_tiled_vae
        self.offline = offline
        self.output_dir = output_dir
        self.show_previews = show_previews
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        self._vision_engine = VisionDescribeEngine(vision_model, offline)
        self._prompt_engine = EasyPromptEngine(llm_model, offline, keep_loaded=False)

        self._clip_paths: List[str] = []
        self._scene_history: List[str] = []

    def _rolling_ctx(self) -> str:
        return "\n".join(self._scene_history[-2:])

    def _process_beat(self, beat: str, current_image: Optional[torch.Tensor],
                      beat_idx: int, base_seed: int) -> tuple:
        beat_seed = base_seed + (beat_idx - 1) * 1000

        if beat_idx > 1 and current_image is not None and self.use_vision:
            print(f"   [Beat {beat_idx}] Vision Describe...")
            scene_ctx = self._vision_engine.describe(current_image)
            cleanup_memory()
        else:
            scene_ctx = self._rolling_ctx()

        print(f"   [Beat {beat_idx}] EasyPrompt expand...")
        prompt, neg = self._prompt_engine.generate(
            user_input=beat,
            frame_count=self.frames,
            creativity=self.creativity,
            seed=beat_seed,
            scene_context=scene_ctx,
            lora_triggers=self.lora_triggers,
            character_bible=self.bible.to_prompt_block(),
        )
        cleanup_memory()

        self._scene_history.append(prompt[:400])
        if len(self._scene_history) > 2:
            self._scene_history = self._scene_history[-2:]

        return prompt, neg, beat_seed

    def run(
        self,
        story_beats: List[str],
        seed_image_path: Optional[str] = None,
        base_seed: int = 42,
    ) -> List[str]:
        """
        Process each beat in story_beats.
        Returns list of output .mp4 paths (one per beat).
        """
        current_image: Optional[torch.Tensor] = None

        if seed_image_path and os.path.exists(seed_image_path):
            current_image = load_image_tensor(seed_image_path)
            print(f"[IFE] Seed image loaded: {seed_image_path}")

            if not self.bible.has_characters() and self.use_vision:
                print("[IFE] Extracting Character Bible from seed image...")
                desc = self._vision_engine.describe(current_image)
                cleanup_memory()
                self.bible.extract_from_description("Main Character", desc)
                self._scene_history.append(desc)
                print(f"[IFE] Bible set: {desc[:200]}...")

        print(f"\n[IFE] === INFINITE FLOW ENGINE ===")
        print(f"[IFE]  Beats    : {len(story_beats)}")
        print(f"[IFE]  Size     : {self.width}x{self.height}  {self.frames}f @ {self.fps}fps")
        print(f"[IFE]  LLM      : {self.llm_model}  Vision: {self.vision_model}")
        print(f"[IFE]  Bible    : {self.bible.names() or 'empty (T2V)'}")
        print(f"[IFE]  Mode     : {'I2V' if current_image is not None else 'T2V'}\n")

        for beat_idx, beat in enumerate(story_beats, 1):
            print(f"\n[IFE] -- Beat {beat_idx}/{len(story_beats)} --")
            print(f"[IFE]   {beat[:80]}{'...' if len(beat) > 80 else ''}")

            prompt, neg, beat_seed = self._process_beat(
                beat, current_image, beat_idx, base_seed)

            print(f"\n  EXPANDED ({len(prompt.split())}w): {prompt[:200]}...")
            print(f"  NEG: {neg[:100]}...\n")

            try:
                clip_path = generate_clip(
                    image_tensor=current_image,
                    prompt=prompt,
                    neg_prompt=neg,
                    width=self.width,
                    height=self.height,
                    frames=self.frames,
                    fps=self.fps,
                    seed=beat_seed,
                    image_strength=self.img_strength,
                    use_tiled_vae=self.use_tiled_vae,
                    output_prefix=f"IFE_{beat_idx:03d}",
                )
            except torch.cuda.OutOfMemoryError:
                cleanup_memory()
                print(f"   OOM on beat {beat_idx}. Retrying T2V...")
                clip_path = generate_clip(
                    image_tensor=None,
                    prompt=prompt,
                    neg_prompt=neg,
                    width=self.width,
                    height=self.height,
                    frames=self.frames,
                    fps=self.fps,
                    seed=beat_seed + 1,
                    use_tiled_vae=False,
                    output_prefix=f"IFE_{beat_idx:03d}_retry",
                )

            dest = os.path.join(self.output_dir, f"scene_{beat_idx:03d}.mp4")
            shutil.copy2(clip_path, dest)
            self._clip_paths.append(dest)
            print(f"\n[IFE] Beat {beat_idx} -> {dest}")

            last_frame = get_last_frame_tensor(dest)
            current_image = last_frame if last_frame is not None else None
            if last_frame is not None:
                print(f"[IFE]    Last frame extracted for beat {beat_idx + 1}.")
            else:
                print(f"[IFE]    No last frame - next beat will be T2V.")

            if self.show_previews:
                display_video(dest)

        print(f"\n[IFE] === {len(story_beats)} scenes complete ===")
        print(f"[IFE]  Output dir: {self.output_dir}")
        return self._clip_paths

    def concat_all(self, output_name: str = "full_video.mp4") -> str:
        """Concatenate all generated clips into one final video."""
        out = os.path.join(self.output_dir, output_name)
        return concatenate_clips(self._clip_paths, out)

    def download_all(self):
        for p in self._clip_paths:
            if os.path.exists(p):
                files.download(p)

    def save_bible(self, path: str = "/content/character_bible.json"):
        self.bible.save(path)


# ══════════════════════════════════════════════════════════════════════════════
# STORYBOARD RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_storyboard(
    storyboard: List[dict],
    seed_image_path: Optional[str] = None,
    base_seed: int = 42,
    output_dir: str = "/content/ComfyUI/output/storyboard",
    show_previews: bool = True,
) -> List[str]:
    """
    Run a storyboard (list of dicts with 'prompt' key) through generate_clip()
    with scene continuity (last frame -> next clip seed image).

    Each storyboard entry can optionally include:
      - camera_lora: filename of camera LoRA to apply
      - shot_data: dict with motion/camera metadata for adaptive strength

    Returns list of output video paths.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    clip_paths = []
    current_image = None
    prev_shot_success = True

    if seed_image_path and os.path.exists(seed_image_path):
        current_image = load_image_tensor(seed_image_path)
        print(f"[Storyboard] Seed image loaded: {seed_image_path}")

    for idx, scene in enumerate(storyboard):
        shot_prompt = scene.get("prompt", "")
        camera_lora = scene.get("camera_lora", None)
        shot_data = scene.get("shot_data", {})
        prev_shot = scene.get("prev_shot", None)

        current_seed = base_seed + idx * 100

        # Calculate adaptive image strength
        if USE_ADAPTIVE_STRENGTH and idx > 0 and shot_data:
            img_str = calculate_adaptive_strength(
                shot_data, prev_shot, prev_shot_success,
                ANCHOR_STRENGTH_HIGH, ANCHOR_STRENGTH_LOW)
        else:
            img_str = IMAGE_STRENGTH if current_image is not None else 0.0

        print(f"\n[Storyboard] Shot {idx + 1}/{len(storyboard)} | seed={current_seed}")
        print(f"   Prompt: {shot_prompt[:120]}...")

        try:
            clip_path = generate_clip(
                image_tensor=current_image,
                prompt=shot_prompt,
                neg_prompt=build_negative_prompt_enhanced() if USE_NEGATIVE_PROMPT_EXPANSION else NEGATIVE_PROMPT,
                width=WIDTH,
                height=HEIGHT,
                frames=FRAMES,
                fps=FPS,
                seed=current_seed,
                image_strength=img_str,
                camera_lora_file=camera_lora,
                output_prefix=f"SB_{idx + 1:03d}",
            )
            prev_shot_success = True
        except Exception as e:
            print(f"   Shot {idx + 1} failed: {e}")
            prev_shot_success = False
            continue

        dest = os.path.join(output_dir, f"shot_{idx + 1:03d}.mp4")
        shutil.copy2(clip_path, dest)
        clip_paths.append(dest)

        if USE_SCENE_CONTINUITY:
            last_frame = get_last_frame_tensor(dest)
            current_image = last_frame if last_frame is not None else current_image

        if show_previews:
            display_video(dest)

    print(f"\n[Storyboard] Complete: {len(clip_paths)} clips generated.")
    return clip_paths


# ── Example STORY_BEATS for InfiniteFlowEngine ────────────────────────────────
STORY_BEATS = [
    "A woman arrives at the entrance of a dimly lit apartment building at night, "
    "pushing through the glass door, rain dripping from her jacket",

    "She takes the elevator, watching the floor numbers tick upward, "
    "her reflection ghostly in the steel doors",

    "She unlocks her front door and steps inside the dark apartment, "
    "not turning on the lights, setting her keys on the counter quietly",

    "She moves to the kitchen, opens the fridge, stares blankly at the shelves, "
    "the cold blue light illuminating her face",

    "She notices a note on the kitchen table, picks it up slowly, "
    "her expression shifting from curiosity to alarm",

    "She grabs her phone, dials a number, presses it to her ear. "
    "No answer. She tries again. Silence.",
]


# ── Build STORYBOARD from SCENE_JSON using build_shot_prompt_pro ──────────────
def build_storyboard_from_json(scene_json: dict) -> List[dict]:
    """Parse SCENE_JSON into a STORYBOARD list using build_shot_prompt_pro."""
    storyboard = []
    shots = scene_json.get("story_action", {}).get("shots", [])
    for idx, shot in enumerate(shots):
        prev_shot = shots[idx - 1] if idx > 0 else None
        full_prompt = build_shot_prompt_pro(shot, scene_json, idx)
        camera_lora_key, camera_lora_file = get_camera_lora_for_shot(shot)
        storyboard.append({
            "id": f"shot_{idx + 1:02d}",
            "prompt": full_prompt,
            "shot_data": shot,
            "camera_lora": camera_lora_file if USE_MOTION_LORAS else None,
            "prev_shot": prev_shot,
        })
    return storyboard


print("InfiniteFlowEngine + Storyboard runner defined.")
print(f"   Example STORY_BEATS: {len(STORY_BEATS)} beats")
print("   Use build_storyboard_from_json(SCENE_JSON) for PRO JSON storyboard")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 9  ─  RUN
# ══════════════════════════════════════════════════════════════════════════════

# @title  { "single-column": true }
# @markdown ## 9. Run Generation
# @markdown **Mode A**: Single clip (quick test)
# @markdown **Mode B**: Storyboard/Infinite Flow (multi-scene)

# @markdown ### Run Mode
RUN_MODE = "single"  # @param ["single", "storyboard", "infinite_flow"]

def _run_single():
    """Generate a single clip using Cell 6 settings."""
    global SEED
    # Determine prompt
    if not BYPASS_EASY_PROMPT and USER_INPUT.strip():
        print("Running EasyPromptEngine...")
        engine = EasyPromptEngine(LLM_MODEL, offline=False, keep_loaded=False)
        pos_prompt, neg_prompt = engine.generate(
            user_input=USER_INPUT,
            frame_count=FRAMES,
            creativity=CREATIVITY,
            seed=SEED,
            lora_triggers=LORA_TRIGGERS,
        )
        cleanup_memory()
        print(f"\n   EXPANDED: {pos_prompt[:200]}...")
    else:
        pos_prompt = POSITIVE_PROMPT
        neg_prompt = NEGATIVE_PROMPT

    # Load image if provided
    img_tensor = None
    if IMAGE_PATH:
        img_tensor = load_image_tensor(IMAGE_PATH)
        if img_tensor is not None:
            print(f"   Image loaded: {IMAGE_PATH}")
        else:
            print(f"   Image not found: {IMAGE_PATH} -> T2V mode")

    # Generate
    output = generate_clip(
        image_tensor=img_tensor,
        prompt=pos_prompt,
        neg_prompt=neg_prompt,
        width=WIDTH,
        height=HEIGHT,
        frames=FRAMES,
        fps=FPS,
        seed=SEED,
        image_strength=IMAGE_STRENGTH,
    )

    if SHOW_PREVIEWS:
        display_video(output)
    if DOWNLOAD_AFTER_GENERATE:
        files.download(output)

    # Auto-increment seed for next run
    if AUTO_INCREMENT_SEED:
        SEED += 1

    return output


def _run_storyboard():
    """Run from SCENE_JSON using build_shot_prompt_pro."""
    storyboard = build_storyboard_from_json(SCENE_JSON)
    print(f"   Parsed {len(storyboard)} shots from SCENE_JSON")
    clip_paths = run_storyboard(
        storyboard=storyboard,
        seed_image_path=IMAGE_PATH,
        base_seed=SEED,
        show_previews=SHOW_PREVIEWS,
    )
    if len(clip_paths) > 1:
        final = concatenate_clips(
            clip_paths, "/content/ComfyUI/output/storyboard/final.mp4")
        print(f"\n   Final video: {final}")
        if SHOW_PREVIEWS:
            display_video(final)
    return clip_paths


def _run_infinite_flow():
    """Run InfiniteFlowEngine with STORY_BEATS."""
    bible = CharacterBible()
    if CHARACTER_DESCRIPTION:
        bible.add(CHARACTER_NAME, description=CHARACTER_DESCRIPTION)

    engine = InfiniteFlowEngine(
        character_bible=bible,
        llm_model=LLM_MODEL,
        vision_model=VISION_MODEL,
        width=WIDTH,
        height=HEIGHT,
        frames=FRAMES,
        fps=FPS,
        image_strength=IMAGE_STRENGTH,
        creativity=CREATIVITY,
        lora_triggers=LORA_TRIGGERS,
        use_vision=USE_VISION,
        use_tiled_vae=USE_TILED_VAE,
        offline=False,
        output_dir="/content/ComfyUI/output/infinite_flow",
        show_previews=SHOW_PREVIEWS,
    )

    clip_paths = engine.run(
        story_beats=STORY_BEATS,
        seed_image_path=IMAGE_PATH,
        base_seed=SEED,
    )

    engine.save_bible("/content/character_bible.json")

    if len(clip_paths) > 1:
        final = engine.concat_all("full_video.mp4")
        print(f"\n   Final video: {final}")
        if SHOW_PREVIEWS:
            display_video(final)

    return clip_paths


# ── Execute ───────────────────────────────────────────────────────────────────
try:
    if RUN_MODE == "single":
        _run_single()
    elif RUN_MODE == "storyboard":
        _run_storyboard()
    elif RUN_MODE == "infinite_flow":
        _run_infinite_flow()
    else:
        print(f"Unknown RUN_MODE: {RUN_MODE}")
except torch.cuda.OutOfMemoryError:
    cleanup_memory()
    print("\n--- OUT OF MEMORY ---")
    print("Quick fixes:")
    print("  1. Reduce FRAMES (try 73 for T4)")
    print("  2. Reduce resolution (try 768x432)")
    print("  3. Set USE_TILED_VAE = True")
    print("  4. Set USE_CHUNK_FF = True")
    print("  5. Restart runtime and run again")
except Exception as e:
    print(f"\nError: {type(e).__name__}: {e}")
    print("If models are missing, run Cell 2 first.")
    import traceback
    traceback.print_exc()
