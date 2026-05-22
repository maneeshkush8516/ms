# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  LTX-2 STUDIO PRO — Complete Unified Pipeline v1.0                         ║
# ║  Merges: LTX_PRO + LTX2_InfiniteFlow_12Scene + LTX2_Infinite_Flow_PRO_v2  ║
# ║  15 cells, single file, no duplicate definitions                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


# ══════════════════════════════════════════════════════════════════════════════
# CELL 1  —  ENVIRONMENT SETUP
# ══════════════════════════════════════════════════════════════════════════════

# @title  { "single-column": true }
# @markdown ## 💥 1. Prepare Environment & Install Custom Nodes
# @markdown Clones all required ComfyUI repos including:
# @markdown - **LTX2EasyPrompt-LD** (LTX2PromptArchitect + LTX2VisionDescribe)
# @markdown - **LTX2-Master-Loader** (LTX2MasterLoaderLD 10-slot LoRA stacker)
# @markdown - **ComfyUI-VideoHelperSuite** (VHS_VideoCombine output node)
# @markdown - **ComfyUI-LTXVideo** (tiled VAE decode + AV helpers)

import os, sys, subprocess

# ── Base Python packages ──────────────────────────────────────────────────────
subprocess.run(["pip", "install", "torch", "torchvision", "torchaudio"], check=False, capture_output=True)

os.chdir("/content")
from IPython.display import clear_output
clear_output()

subprocess.run(["pip", "install", "-q", "torchsde", "einops", "diffusers", "accelerate", "nest_asyncio"], check=False, capture_output=True)
subprocess.run(["pip", "install", "-q", "av", "spandrel", "albumentations", "onnx", "opencv-python", "onnxruntime"], check=False, capture_output=True)
subprocess.run(["pip", "install", "-q", "imageio", "imageio-ffmpeg", "moviepy"], check=False, capture_output=True)
subprocess.run(["pip", "install", "-q", "transformers>=4.43.0", "accelerate", "qwen-vl-utils", "huggingface_hub"], check=False, capture_output=True)

# ── ComfyUI (pinned branch) ───────────────────────────────────────────────────
subprocess.run(["git"] + "clone --branch ComfyUI_22_01_2026_v0.10.0 https://github.com/Isi-dev/ComfyUI.git".split(), check=True, capture_output=True)
subprocess.run(["pip", "install", "-r", "/content/ComfyUI/requirements.txt", "-q"], check=False, capture_output=True)
clear_output()

# ── Custom nodes ──────────────────────────────────────────────────────────────
os.chdir("/content/ComfyUI/custom_nodes")

subprocess.run(["git"] + "clone --branch kj_1.2.6                https://github.com/Isi-dev/ComfyUI_KJNodes".split(), check=True, capture_output=True)
subprocess.run(["git"] + "clone --branch ComfyUI_GGUF_22_01_2026  https://github.com/Isi-dev/ComfyUI_GGUF.git".split(), check=True, capture_output=True)
subprocess.run(["git"] + "clone https://github.com/Lightricks/ComfyUI-LTXVideo.git".split(), check=True, capture_output=True)
subprocess.run(["git"] + "clone https://github.com/seanhan19911990-source/LTX2EasyPrompt-LD.git".split(), check=True, capture_output=True)
subprocess.run(["git"] + "clone https://github.com/seanhan19911990-source/LTX2-Master-Loader.git".split(), check=True, capture_output=True)
subprocess.run(["git"] + "clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git".split(), check=True, capture_output=True)

os.chdir("/content/ComfyUI/custom_nodes/ComfyUI_KJNodes")
subprocess.run(["pip", "install", "-r", "requirements.txt", "-q"], check=False, capture_output=True)

os.chdir("/content/ComfyUI/custom_nodes/ComfyUI_GGUF")
subprocess.run(["pip", "install", "-r", "requirements.txt", "-q"], check=False, capture_output=True)

os.chdir("/content/ComfyUI/custom_nodes/ComfyUI-LTXVideo")
subprocess.run(["pip", "install", "-r", "requirements.txt", "-q"], check=False, capture_output=True)

os.chdir("/content/ComfyUI/custom_nodes/LTX2EasyPrompt-LD")
subprocess.run(["pip", "install", "-r", "requirements.txt", "-q"], check=False, capture_output=True)

os.chdir("/content/ComfyUI/custom_nodes/LTX2-Master-Loader")
subprocess.run(["pip", "install", "-r", "requirements.txt", "-q"], check=False, capture_output=True)

# ── System tools ──────────────────────────────────────────────────────────────
import subprocess

def install_apt_packages():
    packages = ["aria2", "ffmpeg"]
    try:
        subprocess.run(["apt-get", "-y", "install", "-qq"] + packages,
                       check=True, capture_output=True)
        print("✓ apt packages installed")
    except subprocess.CalledProcessError as e:
        print(f"✗ apt error: {e.stderr.decode().strip() or 'unknown'}")

print("Installing apt packages...")
install_apt_packages()

# ── Final setup ───────────────────────────────────────────────────────────────
os.chdir("/content/ComfyUI")
import os, sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, "/content/ComfyUI")

clear_output()
print("✅ Environment setup complete.")
print("   Custom nodes installed:")
print("   ✓ ComfyUI_KJNodes    (kj_1.2.6)   — ImageResizeKJv2, PathchSageAttentionKJ")
print("   ✓ ComfyUI_GGUF       (22_01_2026)  — UnetLoaderGGUF")
print("   ✓ ComfyUI-LTXVideo   (Lightricks)  — LTXVImgToVideoInplace, tiled VAE")
print("   ✓ LTX2EasyPrompt-LD  (LoRa Daddy)  — LTX2PromptArchitect, LTX2VisionDescribe")
print("   ✓ LTX2-Master-Loader (LoRa Daddy)  — LTX2MasterLoaderLD")
print("   ✓ ComfyUI-VideoHelperSuite          — VHS_VideoCombine")



# ══════════════════════════════════════════════════════════════════════════════
# CELL 2  —  MODEL DOWNLOADS
# ══════════════════════════════════════════════════════════════════════════════

# @title  { "single-column": true }
# @markdown ## 💥 2. Download All Model Weights
# @markdown Uses aria2c (fast parallel download). Skips files already cached.
# @markdown **Pick ONE Gemma encoder** based on your GPU (see comments below).

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
        print(f"  ↳ cached: {filename}")
        return filename
    cmd = ["aria2c", "--console-log-level=error",
           "-c", "-x", "16", "-s", "16", "-k", "1M",
           "-d", dest_dir, "-o", filename]
    if silent:
        cmd += ["--summary-interval=0", "--quiet"]
        print(f"  ↓ {filename}...", end=" ", flush=True)
    cmd.append(url)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\n  ❌ Failed: {result.stderr.strip()}")
        return False
    if silent:
        print("done.")
    return filename

# ── Source base URLs ──────────────────────────────────────────────────────────
KIJAI    = "https://huggingface.co/Kijai/LTXV2_comfy/resolve/main"
KIJAI23  = "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main"
COMFYORG = "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files"
LIGHTRIX = "https://huggingface.co/Lightricks"

print("── Core model downloads ──────────────────────────────────────────────────")

# ── UNet: GGUF Q4_K_M distilled ──────────────────────────────────────────────
dit_model = model_download(
    f"{KIJAI}/diffusion_models/ltx-2-19b-distilled_Q4_K_M.gguf",
    "/content/ComfyUI/models/unet")

# ── Text encoders ─────────────────────────────────────────────────────────────
# Gemma fp4 — Blackwell RTX 5000. Use fp8 for T4/A100 (uncomment below).
text_encoder_model = model_download(
    f"{COMFYORG}/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
    "/content/ComfyUI/models/text_encoders")
# Gemma fp8 — T4 / A100 / RTX 3000-4000 (uncomment if fp4 OOMs):
# text_encoder_model = model_download(
#     f"{COMFYORG}/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors",
#     "/content/ComfyUI/models/text_encoders")

text_encoder2_model = model_download(
    f"{KIJAI}/text_encoders/ltx-2-19b-embeddings_connector_distill_bf16.safetensors",
    "/content/ComfyUI/models/text_encoders")

# ── VAEs ──────────────────────────────────────────────────────────────────────
vae_model = model_download(
    f"{KIJAI}/VAE/LTX2_video_vae_bf16.safetensors",
    "/content/ComfyUI/models/vae")

vae_audio_model = model_download(
    f"{KIJAI}/VAE/LTX2_audio_vae_bf16.safetensors",
    "/content/ComfyUI/models/vae")

taeltx2_model = model_download(
    f"{KIJAI23}/vae/taeltx2_3.safetensors",
    "/content/ComfyUI/models/vae")

# ── Spatial upscaler ──────────────────────────────────────────────────────────
upscaler_model = model_download(
    f"{LIGHTRIX}/LTX-2/resolve/main/ltx-2-spatial-upscaler-x2-1.0.safetensors",
    "/content/ComfyUI/models/latent_upscale_models")

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

LORA_DIR = "/content/ComfyUI/models/loras"
os.makedirs(LORA_DIR, exist_ok=True)
print(f"\n── LoRA batch download ({len(LORA_URLS)} files) ─────────────────────────────────")
for name, url in LORA_URLS.items():
    r = model_download(url, LORA_DIR)
    print(f"   {'✅' if r else '❌'}  {name}")

print("\n✅ All model files downloaded.")



# ══════════════════════════════════════════════════════════════════════════════
# CELL 3  —  CORE IMPORTS & UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

# @title  { "single-column": true }
# @markdown ## 💥 3. Imports, Helpers & Utilities

import gc, re, json, time, shutil, warnings, subprocess, asyncio
import numpy as np
import torch
import cv2
from PIL import Image
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any, Union, Sequence, Mapping
from base64 import b64encode
from IPython.display import display, HTML, Image as IPImage, clear_output
from google.colab import files
from tqdm.notebook import tqdm
from moviepy.editor import VideoFileClip, concatenate_videoclips, CompositeVideoClip

warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, "/content/ComfyUI")

# ── ComfyUI core ──────────────────────────────────────────────────────────────
from nodes import NODE_CLASS_MAPPINGS, LoraLoaderModelOnly
import folder_paths

# ── VRAM helpers ──────────────────────────────────────────────────────────────
def cleanup_memory(verbose: bool = False) -> None:
    """Enhanced memory cleanup including ipc_collect for fragmentation."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
    if verbose:
        _print_vram()

# Alias for VisionDescribeEngine / EasyPromptEngine compatibility
_vram_free = cleanup_memory

def _print_vram() -> None:
    if not torch.cuda.is_available():
        return
    used  = torch.cuda.memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    pct   = used / total * 100 if total > 0 else 0
    filled = int(20 * used / total) if total > 0 else 0
    bar   = "█" * filled + "░" * (20 - filled)
    print(f"   💾 VRAM [{bar}] {used:.1f}/{total:.1f} GB ({pct:.1f}%)")

# ── ComfyUI node output accessor ──────────────────────────────────────────────
def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]

# ── Tensor / image conversion ─────────────────────────────────────────────────
def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL → ComfyUI NHWC float tensor."""
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)

def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """ComfyUI NHWC tensor → PIL."""
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

# ── Video display & saving ────────────────────────────────────────────────────
def display_video(path: str) -> None:
    if not path or not os.path.exists(path):
        print(f"   ⚠️  Not found: {path}")
        return
    data = b64encode(open(path, "rb").read()).decode()
    display(HTML(
        '<video width=800 controls autoplay loop muted>'
        f'<source src="data:video/mp4;base64,{data}" type="video/mp4">'
        '</video>'
    ))

def save_video_from_components(video_obj, prefix="LTX-Studio-PRO") -> str:
    """Save a ComfyUI video object and return the output path."""
    from comfy_api.latest import Types
    w, h = video_obj.get_dimensions()
    folder, fname, ctr, _, _ = folder_paths.get_save_image_path(
        prefix, folder_paths.get_output_directory(), w, h)
    ext  = Types.VideoContainer.get_extension("auto")
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
    print(f"   ✓ Concatenated → {output_path}")
    return output_path

# ── ComfyUI async node loader ──────────────────────────────────────────────────
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
            print(f"   ⚠️  Some nodes failed: {[str(n) for n in failed]}")
    try:
        asyncio.run(_load())
    except RuntimeError:
        nest_asyncio.apply()
        asyncio.get_event_loop().run_until_complete(_load())
    _NODES_LOADED = True

def validate_model_files(model_dict: dict) -> bool:
    """Check required model files exist in ComfyUI folder_paths."""
    folder_map = {
        "unet"    : "unet",
        "clip1"   : "text_encoders",
        "clip2"   : "text_encoders",
        "vae_vid" : "vae",
        "vae_aud" : "vae",
        "upscaler": "latent_upscale_models",
    }
    ok = True
    for label, filename in model_dict.items():
        fk    = folder_map.get(label, "loras")
        found = any(
            os.path.exists(os.path.join(base, filename))
            for base in folder_paths.get_folder_paths(fk)
        )
        print(f"   {'✅' if found else '❌'} [{label:9s}] {filename}")
        if not found:
            ok = False
    return ok

def save_metadata_sidecar(output_path: str, meta: dict) -> str:
    """Write a .json sidecar file next to the generated video."""
    sidecar = os.path.splitext(output_path)[0] + "_meta.json"
    try:
        with open(sidecar, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"   📄 Metadata: {sidecar}")
    except Exception as e:
        print(f"   ⚠️  Could not save metadata: {e}")
    return sidecar

def upload_image(save_dir: str = "/content/ComfyUI/input") -> Optional[str]:
    os.makedirs(save_dir, exist_ok=True)
    uploaded = files.upload()
    for fname, data in uploaded.items():
        path = os.path.join(save_dir, fname)
        with open(path, "wb") as f:
            f.write(data)
        print(f"   ✓ Saved: {path}")
        return path
    return None

def _load_audio_vae(vae_name: str):
    """Load audio VAE. Prefers VAELoaderKJ (main_device, fp16), falls back to VAELoader."""
    if "VAELoaderKJ" in NODE_CLASS_MAPPINGS:
        return NODE_CLASS_MAPPINGS["VAELoaderKJ"]().load_vae(
            vae_name=vae_name, device="main_device", weight_dtype="fp16")
    return NODE_CLASS_MAPPINGS["VAELoader"]().load_vae(vae_name=vae_name)

print("✅ Imports & utilities ready.")



# ══════════════════════════════════════════════════════════════════════════════
# CELL 4  —  PERFORMANCE PATCHES & LORA SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

# @title  { "single-column": true }
# @markdown ## 💥 4. Performance Patches & LoRA System

def apply_sage_attention(unet):
    """
    Apply PathchSageAttentionKJ (KJNodes) for flash-attention-style speedup.
    Falls back silently if node is not available or USE_SAGE_ATTENTION is False.
    """
    if not USE_SAGE_ATTENTION:
        return unet
    if "PathchSageAttentionKJ" not in NODE_CLASS_MAPPINGS:
        print("   ⚠️  PathchSageAttentionKJ not found — skipping sage attention.")
        return unet
    try:
        node  = NODE_CLASS_MAPPINGS["PathchSageAttentionKJ"]()
        fn    = getattr(node, node.FUNCTION)
        unet  = get_value_at_index(fn(model=unet), 0)
        print("   ✓ SageAttention patch applied (PathchSageAttentionKJ)")
    except Exception as e:
        print(f"   ⚠️  SageAttention failed ({e}) — continuing without it.")
    return unet

def apply_chunk_ff(unet):
    """
    Apply LTXVChunkFeedForward for memory-efficient chunk-based feedforward.
    Falls back silently if node is not available or USE_CHUNK_FF is False.
    """
    if not USE_CHUNK_FF:
        return unet
    if "LTXVChunkFeedForward" not in NODE_CLASS_MAPPINGS:
        print("   ⚠️  LTXVChunkFeedForward not found — skipping chunk FF.")
        return unet
    try:
        node  = NODE_CLASS_MAPPINGS["LTXVChunkFeedForward"]()
        fn    = getattr(node, node.FUNCTION)
        unet  = get_value_at_index(fn(model=unet), 0)
        print("   ✓ ChunkFeedForward patch applied (LTXVChunkFeedForward)")
    except Exception as e:
        print(f"   ⚠️  ChunkFeedForward failed ({e}) — continuing without it.")
    return unet

def purge_vram(label: str = "") -> None:
    """
    Purge VRAM after model loading phases.
    Tries LayerUtility: PurgeVRAM V2 first, then torch.cuda.empty_cache() fallback.
    """
    if not PURGE_VRAM_AFTER_MODELS:
        return
    tag = f" [{label}]" if label else ""
    if "LayerUtility: PurgeVRAM V2" in NODE_CLASS_MAPPINGS:
        try:
            node = NODE_CLASS_MAPPINGS["LayerUtility: PurgeVRAM V2"]()
            fn   = getattr(node, node.FUNCTION)
            fn()
            print(f"   ✓ VRAM purged via PurgeVRAM V2{tag}")
            return
        except Exception as e:
            print(f"   ⚠️  PurgeVRAM V2 failed ({e}) — using torch fallback.")
    cleanup_memory()
    print(f"   ✓ VRAM cleared via torch.cuda.empty_cache{tag}")

# ── LoRA file dicts ───────────────────────────────────────────────────────────
_IC_LORA_FILES: Dict[str, str] = {
    "none":     "None",
    "detailer": "ltx-2-19b-ic-lora-detailer.safetensors",
    "canny":    "ltx-2-19b-ic-lora-canny-control.safetensors",
    "depth":    "ltx-2-19b-ic-lora-depth-control.safetensors",
    "pose":     "ltx-2-19b-ic-lora-pose-control.safetensors",
}

_CAMERA_LORA_FILES: Dict[str, str] = {
    "none":        "None",
    "dolly-in":    "ltx-2-19b-lora-camera-control-dolly-in.safetensors",
    "dolly-out":   "ltx-2-19b-lora-camera-control-dolly-out.safetensors",
    "dolly-left":  "ltx-2-19b-lora-camera-control-dolly-left.safetensors",
    "dolly-right": "ltx-2-19b-lora-camera-control-dolly-right.safetensors",
    "jib-up":      "ltx-2-19b-lora-camera-control-jib-up.safetensors",
    "jib-down":    "ltx-2-19b-lora-camera-control-jib-down.safetensors",
    "static":      "ltx-2-19b-lora-camera-control-static.safetensors",
}

def _build_lora_stack(ic_lora: str, ic_strength: float,
                      camera_lora: str, camera_strength: float) -> List[Dict]:
    """Build the 10-slot LoRA stack from IC and Camera dropdown selections."""
    ic_file  = _IC_LORA_FILES.get(ic_lora.lower(), "None")
    cam_file = _CAMERA_LORA_FILES.get(camera_lora.lower(), "None")
    stack = [
        {"on": ic_file  != "None", "lora": ic_file,  "guard": False, "strength": ic_strength},
        {"on": cam_file != "None", "lora": cam_file, "guard": False, "strength": camera_strength},
    ]
    for _ in range(8):
        stack.append({"on": False, "lora": "None", "guard": False, "strength": 1.0})
    return stack

def apply_lora_stack(unet, clip_model,
                     lora_stack: Optional[List[Dict]] = None,
                     lora_stack_json: Optional[str] = None):
    """
    Apply LoRA stack via LTX2MasterLoaderLD node when available.
    Falls back to manual LoraLoaderModelOnly loop if the node is missing.
    Returns: (unet, clip_model)
    """
    stack = lora_stack or []
    active = [s for s in stack
              if s.get("on") and s.get("lora") not in (None, "None", "")]

    if not active:
        print("   ℹ️  No active LoRAs in stack — skipping.")
        return unet, clip_model

    # [263] LTX2MasterLoaderLD node (preferred)
    if "LTX2MasterLoaderLD" in NODE_CLASS_MAPPINGS:
        print(f"   [MasterLoader] {len(active)} LoRA(s) via LTX2MasterLoaderLD…")
        try:
            node   = NODE_CLASS_MAPPINGS["LTX2MasterLoaderLD"]()
            fn     = getattr(node, node.FUNCTION)
            result = fn(
                model=unet,
                clip=clip_model,
                stack_data=lora_stack_json or json.dumps(stack),
            )
            unet = get_value_at_index(result, 0)
            print("   [MasterLoader] ✓  Stack applied.")
            return unet, clip_model
        except TypeError as e:
            print(f"   [MasterLoader] ⚠️  kwarg mismatch ({e}) — manual fallback.")
        except Exception as e:
            print(f"   [MasterLoader] ⚠️  Node failed ({e}) — manual fallback.")

    # Fallback: LoraLoaderModelOnly loop
    print(f"   [MasterLoader] {len(active)} LoRA(s) via manual loop…")
    for slot in active:
        name, strength, guard = slot["lora"], slot.get("strength", 1.0), slot.get("guard", False)
        try:
            ll   = LoraLoaderModelOnly()
            unet = ll.load_lora_model_only(unet, name, strength)[0]
            print(f"      ✓ {name} @ {strength}")
        except Exception as e:
            if guard:
                print(f"      ⚠️  {name} skipped (guard): {e}")
            else:
                print(f"      ❌ {name} failed: {e}")

    return unet, clip_model

print("✅ Performance patches & LoRA system ready.")



# ══════════════════════════════════════════════════════════════════════════════
# CELL 5  —  VISION DESCRIBE ENGINE
# ══════════════════════════════════════════════════════════════════════════════

# @title  { "single-column": true }
# @markdown ## 💥 5. Vision Describe Engine (Standalone Qwen2.5-VL)

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
        self.offline   = offline

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
            try:   source = snapshot_download(hf_id)
            except: source = hf_id
        else:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            source = hf_id

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"   [VisionDescribe] Loading {self.model_key} ({image.size}) …")
        processor = AutoProcessor.from_pretrained(source, local_files_only=self.offline)
        model     = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            source, device_map="auto", torch_dtype=dtype,
            local_files_only=self.offline)
        model.eval()

        messages = [
            {"role": "system", "content":
             "You are an image analysis tool. Describe exactly what you see in plain prose."},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text":  self.PROMPT},
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
        _vram_free()
        print(f"   [VisionDescribe] ✓  {len(desc.split())} words.")
        return desc

print("✅ VisionDescribeEngine defined.")



# ══════════════════════════════════════════════════════════════════════════════
# CELL 6  —  EASY PROMPT ENGINE
# ══════════════════════════════════════════════════════════════════════════════

# @title  { "single-column": true }
# @markdown ## 💥 6. Easy Prompt Engine (Standalone LLM Expansion)

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
    if any(w in c for w in ["pussy", "cock", "penis", "vagina", "nude", "naked", "nipple", "breast"]):
        extras.append("censored, mosaic, pixelated, black bar, blurred genitals")
    if any(w in c for w in ["close-up", "close up", "portrait", "headshot"]):
        extras.append("wide angle distortion, fish eye")
    elif any(w in c for w in ["wide shot", "wide angle", "aerial", "bird's-eye"]):
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
3. Character description — age MUST be a specific number (e.g. "a 28-year-old woman"), body type, hair, skin, clothing. Use exact words from the user.
4. Scene & environment (location, time of day, lighting, colour palette, atmosphere)
5. Action & motion — continuous present-tense sequence.
6. Camera movement — prose only, no screenplay brackets like (HOLD) or (DOWN 10 degrees).
7. Audio — max 2 ambient sounds active at once, woven as prose. Dialogue as inline prose with attribution, never as [DIALOGUE:] tags.

RULES:
- Present tense throughout.
- 8-12 sentences of dense flowing prose — no bullet lists.
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
        self.model_size  = model_size
        self.offline     = offline
        self.keep_loaded = keep_loaded
        self._tok        = None
        self._model      = None
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
            try:   source = snapshot_download(hf_id)
            except: source = hf_id
        else:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            source = hf_id
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"   [EasyPrompt] Loading {key} …")
        self._tok   = AutoTokenizer.from_pretrained(source, local_files_only=self.offline)
        self._model = AutoModelForCausalLM.from_pretrained(
            source, device_map="auto", torch_dtype=dtype,
            trust_remote_code=True, local_files_only=self.offline)
        self._model.config.use_cache = True
        self._model.eval()
        self._loaded_key = key
        print("   [EasyPrompt] Loaded.")

    def _unload(self):
        if self._model is not None:
            try: self._model.to("cpu")
            except: pass
        self._model = None; self._tok = None; self._loaded_key = None
        _vram_free()
        print("   [EasyPrompt] VRAM cleared.")

    def _stop_ids(self) -> List[int]:
        delims = ["assistant", "user", "system", "<|eot_id|>", "<|end_of_turn|>",
                  "<|im_end|>", "<end_of_turn>", "[/INST]", "### Human", "### Assistant"]
        ids = [self._tok.eos_token_id]
        for s in delims:
            enc = self._tok.encode(s, add_special_tokens=False)
            if enc and enc[0] not in ids: ids.append(enc[0])
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
        user_input:      str,
        frame_count:     int   = 121,
        creativity:      float = 0.9,
        seed:            int   = -1,
        scene_context:   str   = "",
        lora_triggers:   str   = "",
        character_bible: str   = "",
    ) -> tuple:
        """
        Returns (positive_prompt, negative_prompt).
        character_bible injected as [CHARACTER BIBLE - NON-NEGOTIABLE] block.
        """
        self._load()

        real_seconds  = frame_count / 25.0
        action_count  = max(1, min(10, round(real_seconds / 4)))
        token_budget  = max(256, min(1200, action_count * 120))
        max_tokens    = int(token_budget * 1.05)
        min_tokens    = int(token_budget * 0.75)

        if seed != -1:
            torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

        ordinal = {2:"2nd",3:"3rd"}.get(action_count, f"{action_count}th")
        pacing  = (
            f"This clip is {real_seconds:.0f}s. Write EXACTLY {action_count} "
            f"distinct action{'s' if action_count>1 else ''}. "
            f"HARD STOP after the {ordinal} action. "
            f"Write ~{token_budget} tokens. No trailing notes or brackets after the last sentence."
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
            {"role": "user",   "content": user_content},
        ]

        is_qwen3 = "Qwen3" in self.MODELS.get(self.model_size, "")
        raw = self._tok.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True,
            **({"enable_thinking": False} if is_qwen3 else {}))

        if hasattr(raw, "input_ids"): input_ids = raw.input_ids.to(self._model.device)
        elif isinstance(raw, dict):   input_ids = raw["input_ids"].to(self._model.device)
        elif isinstance(raw, list):   input_ids = torch.tensor([raw], dtype=torch.long).to(self._model.device)
        else:                          input_ids = raw.to(self._model.device)

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

        print(f"   [EasyPrompt] ✓  {len(result.split())} words generated.")
        return result, neg

print("✅ EasyPromptEngine defined.")



# ══════════════════════════════════════════════════════════════════════════════
# CELL 7  —  CHARACTER BIBLE
# ══════════════════════════════════════════════════════════════════════════════

# @title  { "single-column": true }
# @markdown ## 💥 7. Character Bible (Cross-Scene Consistency)

class CharacterBible:
    """
    Records named character attributes and serialises them as a prompt-injection
    block. Passed to EasyPromptEngine as `character_bible` so the LLM receives
    a hard [NON-NEGOTIABLE] constraint preventing attribute drift across all clips.
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
        if not self._chars: return ""
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
        with open(path, "w") as f: json.dump(self._chars, f, indent=2)
        print(f"   [Bible] Saved -> {path}")

    def load(self, path: str):
        with open(path) as f: self._chars = json.load(f)
        print(f"   [Bible] Loaded from {path}")

    def __repr__(self):
        return f"CharacterBible({self.names()})"

print("✅ CharacterBible defined.")



# ══════════════════════════════════════════════════════════════════════════════
# CELL 8  —  COMFYUI NODE WRAPPERS (EasyPrompt + VisionDescribe)
# ══════════════════════════════════════════════════════════════════════════════

# @title  { "single-column": true }
# @markdown ## 💥 8. ComfyUI Node Wrappers (EasyPrompt + VisionDescribe)

_LLM_LABEL_MAP = {
    "8B":  "8B - NeuralDaredevil (High Quality)",
    "3B":  "3B - Llama-3.2 Abliterated (Low VRAM)",
    "14B": "14B - Qwen3 Abliterated (High VRAM)",
}
_VISION_LABEL_MAP = {
    "3B-fast": "Qwen2.5-VL-3B - Fast (huihui abliterated)",
    "7B-nsfw": "Qwen2.5-VL-7B - Better NSFW (prithiv caption)",
}
_CREATIVITY_MAP = {
    0.7: "0.7 - Literal & Grounded",
    0.9: "0.9 - Balanced Professional",
    1.1: "1.1 - Artistic Expansion",
}

def _creativity_label(c: float) -> str:
    closest = min(_CREATIVITY_MAP.keys(), key=lambda x: abs(x - c))
    return _CREATIVITY_MAP[closest]


def run_easy_prompt(user_input: str, frame_count: int, seed: int,
                    scene_context: str = "",
                    llm_model_override: str = None) -> Tuple[str, str]:
    """
    Calls LTX2PromptArchitect node to expand user input into a cinematic prompt.
    Falls back to EasyPromptEngine standalone if node is unavailable.
    Returns: (positive_prompt, negative_prompt)
    """
    _model = llm_model_override if llm_model_override is not None else LLM_MODEL

    if "LTX2PromptArchitect" in NODE_CLASS_MAPPINGS:
        print(f"   [EasyPrompt] LLM={_model} | creativity={CREATIVITY} | frames={frame_count}")
        try:
            node = NODE_CLASS_MAPPINGS["LTX2PromptArchitect"]()
            result = node.generate(
                bypass=False,
                user_input=user_input,
                creativity=_creativity_label(CREATIVITY),
                seed=seed,
                invent_dialogue=INVENT_DIALOGUE,
                keep_model_loaded=False,
                offline_mode=False,
                frame_count=frame_count,
                model=_LLM_LABEL_MAP.get(_model, "8B - NeuralDaredevil (High Quality)"),
                local_path_8b="",
                local_path_3b="",
                local_path_14b="",
                scene_context=scene_context,
                lora_triggers=LORA_TRIGGERS,
            )
            prompt     = result[0]
            neg_prompt = result[2]
            print(f"   [EasyPrompt] ✓  {len(prompt.split())} words generated.")
            cleanup_memory()
            return prompt, neg_prompt
        except Exception as e:
            print(f"   [EasyPrompt] Node failed ({e}) — falling back to standalone engine.")

    # Standalone fallback
    print(f"   [EasyPrompt] Using standalone EasyPromptEngine (model={_model})")
    try:
        engine = EasyPromptEngine(model_size=_model, offline=False, keep_loaded=False)
        return engine.generate(
            user_input=user_input,
            frame_count=frame_count,
            creativity=CREATIVITY,
            seed=seed,
            scene_context=scene_context,
            lora_triggers=LORA_TRIGGERS,
        )
    except Exception as e:
        print(f"   ⚠️  EasyPromptEngine also failed ({e}) — returning raw input.")
        return user_input, ""


def run_vision_describe(image_tensor: torch.Tensor,
                        character_desc: str = "",
                        use_vision_override: bool = None,
                        vision_model_override: str = None) -> str:
    """
    Calls LTX2VisionDescribe node to analyse the image and return scene context.
    Falls back to VisionDescribeEngine standalone if node is unavailable.
    Returns: scene_context string (empty string on failure).
    """
    _use_v   = use_vision_override   if use_vision_override   is not None else USE_VISION
    _vis_mod = vision_model_override if vision_model_override is not None else VISION_MODEL

    if not _use_v:
        return character_desc

    if "LTX2VisionDescribe" in NODE_CLASS_MAPPINGS:
        print(f"   [VisionDescribe] model={_vis_mod} | image shape={image_tensor.shape}")
        try:
            node = NODE_CLASS_MAPPINGS["LTX2VisionDescribe"]()
            result = node.describe(
                image=image_tensor,
                model_name=_VISION_LABEL_MAP.get(_vis_mod, "Qwen2.5-VL-3B - Fast (huihui abliterated)"),
                offline_mode=False,
                local_path="",
            )
            ctx = result[0]
            if character_desc:
                ctx = character_desc + " " + ctx
            print(f"   [VisionDescribe] ✓  {len(ctx.split())} words.")
            cleanup_memory()
            return ctx
        except Exception as e:
            print(f"   [VisionDescribe] Node failed ({e}) — falling back to standalone engine.")

    # Standalone fallback
    print(f"   [VisionDescribe] Using standalone VisionDescribeEngine (model={_vis_mod})")
    try:
        engine = VisionDescribeEngine(model_key=_vis_mod)
        desc = engine.describe(image_tensor)
        if character_desc:
            desc = character_desc + " " + desc
        return desc
    except Exception as e:
        print(f"   ⚠️  VisionDescribeEngine also failed ({e}) — returning character_desc.")
        return character_desc

print("✅ Node wrappers ready.")



# ══════════════════════════════════════════════════════════════════════════════
# CELL 9  —  CHARACTER CONSISTENCY & JSON STORY SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

# @title  { "single-column": true }
# @markdown ## 💥 9. Character Consistency & JSON Story System

CAMERA_LORA_MAPPING = {
    "dolly_in":       "ltx-2-19b-lora-camera-control-dolly-in.safetensors",
    "dolly_out":      "ltx-2-19b-lora-camera-control-dolly-out.safetensors",
    "dolly_left":     "ltx-2-19b-lora-camera-control-dolly-left.safetensors",
    "dolly_right":    "ltx-2-19b-lora-camera-control-dolly-right.safetensors",
    "jib_up":         "ltx-2-19b-lora-camera-control-jib-up.safetensors",
    "jib_down":       "ltx-2-19b-lora-camera-control-jib-down.safetensors",
    "static":         "ltx-2-19b-lora-camera-control-static.safetensors",
    # Aliases for JSON storyboard shot data compatibility
    "dolly_forward":  "ltx-2-19b-lora-camera-control-dolly-in.safetensors",
    "dolly_backward": "ltx-2-19b-lora-camera-control-dolly-out.safetensors",
    "zoom_in_slow":   "ltx-2-19b-lora-camera-control-dolly-in.safetensors",
    "zoom_in_fast":   "ltx-2-19b-lora-camera-control-dolly-in.safetensors",
    "tilt_up_slight": "ltx-2-19b-lora-camera-control-jib-up.safetensors",
    "tilt_up_dramatic": "ltx-2-19b-lora-camera-control-jib-up.safetensors",
    "tilt_up_reveal": "ltx-2-19b-lora-camera-control-jib-up.safetensors",
    "low_angle_hero": "ltx-2-19b-lora-camera-control-jib-up.safetensors",
}

def build_character_prompt_detailed(character_data: dict) -> str:
    """Build highly detailed character description for consistency."""
    char_name  = character_data["name"]
    appearance = character_data["detailed_appearance"]
    prompt = f"{char_name}: "
    prompt += f"{appearance['face']}, "
    prompt += f"{appearance['hair']}, "
    prompt += f"wearing {appearance['clothing']}, "
    prompt += f"{appearance['build']}, {appearance['skin_tone']}, "
    prompt += f"{appearance['accessories']}. "
    prompt += (f"ALWAYS MAINTAIN: {char_name} has {appearance['face'].split(',')[0]}, "
               f"{appearance['hair'].split(',')[0]}, {appearance['clothing'].split(',')[0]}. ")
    return prompt

def get_character_consistency_prefix(scene_json: dict) -> str:
    """Generate character consistency prefix for ALL prompts."""
    char_prompts = []
    for char in scene_json["main_characters"]:
        char_prompts.append(build_character_prompt_detailed(char))
    consistency_prompt = "CHARACTER CONSISTENCY CRITICAL: " + " | ".join(char_prompts)
    consistency_prompt += " | MAINTAIN EXACT SAME CHARACTER APPEARANCE THROUGHOUT ENTIRE SCENE. NO MORPHING. NO STYLE CHANGES."
    return consistency_prompt

def get_motion_guidance_prompt(shot: dict) -> str:
    """Build motion-specific guidance prompt."""
    motion_intensity = shot.get("motion_intensity", 0.5)
    camera_movement  = shot.get("camera_movement", "static")
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

def get_camera_lora_for_shot(shot: dict) -> Tuple[Optional[str], Optional[str]]:
    """Determine which camera LoRA to use for this shot."""
    camera_movement = shot.get("camera_movement", "static")
    for key in CAMERA_LORA_MAPPING.keys():
        if key in camera_movement:
            return key, CAMERA_LORA_MAPPING[key]
    return None, None

def get_dialogue_for_shot_enhanced(start_time: int, end_time: int,
                                   dialogue_list: list) -> str:
    """Enhanced dialogue injection with voice sync guidance."""
    lines = []
    for entry in dialogue_list:
        if start_time <= entry["time"] < end_time:
            char         = entry["character"]
            text         = entry["dialogue"]
            emotion      = entry.get("emotion", "neutral")
            voice_dir    = entry.get("voice_direction", "")
            lip_sync     = entry.get("lip_sync_emphasis", "medium")
            if char == "The Cave":
                lines.append(f"AUDIO EFFECT: Eerie cave whisper saying '{text}' with hollow reverb and inhuman quality")
            else:
                if lip_sync == "high":
                    lines.append(f"LIP SYNC CRITICAL: {char} speaks '{text}' with {emotion} emotion. {voice_dir}. Mouth movements MUST match dialogue exactly.")
                else:
                    lines.append(f"{char} says '{text}' with {emotion} emotion. {voice_dir}.")
    return " | ".join(lines) if lines else ""

def build_audio_atmosphere_prompt(shot: dict, scene_json: dict,
                                  start_s: int, end_s: int) -> str:
    """Build comprehensive audio prompt including dialogue and SFX."""
    audio_config = scene_json["audio"]
    atmosphere   = f"AUDIO ATMOSPHERE: {audio_config['background_music']}. "
    atmosphere  += f"SOUND EFFECTS: {audio_config['environment_sfx']}. "
    atmosphere  += f"VOICE: {audio_config['voice_processing']}. "
    dialogue_text = get_dialogue_for_shot_enhanced(
        start_s, end_s, scene_json["dialogue_with_timing"])
    if dialogue_text:
        atmosphere += dialogue_text
    return atmosphere

def build_shot_prompt_pro(shot: dict, json_data: dict,
                          shot_index: int, prev_shot_success: bool = True) -> str:
    """PRO prompt builder with character consistency, motion guidance, voice sync."""
    try:
        times   = shot["time"].replace("s", "").split("-")
        start_s = int(times[0])
        end_s   = int(times[1])
    except Exception:
        start_s, end_s = 0, 5

    character_prompt = get_character_consistency_prefix(json_data) if INJECT_CHARACTER_EVERY_SHOT else ""
    action_prompt    = f"SHOT {shot_index + 1}: {shot['action']}. "
    camera_prompt    = f"CAMERA: {shot['camera']}. "
    motion_prompt    = get_motion_guidance_prompt(shot)
    env              = json_data["environment"]
    env_prompt       = (f"ENVIRONMENT: {env['location']}. LIGHTING: {env['lighting']}. "
                        f"TIME: {env['time']}. WEATHER: {env['weather']}. "
                        f"MOOD: {env['mood']}. COLOR PALETTE: {env['color_palette']}. ")
    style_prompt     = f"STYLE: {json_data['video_style']}. "
    audio_prompt     = build_audio_atmosphere_prompt(shot, json_data, start_s, end_s)
    vfx_prompt       = f"VISUAL EFFECTS: {shot.get('visual_effects', 'natural')}. "
    emotion_prompt   = f"EMOTION: {shot.get('emotion', 'neutral')}. FOCUS: {shot.get('character_focus', 'scene')}. "

    final_prompt = (character_prompt + " " + action_prompt + camera_prompt +
                    motion_prompt + emotion_prompt + env_prompt + vfx_prompt +
                    style_prompt + audio_prompt)
    return final_prompt

def calculate_adaptive_strength(shot: dict, prev_shot: Optional[dict],
                                prev_shot_success: bool) -> float:
    """Calculate anchor strength based on motion change and previous success."""
    if not USE_ADAPTIVE_STRENGTH:
        return ANCHOR_STRENGTH_HIGH
    strength = ANCHOR_STRENGTH_HIGH
    if prev_shot:
        motion_change = abs(shot.get("motion_intensity", 0.5) -
                            prev_shot.get("motion_intensity", 0.5))
        if motion_change > 0.4:
            strength -= 0.10
        elif motion_change < 0.2:
            strength += 0.05
        if shot.get("character_focus") != prev_shot.get("character_focus"):
            strength -= 0.05
    if not prev_shot_success:
        strength -= 0.10
    return max(ANCHOR_STRENGTH_LOW, min(ANCHOR_STRENGTH_HIGH, strength))

def build_negative_prompt_enhanced() -> str:
    """Build comprehensive negative prompt."""
    base_neg = NEGATIVE_PROMPT
    if USE_NEGATIVE_PROMPT_EXPANSION:
        char_neg    = "character morphing, face changing, inconsistent character design, different clothing in same scene, style shift, "
        motion_neg  = "motion blur artifacts, jittery movement, unnatural animation, robotic motion, floating characters, "
        audio_neg   = "desynchronized lips, mouth not moving during speech, frozen face during dialogue, mismatched audio, "
        quality_neg = "compression artifacts, pixelation, banding, color shifts, lighting inconsistency, flickering, "
        base_neg    = char_neg + motion_neg + audio_neg + quality_neg + base_neg
    return base_neg

def extract_overlap_anchor_enhanced(video_path: str,
                                    output_folder: str = "/content/ComfyUI/input",
                                    scene_idx: int = 0,
                                    overlap: int = 16) -> Optional[str]:
    """Enhanced anchor extraction with brightness+sharpness quality scoring."""
    if not os.path.exists(video_path):
        print(f"   ❌ Video not found: {video_path}")
        return None
    cap          = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    candidates   = [total_frames - overlap, total_frames - overlap - 2, total_frames - overlap + 2]
    best_frame   = None
    best_score   = 0
    for candidate_idx in candidates:
        if candidate_idx < 0 or candidate_idx >= total_frames:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, candidate_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = cv2.mean(gray)[0]
        laplacian  = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness  = laplacian.var()
        score      = brightness + sharpness * 0.1
        if brightness < 5:
            continue
        if score > best_score:
            best_score = score
            best_frame = frame
    cap.release()
    if best_frame is not None:
        filename  = f"anchor_scene_{scene_idx}.png"
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, best_frame)
        print(f"   ✓ Anchor extracted (quality score: {best_score:.2f})")
        return save_path
    else:
        print("   ✗ No valid anchor frame found")
        return None

def stitch_videos_with_overlap_pro(video_paths: List[str],
                                   output_filename: str = "LTX_Final_PRO.mp4",
                                   overlap_frames: int = 16) -> Optional[str]:
    """Enhanced stitching with audio crossfades and 10000k bitrate."""
    if not video_paths:
        return None
    print(f"\n🧵 Stitching {len(video_paths)} clips with PRO settings...")
    final_clips = []
    for i, path in enumerate(video_paths):
        if os.path.exists(path):
            try:
                clip = VideoFileClip(path)
                if clip.fps != FPS:
                    clip = clip.set_fps(FPS)
                if i < len(video_paths) - 1:
                    duration_to_keep = clip.duration - (overlap_frames / float(FPS))
                    if duration_to_keep > 0:
                        clip = clip.subclip(0, duration_to_keep)
                if clip.audio is not None:
                    clip = clip.audio_fadein(0.1).audio_fadeout(0.1)
                    clip = clip.volumex(0.9)
                final_clips.append(clip)
            except Exception as e:
                print(f"   ⚠️ Skipping corrupted clip {path}: {e}")
    if not final_clips:
        return None
    final_video  = concatenate_videoclips(final_clips, method="compose")
    output_path  = f"/content/ComfyUI/output/{output_filename}"
    print("  Rendering Final Movie (PRO Quality)...")
    final_video.write_videofile(
        output_path, fps=FPS, codec="libx264", audio_codec="aac",
        bitrate="10000k", preset="slow", threads=4, logger=None)
    return output_path

print("✅ Character consistency & JSON story system ready.")



# ══════════════════════════════════════════════════════════════════════════════
# CELL 10  —  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# @title  { "single-column": true }
# @markdown ## 💥 10. Configuration

# ── LLM / Vision ──────────────────────────────────────────────────────────────
LLM_MODEL  = "8B"    # @param ["8B", "3B", "14B"]
CREATIVITY = 0.9     # @param {type:"number"}
INVENT_DIALOGUE    = True   # @param {type:"boolean"}
BYPASS_EASY_PROMPT = False  # @param {type:"boolean"}
LORA_TRIGGERS = ""          # @param {type:"string"}
USE_VISION   = True          # @param {type:"boolean"}
VISION_MODEL = "3B-fast"     # @param ["3B-fast", "7B-nsfw"]
SHOW_PREVIEWS           = True   # @param {type:"boolean"}
DOWNLOAD_AFTER_GENERATE = False  # @param {type:"boolean"}

# ── Video settings ────────────────────────────────────────────────────────────
WIDTH  = 768   # @param {type:"integer"}
HEIGHT = 512   # @param {type:"integer"}
FRAMES = 121   # @param {type:"integer"}
FPS    = 25    # @param {type:"integer"}
SEED                = 47    # @param {type:"integer"}
AUTO_INCREMENT_SEED = True  # @param {type:"boolean"}
IMAGE_PATH = None   # @param {type:"string"}
IMAGE_STRENGTH = 1.0  # @param {type:"number"}

# ── Manual prompts (BYPASS mode) ──────────────────────────────────────────────
USER_INPUT = "a woman walks through a rain-soaked city street at night, neon reflections on the wet pavement, looking over her shoulder"  # @param {type:"string"}
POSITIVE_PROMPT = "Busy city street at night, cinematic, neon reflections on wet pavement, woman walking, bokeh streetlights, moody atmosphere, ultra detailed, professional cinematography, shallow depth of field, film grain"  # @param {type:"string"}
NEGATIVE_PROMPT = "blurry, distorted, low quality, watermark, text, bad anatomy, deformed, grainy, overexposed, underexposed, flickering, motion artifacts, flat lighting"  # @param {type:"string"}

# ── Character Consistency ──────────────────────────────────────────────────────
CHARACTER_IMAGE_PATH = None  # @param {type:"string"}
CHARACTER_STRENGTH = 1.0     # @param {type:"number"}
CHARACTER_CONSISTENCY_MODE = "i2v"  # @param ["i2v", "anchor", "both", "none"]
CHARACTER_NAME = "Character"  # @param {type:"string"}
CHARACTER_DESCRIPTION = ""   # @param {type:"string"}

# ── LoRA ──────────────────────────────────────────────────────────────────────
IC_LORA          = "detailer"  # @param ["none", "detailer", "canny", "depth", "pose"]
IC_LORA_STRENGTH = 0.4         # @param {type:"number"}
CAMERA_LORA          = "none"  # @param ["none", "dolly-in", "dolly-out", "dolly-left", "dolly-right", "jib-up", "jib-down", "static"]
CAMERA_LORA_STRENGTH = 1.0     # @param {type:"number"}

# ── Performance ───────────────────────────────────────────────────────────────
USE_SAGE_ATTENTION      = False  # @param {type:"boolean"}
USE_CHUNK_FF            = False  # @param {type:"boolean"}
PURGE_VRAM_AFTER_MODELS = True   # @param {type:"boolean"}

# ── PRO mode ──────────────────────────────────────────────────────────────────
PRO_MODE      = False    # @param {type:"boolean"}
PRO_STEPS     = 4        # @param {type:"integer"}
PRO_SCHEDULER = "simple" # @param {type:"string"}
PRO_SPLIT_AT  = 2        # @param {type:"integer"}

# ── Model filenames ───────────────────────────────────────────────────────────
UNET_MODEL      = "ltx-2-19b-distilled_Q4_K_M.gguf"
CLIP_NAME1      = "gemma_3_12B_it_fp4_mixed.safetensors"
CLIP_NAME2      = "ltx-2-19b-embeddings_connector_distill_bf16.safetensors"
VAE_VIDEO_MODEL = "LTX2_video_vae_bf16.safetensors"
VAE_AUDIO_MODEL = "LTX2_audio_vae_bf16.safetensors"
UPSCALER_MODEL  = "ltx-2-spatial-upscaler-x2-1.0.safetensors"

# ── Pass 1/2 sampling ─────────────────────────────────────────────────────────
PASS1_SIGMAS  = "1., 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
PASS1_SAMPLER = "euler"           # @param {type:"string"}
PASS1_CFG     = 1.0               # @param {type:"number"}
PASS2_SIGMAS  = "0.909375, 0.725, 0.421875, 0.0"
PASS2_SAMPLER = "gradient_estimation"  # @param {type:"string"}
PASS2_CFG     = 1.0               # @param {type:"number"}
PASS2_SEED    = 0                  # @param {type:"integer"}
# PASS2_SEED=0 is intentional: deterministic Pass 2 noise keeps refinement consistent
# across seeds. Set to SEED+1 if you want varied refinement texture per run.

# ── Tiled VAE ─────────────────────────────────────────────────────────────────
USE_TILED_VAE          = True   # @param {type:"boolean"}
TILED_SPATIAL_TILES    = 2      # @param {type:"integer"}
TILED_SPATIAL_OVERLAP  = 8      # @param {type:"integer"}
TILED_TEMPORAL_LEN     = 48     # @param {type:"integer"}
TILED_TEMPORAL_OVERLAP = 4      # @param {type:"integer"}
TILED_LAST_FRAME_FIX   = False  # @param {type:"boolean"}

# ── Output / continuity ───────────────────────────────────────────────────────
OUTPUT_PREFIX = "LTX-Studio-PRO"  # @param {type:"string"}
USE_SCENE_CONTINUITY = True  # @param {type:"boolean"}

# ── Multi-scene (JSON runner + InfiniteFlow) ──────────────────────────────────
OVERLAP_FRAMES = 16                # @param {type:"integer"}
ANCHOR_STRENGTH_HIGH = 0.85        # @param {type:"number"}
ANCHOR_STRENGTH_LOW = 0.70         # @param {type:"number"}
USE_ADAPTIVE_STRENGTH = True       # @param {type:"boolean"}
USE_CHARACTER_LORAS = True         # @param {type:"boolean"}
INJECT_CHARACTER_EVERY_SHOT = True # @param {type:"boolean"}
USE_MOTION_LORAS = True            # @param {type:"boolean"}
USE_VOICE_SYNC = True              # @param {type:"boolean"}
USE_NEGATIVE_PROMPT_EXPANSION = True  # @param {type:"boolean"}
BASE_PROMPT = "cinematic, ultra detailed, professional quality"  # @param {type:"string"}

# ── Build LoRA stack ──────────────────────────────────────────────────────────
LORA_STACK      = _build_lora_stack(IC_LORA, IC_LORA_STRENGTH, CAMERA_LORA, CAMERA_LORA_STRENGTH)
LORA_STACK_JSON = json.dumps(LORA_STACK)

# ── CharacterBible instance ───────────────────────────────────────────────────
bible = CharacterBible()

_active_count = sum(1 for s in LORA_STACK if s["on"])
print("✅ Configuration ready.")
print(f"   Resolution : {WIDTH}x{HEIGHT}  |  Frames: {FRAMES}  ({FRAMES/FPS:.1f}s @ {FPS}fps)")
print(f"   LLM        : {LLM_MODEL}  |  Vision: {VISION_MODEL}  |  Bypass: {BYPASS_EASY_PROMPT}")
print(f"   IC LoRA    : {IC_LORA} @ {IC_LORA_STRENGTH}  |  Camera: {CAMERA_LORA} @ {CAMERA_LORA_STRENGTH}")
print(f"   Active LoRA slots: {_active_count}/10  |  Pro Mode: {PRO_MODE}")
print(f"   Character mode: {CHARACTER_CONSISTENCY_MODE}  |  seed: {SEED}")


# ══════════════════════════════════════════════════════════════════════════════
# CELL 11  —  generate_pro()
# ══════════════════════════════════════════════════════════════════════════════

# @title  { "single-column": true }
# @markdown ## 💥 11. Define generate_pro()

def generate_pro(
    user_input:              str   = None,
    image_path:              str   = None,
    positive_prompt:         str   = None,
    negative_prompt:         str   = None,
    width:                   int   = None,
    height:                  int   = None,
    frames:                  int   = None,
    fps:                     int   = None,
    seed:                    int   = None,
    image_strength:          float = None,
    character_image_path:    str   = None,
    character_strength:      float = None,
    character_mode:          str   = None,
    character_name:          str   = None,
    character_description:   str   = None,
    pass1_sigmas:            str   = None,
    pass1_sampler:           str   = None,
    pass1_cfg:               float = None,
    pass2_sigmas:            str   = None,
    pass2_sampler:           str   = None,
    pass2_cfg:               float = None,
    pass2_seed:              int   = None,
    pro_mode:                bool  = None,
    pro_steps:               int   = None,
    pro_scheduler:           str   = None,
    pro_split_at:            int   = None,
    use_tiled_vae:           bool  = None,
    tiled_spatial_tiles:     int   = None,
    tiled_spatial_overlap:   int   = None,
    tiled_temporal_len:      int   = None,
    tiled_temporal_overlap:  int   = None,
    tiled_last_frame_fix:    bool  = None,
    lora_stack:              list  = None,
    lora_stack_json:         str   = None,
    output_prefix:           str   = None,
    bypass_easy_prompt:      bool  = None,
    llm_model:               str   = None,
    use_vision:              bool  = None,
    vision_model:            str   = None,
    unet_model:              str   = None,
    clip_name1:              str   = None,
    clip_name2:              str   = None,
) -> Optional[str]:
    """
    LTX-2 Studio PRO -- Two-pass generation pipeline with Character Consistency.

    +---------------------------------------------------------------------------+
    |  PHASE 0 - EASY PROMPT (before video model -- LLM/Vision then unload)    |
    |                                                                           |
    |  [LTX2VisionDescribe]  image  -----------------------> scene_context     |
    |  [LTX2PromptArchitect] user_input + scene_ctx -------> positive, neg     |
    +---------------------------------------------------------------------------+
    |  PHASE 1 - MODEL LOADING                                                 |
    |                                                                           |
    |  [197/UnetLoaderGGUF]     --> unet (raw)                                 |
    |  [190/DualCLIPLoader]     --> clip_model                                 |
    |  [263/LTX2MasterLoaderLD] --> unet + clip (LoRA stack)                   |
    |  [PathchSageAttentionKJ]  --> unet (sage attn patch, optional)           |
    |  [LTXVChunkFeedForward]   --> unet (chunk FF patch, optional)            |
    |  [184/VAELoader]          --> vae_video                                  |
    |  [196/VAELoaderKJ]        --> vae_audio                                  |
    |  [189/LatentUpscaleModel] --> upscale_model                              |
    +---------------------------------------------------------------------------+
    |  PHASE 2 - TEXT ENCODING                                                 |
    |                                                                           |
    |  [121/CLIPTextEncode]     positive --> cond_pos                          |
    |  [110/CLIPTextEncode]     negative --> cond_neg                          |
    |  [ConditioningZeroOut]    cond_pos  --> zero_out                         |
    |  [107/LTXVConditioning]   fps meta  --> cond[0]=pos, cond[1]=neg         |
    +---------------------------------------------------------------------------+
    |  PHASE 3 - CHARACTER ANCHOR (mode "anchor" or "both")                    |
    |                                                                           |
    |  [165/ImageResizeKJv2]    char_img  --> resized to W x H                 |
    |  [295/VAEEncode]          pixels    --> anchor_samples LATENT            |
    +---------------------------------------------------------------------------+
    |  PHASE 4 - LATENT PREPARATION                                            |
    |                                                                           |
    |  [246/ResizeImagesByLongerEdge]  image --> 1536px long-edge              |
    |  [165/ImageResizeKJv2]           image --> target W x H, lanczos         |
    |  [164/ResizeImageMaskNode]       image --> x0.5 half-res                 |
    |  [163/GetImageSize]              --> half_w, half_h                      |
    |  [108/EmptyLTXVLatentVideo]      --> vid_lat (half-res)                  |
    |  [162/LTXVPreprocess]            img_compression=33 --> pp_img           |
    |  [161/LTXVImgToVideoInplace]     I2V mode --> vid_lat (conditioned)      |
    |  [199/LTXVEmptyLatentAudio]      --> aud_lat                             |
    |  [109/LTXVConcatAVLatent]        vid_lat+aud_lat --> combined            |
    +---------------------------------------------------------------------------+
    |  PHASE 5 - SIGMA SCHEDULE                                                |
    |                                                                           |
    |  Standard: [ManualSigmas] pass1_sigmas --> sig_p1                        |
    |  PRO mode: [ModelSamplingSD3 shift=8] + [BasicScheduler] +              |
    |            [SplitSigmas step=pro_split_at] --> sig_high, sig_low         |
    +---------------------------------------------------------------------------+
    |  PHASE 6 - PASS 1 (first-pass denoising)                                |
    |                                                                           |
    |  [CFGGuider]  model + pos/neg --> guider_p1                              |
    |  [RandomNoise] seed --> noise_p1                                         |
    |  [SamplerCustomAdvanced] --> p1_av_output                                |
    +---------------------------------------------------------------------------+
    |  PHASE 7 - PASS 2 (spatial upscale + refinement)                        |
    |                                                                           |
    |  [LTXVSeparateAVLatent]  p1_av --> vid_lat_p1, aud_lat_p1               |
    |  [LTXVCropGuides]        pos/neg + lat --> cropped cond + lat            |
    |  [CFGGuider]  model + cropped --> guider_p2                              |
    |  [LTXVLatentUpsampler]   x2 upsample --> upsampled                       |
    |  [LTXVConcatAVLatent]    upsampled + aud --> av_lat2                     |
    |  [SamplerCustomAdvanced] --> p2_denoised                                 |
    +---------------------------------------------------------------------------+
    |  PHASE 8 - DECODE                                                        |
    |                                                                           |
    |  [LTXVSeparateAVLatent]                                                  |
    |  [265/LTXVSpatioTemporalTiledVAEDecode] or [VAEDecode] --> frames        |
    |  [201/LTXVAudioVAEDecode] --> audio                                      |
    +---------------------------------------------------------------------------+
    |  PHASE 9 - SAVE                                                          |
    |                                                                           |
    |  [319/VHS_VideoCombine] h264-mp4, crf=19, yuv420p (preferred)           |
    |  Fallback: [CreateVideo] --> save_video_from_components                  |
    |  save_metadata_sidecar() --> JSON sidecar                                |
    +---------------------------------------------------------------------------+

    Returns: output video path (str) or None on failure.
    """
    # Resolve all None defaults to globals
    if user_input           is None: user_input           = USER_INPUT
    if image_path           is None: image_path           = IMAGE_PATH
    if positive_prompt      is None: positive_prompt      = POSITIVE_PROMPT
    if negative_prompt      is None: negative_prompt      = NEGATIVE_PROMPT
    if width                is None: width                = WIDTH
    if height               is None: height               = HEIGHT
    if frames               is None: frames               = FRAMES
    if fps                  is None: fps                  = FPS
    if seed                 is None: seed                 = SEED
    if image_strength       is None: image_strength       = IMAGE_STRENGTH
    if character_image_path is None: character_image_path = CHARACTER_IMAGE_PATH
    if character_strength   is None: character_strength   = CHARACTER_STRENGTH
    if character_mode       is None: character_mode       = CHARACTER_CONSISTENCY_MODE
    if character_name       is None: character_name       = CHARACTER_NAME
    if character_description is None: character_description = CHARACTER_DESCRIPTION
    if pass1_sigmas         is None: pass1_sigmas         = PASS1_SIGMAS
    if pass1_sampler        is None: pass1_sampler        = PASS1_SAMPLER
    if pass1_cfg            is None: pass1_cfg            = PASS1_CFG
    if pass2_sigmas         is None: pass2_sigmas         = PASS2_SIGMAS
    if pass2_sampler        is None: pass2_sampler        = PASS2_SAMPLER
    if pass2_cfg            is None: pass2_cfg            = PASS2_CFG
    if pass2_seed           is None: pass2_seed           = PASS2_SEED
    if pro_mode             is None: pro_mode             = PRO_MODE
    if pro_steps            is None: pro_steps            = PRO_STEPS
    if pro_scheduler        is None: pro_scheduler        = PRO_SCHEDULER
    if pro_split_at         is None: pro_split_at         = PRO_SPLIT_AT
    if use_tiled_vae        is None: use_tiled_vae        = USE_TILED_VAE
    if tiled_spatial_tiles  is None: tiled_spatial_tiles  = TILED_SPATIAL_TILES
    if tiled_spatial_overlap is None: tiled_spatial_overlap = TILED_SPATIAL_OVERLAP
    if tiled_temporal_len   is None: tiled_temporal_len   = TILED_TEMPORAL_LEN
    if tiled_temporal_overlap is None: tiled_temporal_overlap = TILED_TEMPORAL_OVERLAP
    if tiled_last_frame_fix is None: tiled_last_frame_fix = TILED_LAST_FRAME_FIX
    if output_prefix        is None: output_prefix        = OUTPUT_PREFIX

    _lora_stack = lora_stack if lora_stack is not None else LORA_STACK
    _lora_json  = lora_stack_json if lora_stack_json is not None else LORA_STACK_JSON
    _char_mode  = character_mode.lower().strip()
    _bypass     = bypass_easy_prompt if bypass_easy_prompt is not None else BYPASS_EASY_PROMPT
    _llm_model  = llm_model  if llm_model  is not None else LLM_MODEL
    _use_vision = use_vision if use_vision is not None else USE_VISION
    _vis_model  = vision_model if vision_model is not None else VISION_MODEL
    _unet       = unet_model  if unet_model  is not None else UNET_MODEL
    _clip1      = clip_name1  if clip_name1  is not None else CLIP_NAME1
    _clip2      = clip_name2  if clip_name2  is not None else CLIP_NAME2

    t0 = time.time()
    import_custom_nodes()
    clear_output()

    print("\U0001f3ac LTX-2 Studio PRO -- Generation Starting")
    print(f"   Resolution   : {width}x{height}  |  Frames: {frames}  |  Seed: {seed}")
    print(f"   Mode         : {'I2V' if image_path else 'T2V'}"
          f"  |  Character: {_char_mode}  |  Pro: {pro_mode}")
    print(f"   Easy Prompt  : {'BYPASS' if _bypass else f'LLM={_llm_model}'}")
    _print_vram()

    print("\n\U0001f50d Model file check...")
    all_ok = validate_model_files({
        "unet"    : _unet,
        "clip1"   : _clip1,
        "clip2"   : _clip2,
        "vae_vid" : VAE_VIDEO_MODEL,
        "vae_aud" : VAE_AUDIO_MODEL,
        "upscaler": UPSCALER_MODEL,
    })
    if not all_ok:
        raise FileNotFoundError(
            "One or more model files are missing -- run Cell 2 first.")

    # ====================================================================
    # PHASE 0 -- EASY PROMPT
    # ====================================================================
    seed_image_tensor = None
    if image_path:
        seed_image_tensor = load_image_tensor(image_path)
        if seed_image_tensor is None:
            print(f"   \u26a0\ufe0f  Image not found: {image_path} -- switching to T2V")
        else:
            print(f"   \u2713 Reference image loaded: {image_path}  {seed_image_tensor.shape}")

    char_image_tensor = None
    if character_image_path and _char_mode != "none":
        char_image_tensor = load_image_tensor(character_image_path)
        if char_image_tensor is None:
            print(f"   \u26a0\ufe0f  Character image not found: {character_image_path} -- skipping anchor.")
        else:
            print(f"   \u2713 Character image loaded: {character_image_path}  {char_image_tensor.shape}")

    analysis_tensor = char_image_tensor if char_image_tensor is not None else seed_image_tensor
    scene_context = character_description or ""

    if analysis_tensor is not None and _use_vision and not _bypass:
        print("\n\U0001f441\ufe0f  Vision Describe...")
        scene_context = run_vision_describe(
            analysis_tensor, character_description,
            use_vision_override=_use_vision, vision_model_override=_vis_model)
        if scene_context:
            print(f"   Scene context: {scene_context[:120]}...")

    final_positive = positive_prompt
    final_negative = negative_prompt
    if not _bypass and user_input.strip():
        print("\n\U0001f9e0 Easy Prompt expansion...")
        final_positive, final_negative = run_easy_prompt(
            user_input=user_input, frame_count=frames, seed=seed,
            scene_context=scene_context, llm_model_override=_llm_model)
        print(f"\n   -- EXPANDED PROMPT -----")
        print(f"   {final_positive[:300]}{'...' if len(final_positive) > 300 else ''}")
        print(f"\n   -- NEGATIVE PROMPT -----")
        print(f"   {final_negative[:150]}...")
    else:
        print("   [EasyPrompt] Bypassed -- using manual POSITIVE_PROMPT.")

    cleanup_memory(verbose=True)

    # ====================================================================
    # PHASE 1 -- MODEL LOADING
    # ====================================================================
    with torch.inference_mode():

        # [197] UnetLoaderGGUF
        print("\n\U0001f4e6 Loading UNet (GGUF Q4_K_M distilled)...")
        try:
            unet_loader = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
            unet        = get_value_at_index(unet_loader.load_unet(unet_name=_unet), 0)
        except KeyError:
            raise RuntimeError(
                "UnetLoaderGGUF not found.\n"
                "  Fix: Run Cell 1 to clone ComfyUI_GGUF custom node.")

        # [190] DualCLIPLoader
        print("   Loading CLIP encoders (DualCLIPLoader)...")
        try:
            clip_loader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
            clip_model  = get_value_at_index(
                clip_loader.load_clip(clip_name1=_clip1, clip_name2=_clip2,
                                      type="ltxv", device="default"), 0)
        except Exception as e:
            print(f"   \u26a0\ufe0f  fp4 CLIP failed ({type(e).__name__}: {e})")
            print("      Trying fp8 fallback...")
            fp8 = "gemma_3_12B_it_fp8_scaled.safetensors"
            try:
                clip_model = get_value_at_index(
                    clip_loader.load_clip(clip_name1=fp8, clip_name2=_clip2,
                                          type="ltxv", device="default"), 0)
                print("   \u2713 fp8 CLIP loaded.")
            except Exception as e2:
                raise RuntimeError(f"DualCLIPLoader failed: {e2}")

        # [263] LTX2MasterLoaderLD -- LoRA stack
        print("   Applying LoRA stack (LTX2MasterLoaderLD)...")
        unet, clip_model = apply_lora_stack(unet, clip_model, _lora_stack, _lora_json)

        unet = apply_sage_attention(unet)
        unet = apply_chunk_ff(unet)
        purge_vram("after unet+lora")
        _print_vram()

        # [184] VAELoader -- video VAE
        print("   Loading VAEs...")
        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        vae_video = get_value_at_index(vaeloader.load_vae(vae_name=VAE_VIDEO_MODEL), 0)

        # [196] VAELoaderKJ -- audio VAE
        try:
            vae_audio = get_value_at_index(_load_audio_vae(VAE_AUDIO_MODEL), 0)
        except Exception as e:
            raise RuntimeError(f"Audio VAE load failed: {e}")

        # [189] LatentUpscaleModelLoader
        try:
            uml = NODE_CLASS_MAPPINGS["LatentUpscaleModelLoader"]()
            if hasattr(uml, "EXECUTE_NORMALIZED"):
                upscale_model = get_value_at_index(uml.EXECUTE_NORMALIZED(model_name=UPSCALER_MODEL), 0)
            elif hasattr(uml, "load_model"):
                upscale_model = get_value_at_index(uml.load_model(model_name=UPSCALER_MODEL), 0)
            else:
                raise AttributeError("LatentUpscaleModelLoader: no load method found")
        except Exception as e:
            raise RuntimeError(f"LatentUpscaleModelLoader failed: {e}")

        # ====================================================================
        # PHASE 2 -- TEXT ENCODING
        # ====================================================================
        print("\n\U0001f4dd Encoding prompts...")
        try:
            # [121] CLIPTextEncode positive
            cte      = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
            cond_pos = cte.encode(text=final_positive, clip=clip_model)
            cond_neg = cte.encode(text=final_negative, clip=clip_model)

            # ConditioningZeroOut
            zero_out  = NODE_CLASS_MAPPINGS["ConditioningZeroOut"]()
            cond_zero = zero_out.zero_out(conditioning=get_value_at_index(cond_pos, 0))

            # [107] LTXVConditioning
            ltxv_cond = NODE_CLASS_MAPPINGS["LTXVConditioning"]()
            cond = ltxv_cond.EXECUTE_NORMALIZED(
                frame_rate=float(fps),
                positive=get_value_at_index(cond_pos, 0),
                negative=get_value_at_index(cond_zero, 0))
        except Exception as e:
            raise RuntimeError(f"Text encoding failed: {e}")

        del clip_model
        cleanup_memory()

        # ====================================================================
        # PHASE 4 -- LATENT PREPARATION (Phase 3 anchor deferred here)
        # ====================================================================
        print("\n\U0001f5c2\ufe0f  Preparing latents...")
        _print_vram()

        _ref_tensor = seed_image_tensor
        if _ref_tensor is None and _char_mode in ("i2v", "both"):
            _ref_tensor = char_image_tensor
        _use_i2v = (_ref_tensor is not None and _char_mode in ("i2v", "both")) or \
                   (_ref_tensor is not None and image_path is not None and _char_mode == "none")

        # [256] EmptyImage -> [164] ResizeImageMaskNode x0.5 -> [163] GetImageSize
        ei       = NODE_CLASS_MAPPINGS["EmptyImage"]()
        full_img = ei.generate(width=width, height=height, batch_size=1, color=0)
        rimn     = NODE_CLASS_MAPPINGS["ResizeImageMaskNode"]()
        half_img = rimn.EXECUTE_NORMALIZED(
            input=get_value_at_index(full_img, 0), scale_method="area",
            resize_type={"resize_type": "scale by multiplier", "multiplier": 0.5})
        gis     = NODE_CLASS_MAPPINGS["GetImageSize"]()
        half_sz = gis.EXECUTE_NORMALIZED(image=get_value_at_index(half_img, 0))
        half_w  = get_value_at_index(half_sz, 0)
        half_h  = get_value_at_index(half_sz, 1)
        print(f"   Latent dims : {half_w}x{half_h}  (half of {width}x{height})")

        # PHASE 3 -- Character anchor (deferred for half-res dims)
        # [295] VAEEncode
        anchor_latent = None
        if char_image_tensor is not None and _char_mode in ("anchor", "both"):
            print("\n\U0001f9ec Character Anchor -- encoding character image as latent...")
            try:
                ikj = NODE_CLASS_MAPPINGS["ImageResizeKJv2"]()
                char_resized = get_value_at_index(
                    ikj.resize(image=char_image_tensor, width=half_w, height=half_h,
                               upscale_method="lanczos", keep_proportion="crop",
                               pad_color="0, 0, 0", crop_position="center",
                               divisible_by=32, device="cpu"), 0)
                vae_enc       = NODE_CLASS_MAPPINGS["VAEEncode"]()
                anchor_latent = get_value_at_index(
                    vae_enc.encode(pixels=char_resized, vae=vae_video), 0)
                print(f"   \u2713 Character anchor encoded at {half_w}x{half_h}")
            except Exception as e:
                print(f"   \u26a0\ufe0f  Character anchor failed ({e}) -- continuing without anchor.")
                anchor_latent = None

        # [108] EmptyLTXVLatentVideo
        eltxv   = NODE_CLASS_MAPPINGS["EmptyLTXVLatentVideo"]()
        vid_lat = eltxv.EXECUTE_NORMALIZED(width=half_w, height=half_h, length=frames, batch_size=1)

        if _use_i2v and _ref_tensor is not None:
            try:
                if "ResizeImagesByLongerEdge" in NODE_CLASS_MAPPINGS:
                    rle = NODE_CLASS_MAPPINGS["ResizeImagesByLongerEdge"]()
                    _ref_tensor = get_value_at_index(rle.resize(images=_ref_tensor, longer_edge=1536), 0)
                if "ImageResizeKJv2" in NODE_CLASS_MAPPINGS:
                    ikj2 = NODE_CLASS_MAPPINGS["ImageResizeKJv2"]()
                    _ref_tensor = get_value_at_index(
                        ikj2.resize(image=_ref_tensor, width=half_w * 2, height=half_h * 2,
                                    upscale_method="lanczos", keep_proportion="crop",
                                    pad_color="0, 0, 0", crop_position="center",
                                    divisible_by=2, device="cpu"), 0)
                else:
                    rim2 = NODE_CLASS_MAPPINGS["ResizeImageMaskNode"]()
                    _orig_h, _orig_w = _ref_tensor.shape[1], _ref_tensor.shape[2]
                    _scale = max((half_w * 2) / max(_orig_w, 1), (half_h * 2) / max(_orig_h, 1))
                    _ref_tensor = get_value_at_index(
                        rim2.EXECUTE_NORMALIZED(input=_ref_tensor, scale_method="lanczos",
                            resize_type={"resize_type": "scale by multiplier", "multiplier": _scale}), 0)

                # [162] LTXVPreprocess
                pp_node = NODE_CLASS_MAPPINGS["LTXVPreprocess"]()
                pp_img  = get_value_at_index(
                    pp_node.EXECUTE_NORMALIZED(img_compression=33, image=_ref_tensor), 0)

                # [161] LTXVImgToVideoInplace
                _i2v_strength = character_strength if _char_mode in ("i2v", "both") else image_strength
                i2v     = NODE_CLASS_MAPPINGS["LTXVImgToVideoInplace"]()
                vid_lat = i2v.EXECUTE_NORMALIZED(
                    strength=_i2v_strength, bypass=False,
                    vae=vae_video, image=pp_img,
                    latent=get_value_at_index(vid_lat, 0))
                print(f"   \u2713 I2V conditioning applied  (strength={_i2v_strength})")
            except KeyError as e:
                print(f"   \u26a0\ufe0f  I2V node missing ({e}) -- using empty latent (T2V mode).")
                vid_lat = (get_value_at_index(vid_lat, 0),)
            except Exception as e:
                print(f"   \u26a0\ufe0f  I2V conditioning failed ({e}) -- using empty latent.")
                vid_lat = (get_value_at_index(vid_lat, 0),)
        else:
            vid_lat = (get_value_at_index(vid_lat, 0),)

        _vid_lat_input = get_value_at_index(vid_lat, 0)
        if anchor_latent is not None:
            try:
                _anch_shape = anchor_latent.get("samples", torch.empty(0)).shape
                if len(_anch_shape) == 4:
                    _s = anchor_latent["samples"].unsqueeze(2)
                    _vid_lat_input = {**anchor_latent, "samples": _s}
                else:
                    _vid_lat_input = anchor_latent
                print(f"   \u2713 Character anchor injected as video latent seed  (mode={_char_mode})")
            except Exception as e:
                print(f"   \u26a0\ufe0f  Anchor injection error ({e}) -- using empty/I2V latent.")
                _vid_lat_input = get_value_at_index(vid_lat, 0)

        # [199] LTXVEmptyLatentAudio
        elalat  = NODE_CLASS_MAPPINGS["LTXVEmptyLatentAudio"]()
        aud_lat = elalat.EXECUTE_NORMALIZED(
            frames_number=frames, frame_rate=fps, batch_size=1, audio_vae=vae_audio)

        # [109] LTXVConcatAVLatent
        catav          = NODE_CLASS_MAPPINGS["LTXVConcatAVLatent"]()
        av_lat1        = catav.EXECUTE_NORMALIZED(
            video_latent=_vid_lat_input,
            audio_latent=get_value_at_index(aud_lat, 0))
        combined_latent = get_value_at_index(av_lat1, 0)

        # ====================================================================
        # PHASE 5 -- SIGMA SCHEDULE
        # ====================================================================
        manualsigmas   = NODE_CLASS_MAPPINGS["ManualSigmas"]()
        ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
        randomnoise    = NODE_CLASS_MAPPINGS["RandomNoise"]()
        cfgguider      = NODE_CLASS_MAPPINGS["CFGGuider"]()
        sca            = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()

        sig_p1_high = None
        sig_p2_low  = None

        if pro_mode:
            print(f"\n\u2699\ufe0f  PRO sigma schedule -- BasicScheduler steps={pro_steps} "
                  f"sched={pro_scheduler} split@{pro_split_at}")
            try:
                ms3  = NODE_CLASS_MAPPINGS["ModelSamplingSD3"]()
                unet_sampled = get_value_at_index(ms3.patch(model=unet, shift=8.0), 0)
                bs   = NODE_CLASS_MAPPINGS["BasicScheduler"]()
                sigs = get_value_at_index(
                    bs.get_sigmas(model=unet_sampled, scheduler=pro_scheduler,
                                  steps=pro_steps, denoise=1.0), 0)
                ss         = NODE_CLASS_MAPPINGS["SplitSigmas"]()
                split_out  = ss.get_sigmas(sigmas=sigs, step=pro_split_at)
                sig_p1_high = get_value_at_index(split_out, 0)
                sig_p2_low  = get_value_at_index(split_out, 1)
                unet        = unet_sampled
                sampler_p1  = ksamplerselect.EXECUTE_NORMALIZED(sampler_name="euler")
                sampler_p2  = sampler_p1
                print("   \u2713 PRO sigmas computed (ModelSamplingSD3 + BasicScheduler + SplitSigmas)")
            except KeyError as e:
                print(f"   \u26a0\ufe0f  PRO mode node missing: {e} -- falling back to ManualSigmas.")
                pro_mode = False
            except Exception as e:
                print(f"   \u26a0\ufe0f  PRO schedule failed ({e}) -- falling back to ManualSigmas.")
                pro_mode = False

        if not pro_mode:
            print(f"\n\u2699\ufe0f  Standard sigma schedule -- Pass1: {pass1_sigmas[:45]}...")
            sig_p1_high = get_value_at_index(manualsigmas.EXECUTE_NORMALIZED(sigmas=pass1_sigmas), 0)
            sampler_p1  = ksamplerselect.EXECUTE_NORMALIZED(sampler_name=pass1_sampler)
            sig_p2_low  = get_value_at_index(manualsigmas.EXECUTE_NORMALIZED(sigmas=pass2_sigmas), 0)
            sampler_p2  = ksamplerselect.EXECUTE_NORMALIZED(sampler_name=pass2_sampler)

        # ====================================================================
        # PHASE 6 -- PASS 1
        # ====================================================================
        print(f"\n\U0001f680 Pass 1 -- denoising...")
        _print_vram()

        noise_p1  = randomnoise.EXECUTE_NORMALIZED(noise_seed=seed)
        guider_p1 = cfgguider.EXECUTE_NORMALIZED(
            cfg=pass1_cfg, model=unet,
            positive=get_value_at_index(cond, 0),
            negative=get_value_at_index(cond, 1))
        try:
            out1 = sca.EXECUTE_NORMALIZED(
                noise=get_value_at_index(noise_p1, 0),
                guider=get_value_at_index(guider_p1, 0),
                sampler=get_value_at_index(sampler_p1, 0),
                sigmas=sig_p1_high,
                latent_image=combined_latent)
            p1_av = get_value_at_index(out1, 0)
        except Exception as e:
            raise RuntimeError(f"Pass 1 sampling failed: {e}")

        del guider_p1
        cleanup_memory()
        print("   \u2713 Pass 1 complete")

        # ====================================================================
        # PHASE 7 -- PASS 2
        # ====================================================================
        print(f"\n\U0001f527 Pass 2 -- upscale + refinement...")
        _print_vram()

        # [LTXVSeparateAVLatent]
        ltxvsep    = NODE_CLASS_MAPPINGS["LTXVSeparateAVLatent"]()
        s1         = ltxvsep.EXECUTE_NORMALIZED(av_latent=p1_av)
        vid_lat_p1 = get_value_at_index(s1, 0)
        aud_lat_p1 = get_value_at_index(s1, 1)

        # [LTXVCropGuides]
        ltxvcrop = NODE_CLASS_MAPPINGS["LTXVCropGuides"]()
        cropped  = ltxvcrop.EXECUTE_NORMALIZED(
            positive=get_value_at_index(cond, 0),
            negative=get_value_at_index(cond, 1),
            latent=vid_lat_p1)

        guider_p2 = cfgguider.EXECUTE_NORMALIZED(
            cfg=pass2_cfg, model=unet,
            positive=get_value_at_index(cropped, 0),
            negative=get_value_at_index(cropped, 1))

        # [LTXVLatentUpsampler] -- 2x spatial upsample
        ltxvup    = NODE_CLASS_MAPPINGS["LTXVLatentUpsampler"]()
        upsampled = ltxvup.upsample_latent(
            samples=get_value_at_index(cropped, 2),
            upscale_model=upscale_model, vae=vae_video)
        del upscale_model
        cleanup_memory()

        # [LTXVConcatAVLatent]
        av_lat2 = catav.EXECUTE_NORMALIZED(
            video_latent=get_value_at_index(upsampled, 0),
            audio_latent=aud_lat_p1)

        noise_p2 = randomnoise.EXECUTE_NORMALIZED(noise_seed=pass2_seed)
        try:
            out2 = sca.EXECUTE_NORMALIZED(
                noise=get_value_at_index(noise_p2, 0),
                guider=get_value_at_index(guider_p2, 0),
                sampler=get_value_at_index(sampler_p2, 0),
                sigmas=sig_p2_low,
                latent_image=get_value_at_index(av_lat2, 0))
            p2_denoised = get_value_at_index(out2, 1)
        except Exception as e:
            raise RuntimeError(f"Pass 2 sampling failed: {e}")

        del guider_p2, unet
        cleanup_memory()
        print("   \u2713 Pass 2 complete")

        # ====================================================================
        # PHASE 8 -- DECODE
        # ====================================================================
        print("\n\U0001f39e\ufe0f  Decoding video & audio...")
        _print_vram()

        s2          = ltxvsep.EXECUTE_NORMALIZED(av_latent=p2_denoised)
        vid_lat_fin = get_value_at_index(s2, 0)
        aud_lat_fin = get_value_at_index(s2, 1)

        decoded_frames = None
        if use_tiled_vae:
            # [265] LTXVSpatioTemporalTiledVAEDecode
            try:
                tiled_dec = NODE_CLASS_MAPPINGS["LTXVSpatioTemporalTiledVAEDecode"]()
                decoded_frames = get_value_at_index(
                    tiled_dec.EXECUTE_NORMALIZED(
                        vae=vae_video, latents=vid_lat_fin,
                        spatial_tiles=tiled_spatial_tiles,
                        spatial_overlap=tiled_spatial_overlap,
                        temporal_tile_length=tiled_temporal_len,
                        temporal_overlap=tiled_temporal_overlap,
                        last_frame_fix=tiled_last_frame_fix,
                        working_device="auto", working_dtype="auto"), 0)
                print("   \u2713 Tiled VAE decode (LTXVSpatioTemporalTiledVAEDecode)")
            except (KeyError, Exception) as e:
                print(f"   \u26a0\ufe0f  Tiled VAE unavailable ({type(e).__name__}: {e})")
                print("      Falling back to standard VAEDecode.")
                use_tiled_vae = False

        if not use_tiled_vae or decoded_frames is None:
            vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
            decoded_frames = get_value_at_index(
                vaedecode.decode(samples=vid_lat_fin, vae=vae_video), 0)
            print("   \u2713 Standard VAE decode (VAEDecode)")

        del vae_video
        cleanup_memory()

        # [201] LTXVAudioVAEDecode
        try:
            aud_dec   = NODE_CLASS_MAPPINGS["LTXVAudioVAEDecode"]()
            audio_out = aud_dec.EXECUTE_NORMALIZED(samples=aud_lat_fin, audio_vae=vae_audio)
        except Exception as e:
            print(f"   \u26a0\ufe0f  Audio decode failed ({e}) -- proceeding without audio.")
            audio_out = None

        del vae_audio
        cleanup_memory()

        # ====================================================================
        # PHASE 9 -- SAVE
        # ====================================================================
        print("\n\U0001f4be Saving video...")
        _print_vram()
        output_path = None

        _prefix = output_prefix
        if character_name and character_name != "Character":
            _prefix = f"{output_prefix}-{character_name}"

        # [319/VHS_VideoCombine] preferred -- h264-mp4, crf=19, yuv420p
        if "VHS_VideoCombine" in NODE_CLASS_MAPPINGS and audio_out is not None:
            try:
                vhs  = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()
                _audio_data = get_value_at_index(audio_out, 0)
                vhs_out = vhs.combine_video(
                    images=decoded_frames, frame_rate=fps, loop_count=0,
                    filename_prefix=_prefix, format="video/h264-mp4",
                    pix_fmt="yuv420p", crf=19, save_metadata=True,
                    trim_to_audio=False, pingpong=False, save_output=True,
                    audio=_audio_data)
                _fnames = get_value_at_index(vhs_out, 0)
                if hasattr(_fnames, "video_paths") and _fnames.video_paths:
                    output_path = _fnames.video_paths[0]
                elif hasattr(_fnames, "video_path"):
                    output_path = _fnames.video_path
                elif isinstance(_fnames, (list, tuple)) and len(_fnames) > 0:
                    output_path = _fnames[0]
                else:
                    output_path = None
                if output_path:
                    print("   \u2713 Saved via VHS_VideoCombine (h264-mp4, crf=19, yuv420p)")
            except Exception as e:
                print(f"   \u26a0\ufe0f  VHS_VideoCombine failed ({e}) -- using CreateVideo fallback.")
                output_path = None

        # Fallback: CreateVideo
        if output_path is None:
            try:
                createvideo = NODE_CLASS_MAPPINGS["CreateVideo"]()
                _aud_arg    = get_value_at_index(audio_out, 0) if audio_out else None
                if _aud_arg is not None:
                    vid_obj = createvideo.EXECUTE_NORMALIZED(fps=fps, images=decoded_frames, audio=_aud_arg)
                else:
                    vid_obj = createvideo.EXECUTE_NORMALIZED(fps=fps, images=decoded_frames)
                output_path = save_video_from_components(get_value_at_index(vid_obj, 0), prefix=_prefix)
                print("   \u2713 Saved via CreateVideo fallback")
            except Exception as e:
                raise RuntimeError(f"Video save failed: {e}")

    # Timing
    elapsed = time.time() - t0
    mins, secs = divmod(int(elapsed), 60)

    _active_loras = [s["lora"] for s in _lora_stack if s.get("on")]
    meta = {
        "seed": seed, "width": width, "height": height, "frames": frames, "fps": fps,
        "positive_prompt": final_positive, "negative_prompt": final_negative,
        "user_input": user_input, "image_path": image_path,
        "character_image": character_image_path, "character_mode": _char_mode,
        "character_name": character_name, "character_strength": character_strength,
        "pro_mode": pro_mode, "loras": _active_loras,
        "unet_model": UNET_MODEL, "elapsed_seconds": elapsed, "output_path": output_path,
    }
    if output_path:
        save_metadata_sidecar(output_path, meta)

    print(f"\n\u2705 Done in {mins}m {secs}s")
    print(f"   \U0001f4c1 {output_path}")
    _print_vram()

    if SHOW_PREVIEWS and output_path:
        print("\n\u25b6 Preview:")
        display_video(output_path)

    if DOWNLOAD_AFTER_GENERATE and output_path:
        print("   \u2b07\ufe0f  Auto-downloading...")
        try:
            files.download(output_path)
        except Exception as e:
            print(f"   \u26a0\ufe0f  Download failed ({e}) -- file is at {output_path}")

    return output_path


print("\u2705 generate_pro() defined.")
print("   Signature: generate_pro(user_input, image_path, width, height, frames, ...)")


# ======================================================================
# CELL 12  --  InfiniteFlowEngine
# ======================================================================

# @title  { "single-column": true }
# @markdown ## 💥 12. Infinite Flow Engine (Multi-Scene Orchestrator)

class InfiniteFlowEngine:
    """
    Generates a sequence of clips where:
      1. VisionDescribeEngine analyses the current seed/last frame
      2. EasyPromptEngine expands story beats into cinematic prompts,
         injecting CharacterBible as a hard consistency lock
      3. generate_pro() renders each clip with the two-pass pipeline
      4. get_last_frame_tensor() extracts the last frame for chaining
    """

    def __init__(
        self,
        character_bible:   Optional["CharacterBible"] = None,
        llm_model:         str   = "8B",
        vision_model:      str   = "3B-fast",
        width:             int   = 768,
        height:            int   = 512,
        frames:            int   = 121,
        fps:               int   = 25,
        image_strength:    float = 1.0,
        creativity:        float = 0.9,
        lora_triggers:     str   = "",
        use_vision:        bool  = True,
        use_tiled_vae:     bool  = True,
        offline:           bool  = False,
        output_dir:        str   = "/content/ComfyUI/output/infinite_flow",
        show_previews:     bool  = True,
    ):
        self.bible          = character_bible or CharacterBible()
        self.llm_model      = llm_model
        self.vision_model   = vision_model
        self.width          = width
        self.height         = height
        self.frames         = frames
        self.fps            = fps
        self.img_strength   = image_strength
        self.creativity     = creativity
        self.lora_triggers  = lora_triggers
        self.use_vision     = use_vision
        self.use_tiled_vae  = use_tiled_vae
        self.offline        = offline
        self.output_dir     = output_dir
        self.show_previews  = show_previews
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        self._vision_engine  = VisionDescribeEngine(vision_model, offline)
        self._prompt_engine  = EasyPromptEngine(llm_model, offline, keep_loaded=False)
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
            _vram_free()
        else:
            scene_ctx = self._rolling_ctx()

        print(f"   [Beat {beat_idx}] EasyPrompt expand...")
        prompt, neg = self._prompt_engine.generate(
            user_input=beat, frame_count=self.frames,
            creativity=self.creativity, seed=beat_seed,
            scene_context=scene_ctx, lora_triggers=self.lora_triggers,
            character_bible=self.bible.to_prompt_block())
        _vram_free()

        self._scene_history.append(prompt[:400])
        if len(self._scene_history) > 2:
            self._scene_history = self._scene_history[-2:]

        return prompt, neg, beat_seed

    def run(self, story_beats: List[str], seed_image_path: Optional[str] = None,
            base_seed: int = 42) -> List[str]:
        """Process each beat in story_beats. Returns list of output .mp4 paths."""
        current_image: Optional[torch.Tensor] = None

        if seed_image_path and os.path.exists(seed_image_path):
            current_image = load_image_tensor(seed_image_path)
            print(f"[IFE] Seed image loaded: {seed_image_path}")
            if not self.bible.has_characters() and self.use_vision:
                print("[IFE] Extracting Character Bible from seed image...")
                desc = self._vision_engine.describe(current_image)
                _vram_free()
                self.bible.extract_from_description("Main Character", desc)
                self._scene_history.append(desc)
                print(f"[IFE] Bible set:\n{desc[:200]}...\n")

        print(f"\n[IFE] === INFINITE FLOW ENGINE ===")
        print(f"[IFE]  Beats    : {len(story_beats)}")
        print(f"[IFE]  Size     : {self.width}x{self.height}  {self.frames}f @ {self.fps}fps")
        print(f"[IFE]  LLM      : {self.llm_model}  Vision: {self.vision_model}")
        print(f"[IFE]  Bible    : {self.bible.names() or 'empty (T2V)'}")
        print(f"[IFE]  Mode     : {'I2V' if current_image is not None else 'T2V'}\n")

        for beat_idx, beat in enumerate(story_beats, 1):
            print(f"\n[IFE] -- Beat {beat_idx}/{len(story_beats)} ----")
            print(f"[IFE]   {beat[:80]}{'...' if len(beat)>80 else ''}")

            prompt, neg, beat_seed = self._process_beat(
                beat, current_image, beat_idx, base_seed)
            print(f"\n  EXPANDED ({len(prompt.split())}w): {prompt[:200]}...")
            print(f"  NEG:  {neg[:100]}...\n")

            # Save current_image to tmp file for generate_pro() (requires path, not tensor)
            _anchor_path = None
            if current_image is not None:
                _anchor_path = "/tmp/ife_anchor.jpg"
                try:
                    pil_frame = tensor_to_pil(current_image)
                    pil_frame.save(_anchor_path, "JPEG", quality=95)
                except Exception as e:
                    print(f"   [IFE] Could not save anchor frame: {e}")
                    _anchor_path = None

            try:
                clip_path = generate_pro(
                    user_input           = beat,
                    image_path           = _anchor_path,
                    positive_prompt      = prompt,
                    negative_prompt      = neg,
                    width                = self.width,
                    height               = self.height,
                    frames               = self.frames,
                    fps                  = self.fps,
                    seed                 = beat_seed,
                    image_strength       = self.img_strength,
                    use_tiled_vae        = self.use_tiled_vae,
                    output_prefix        = f"IFE_{beat_idx:03d}",
                    bypass_easy_prompt   = True,
                )
            except torch.cuda.OutOfMemoryError:
                _vram_free()
                print(f"   [IFE] OOM on beat {beat_idx}. Retrying T2V with more tiles...")
                clip_path = generate_pro(
                    user_input           = beat,
                    image_path           = None,  # T2V mode — drop anchor to save VRAM
                    positive_prompt      = prompt,
                    negative_prompt      = neg,
                    width                = self.width,
                    height               = self.height,
                    frames               = self.frames,
                    fps                  = self.fps,
                    seed                 = beat_seed + 1,
                    use_tiled_vae        = True,   # Keep tiled VAE for memory efficiency
                    tiled_spatial_tiles  = 4,       # More tiles = less VRAM per tile
                    tiled_spatial_overlap= 4,
                    output_prefix        = f"IFE_{beat_idx:03d}_retry",
                    bypass_easy_prompt   = True,
                )

            dest = os.path.join(self.output_dir, f"scene_{beat_idx:03d}.mp4")
            if clip_path:
                shutil.copy2(clip_path, dest)
            self._clip_paths.append(dest)
            print(f"\n[IFE] \u2705  Beat {beat_idx} -> {dest}")

            last_frame = get_last_frame_tensor(dest)
            current_image = last_frame if last_frame is not None else None
            if last_frame is not None:
                print(f"[IFE]    Last frame extracted for beat {beat_idx+1}.")
            else:
                print(f"[IFE]    \u26a0  No last frame -- next beat will be T2V.")

            if self.show_previews:
                print(f"\n  \u25b6 Scene {beat_idx}:")
                display_video(dest)

        print(f"\n[IFE] === {len(story_beats)} scenes complete ===")
        print(f"[IFE]  Output dir: {self.output_dir}")
        return self._clip_paths

    def concat_all(self, output_name: str = "full_video.mp4") -> str:
        out = os.path.join(self.output_dir, output_name)
        return concatenate_clips(self._clip_paths, out)

    def download_all(self):
        for p in self._clip_paths:
            if os.path.exists(p): files.download(p)

    def save_bible(self, path: str = "/content/character_bible.json"):
        self.bible.save(path)


print("\u2705 InfiniteFlowEngine defined.")


# ======================================================================
# CELL 13  --  Multi-Scene JSON Runner
# ======================================================================

# @title  { "single-column": true }
# @markdown ## 💥 13. Multi-Scene JSON Runner

SCENE_JSON = {
  "scene_id": "scene_02_whispering_woods_pro",
  "project_name": "Whispering_Cave_Part_2_PRO",
  "duration_seconds": 48,
  "video_style": "3D Pixar cartoon style, cinematic animation, consistent character design, professional quality",
  "environment": {
    "location": "Deep within the Greenleaf forest, ancient mossy trees, glowing blue flora",
    "time": "Dappled afternoon light filtering through thick canopy",
    "weather": "Swirling leaves, misty air with floating pollen motes",
    "mood": "Mysterious, magical, slightly spooky",
    "lighting": "Volumetric god rays, atmospheric haze, soft shadows",
    "color_palette": "Emerald greens, deep blues, warm amber highlights"
  },
  "main_characters": [
    {
      "name": "Shiv",
      "desc": "12-year-old Indian boy with consistent appearance",
      "detailed_appearance": {
        "face": "Round friendly face, large expressive brown eyes, small nose, warm smile",
        "hair": "Messy jet-black hair with natural volume, slight widows peak",
        "clothing": "Bright yellow cotton t-shirt with orange trim, blue denim shorts, red sneakers with white laces",
        "build": "Slim athletic build, average height for age",
        "skin_tone": "Warm medium brown skin tone, healthy glow",
        "accessories": "Holding an old weathered treasure map with both hands"
      },
      "lora_path": None,
      "personality_traits": "Curious, slightly nervous, excited about adventure",
      "voice_characteristics": "Young boy voice, Hindi speaker, slightly trembling when scared"
    },
    {
      "name": "Vandana",
      "desc": "12-year-old Indian girl with consistent appearance",
      "detailed_appearance": {
        "face": "Heart-shaped face, determined eyes, defined eyebrows, confident expression",
        "hair": "Long black hair in high ponytail with red scrunchie, slight bangs",
        "clothing": "Denim dungarees over white t-shirt, pink backpack, brown hiking boots",
        "build": "Athletic build, slightly taller than Shiv",
        "skin_tone": "Medium brown skin tone with golden undertones",
        "accessories": "Heavy brass flashlight in right hand, compass on belt"
      },
      "lora_path": None,
      "personality_traits": "Brave, protective, natural leader",
      "voice_characteristics": "Confident girl voice, Hindi speaker, calm and reassuring"
    }
  ],
  "story_action": {
    "shots": [
      {"time": "0-4s",   "camera": "Wide tracking shot, dolly forward",           "camera_movement": "dolly_forward",       "motion_intensity": 0.6, "action": "Shiv and Vandana walk deeper into the forest. Sunlight fades into emerald green glow.", "character_focus": "both",        "emotion": "curious_cautious",    "visual_effects": "Light transition, lens flare, atmospheric particles"},
      {"time": "4-8s",   "camera": "Close-up tracking, low angle focused on feet", "camera_movement": "tilt_up_slight",      "motion_intensity": 0.4, "action": "Close-up of Shivs red sneakers crunching over ancient leaves and glowing blue moss.",    "character_focus": "Shiv_feet",   "emotion": "wonder",              "visual_effects": "Glowing moss reaction, dust particles"},
      {"time": "8-12s",  "camera": "Low-angle upward tilt, slow dramatic pan",     "camera_movement": "tilt_up_dramatic",    "motion_intensity": 0.3, "action": "Camera tilts up from ground to reveal massive ancient trees. Vandana looks up in awe.",   "character_focus": "Vandana",     "emotion": "awe_mixed_fear",      "visual_effects": "Vertical emphasis, dramatic shadows"},
      {"time": "12-16s", "camera": "Medium shot, slight zoom in",                  "camera_movement": "zoom_in_slow",        "motion_intensity": 0.5, "action": "Vandana stops and touches a glowing vine. The vine reacts with rippling light.",            "character_focus": "Vandana",     "emotion": "curious_alert",       "visual_effects": "Glowing vine interaction, magical particles"},
      {"time": "16-20s", "camera": "POV from Shivs perspective, slight shake",     "camera_movement": "handheld_pov",        "motion_intensity": 0.7, "action": "The treasure map in Shivs hands vibrates and glows. His hands tremble with excitement.",    "character_focus": "Shiv_hands",  "emotion": "excited_nervous",     "visual_effects": "Map glowing, hand tremor, magical symbols"},
      {"time": "20-24s", "camera": "Extreme close-up on Shivs face",               "camera_movement": "static_intense",      "motion_intensity": 0.2, "action": "Extreme close-up of Shivs wide brown eyes. He hears a ghostly whisper calling his name.",   "character_focus": "Shiv_face",   "emotion": "frightened_alert",    "visual_effects": "Eye reflection detail, hair movement, cold breath"},
      {"time": "24-28s", "camera": "Wide establishing shot through trees",          "camera_movement": "dolly_reveal",        "motion_intensity": 0.5, "action": "A dark cave mouth appears in a limestone cliff. Mist pours out. Both children freeze.",      "character_focus": "both",        "emotion": "ominous_discovery",   "visual_effects": "Atmospheric mist, ominous lighting"},
      {"time": "28-32s", "camera": "Fast zoom into cave entrance",                  "camera_movement": "zoom_in_fast",        "motion_intensity": 0.8, "action": "Rapid zoom toward the cave mouth. Darkness swirls. Blue-green lights flicker within.",        "character_focus": "environment", "emotion": "threatening_mysterious", "visual_effects": "Swirling darkness, ethereal lights"},
      {"time": "32-36s", "camera": "Medium two-shot, slight push in",               "camera_movement": "push_in_slow",        "motion_intensity": 0.4, "action": "Shiv grabs Vandanas arm. Both stand frozen staring at the cave entrance.",                   "character_focus": "both",        "emotion": "fear_determination",  "visual_effects": "Character interaction, emotional expressions"},
      {"time": "36-40s", "camera": "Close-up on Vandanas backpack, tilt up",        "camera_movement": "tilt_up_reveal",      "motion_intensity": 0.5, "action": "Vandana takes a deep breath, reaches into her backpack and pulls out a brass flashlight.",    "character_focus": "Vandana",     "emotion": "brave_resolved",      "visual_effects": "Flashlight beam, volumetric lighting"},
      {"time": "40-44s", "camera": "Hero shot, low angle tracking",                  "camera_movement": "low_angle_hero",      "motion_intensity": 0.6, "action": "Vandana takes her first confident step toward the cave. Shiv gulps and follows.",             "character_focus": "both",        "emotion": "courageous_supportive", "visual_effects": "Heroic lighting, dust kicked up by footsteps"},
      {"time": "44-48s", "camera": "Dramatic silhouette shot, wide composition",     "camera_movement": "static_dramatic",     "motion_intensity": 0.3, "action": "Wide shot: both children silhouetted at the cave mouth, holding hands, looking into darkness.", "character_focus": "both_silhouette", "emotion": "unity_facing_unknown", "visual_effects": "Perfect silhouette, rim lighting, wind effects"},
    ]
  },
  "dialogue_with_timing": [
    {"time": 6,  "character": "Shiv",     "dialogue": "Vandana... kya tumne woh suna? Aisa laga jaise koi mera naam pukar raha hai.", "english_translation": "Vandana... did you hear that? It felt like someone was calling my name.", "emotion": "fearful_questioning", "voice_direction": "Whispered, trembling", "lip_sync_emphasis": "high"},
    {"time": 14, "character": "Vandana",  "dialogue": "Yeh sirf hawa hai, Shiv. Daro mat, main yahan hoon.",                        "english_translation": "Its just the wind, Shiv. Dont be scared, Im here.",                        "emotion": "reassuring_protective", "voice_direction": "Calm, confident", "lip_sync_emphasis": "high"},
    {"time": 25, "character": "The Cave", "dialogue": "*saanson jaisi awaaz*... andar aao...",                                        "english_translation": "*Breathing-like sound*... come inside...",                                   "emotion": "eerie_beckoning",     "voice_direction": "Hollow echoing whisper", "lip_sync_emphasis": "none"},
    {"time": 33, "character": "Shiv",     "dialogue": "Naksha... yeh kamp raha hai! Hum sahi jagah par hain.",                       "english_translation": "The map... its shaking! Were at the right place.",                         "emotion": "excited_scared",      "voice_direction": "Voice rising with excitement", "lip_sync_emphasis": "high"},
    {"time": 41, "character": "Vandana",  "dialogue": "Apni torch jalao. Ab peeche mudne ka koi rasta nahin hai.",                    "english_translation": "Turn on your torch. Theres no turning back now.",                         "emotion": "determined_brave",    "voice_direction": "Firm, determined", "lip_sync_emphasis": "high"},
  ],
  "audio": {
    "background_music": "Low ambient humming building into tense mystical choir with deep bass thumps, orchestral swells, mysterious woodwind flutes",
    "environment_sfx": "Dry leaves crunching, hollow echoing whispers, flashlight clicking, heavy breathing, eerie wind howling, distant bird calls, magical chimes, cave dripping water",
    "voice_processing": "Natural reverb for outdoor forest setting, slight echo near cave entrance"
  },
  "motion_guidance": {
    "global_motion": "Steady forward progression toward cave, building tension",
    "character_motion": "Realistic walking pace, natural body language, reactive expressions",
    "camera_motion": "Cinematic camera movements, smooth transitions, professional framing"
  }
}


def build_storyboard_from_json(scene_json: dict) -> List[dict]:
    """Build storyboard list from JSON scene definition."""
    storyboard = []
    shots = scene_json["story_action"]["shots"]
    for idx, shot in enumerate(shots):
        prev_shot    = shots[idx - 1] if idx > 0 else None
        full_prompt  = build_shot_prompt_pro(shot, scene_json, idx)
        cam_key, cam_file = get_camera_lora_for_shot(shot)
        storyboard.append({
            "id"         : f"shot_{idx+1:02d}",
            "prompt"     : full_prompt,
            "shot_data"  : shot,
            "camera_lora": cam_file if USE_MOTION_LORAS else None,
            "prev_shot"  : prev_shot,
        })
    return storyboard


def run_json_storyboard(scene_json: Optional[dict] = None,
                        base_seed: int = 42) -> Optional[str]:
    """
    Run the full JSON storyboard pipeline:
    - parse shots, auto-resume via cache check
    - 3-retry loop per shot calling generate_pro()
    - extract overlap anchors
    - stitch final movie
    Returns final movie path or None.
    """
    if scene_json is None:
        scene_json = SCENE_JSON

    storyboard = build_storyboard_from_json(scene_json)
    project    = scene_json.get("project_name", "LTX_Studio")
    input_dir  = "/content/ComfyUI/input"
    output_dir = "/content/ComfyUI/output"
    cache_dir  = f"{output_dir}/{project}_cache"
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(input_dir, exist_ok=True)

    print(f"\n\U0001f3ac JSON Storyboard Runner")
    print(f"   Project : {project}")
    print(f"   Shots   : {len(storyboard)}")

    # Auto-resume
    generated_clips   = []
    current_input_img = None
    start_index       = 0
    for i in range(len(storyboard)):
        expected_anchor = f"{input_dir}/anchor_scene_{i}.png"
        expected_clip   = f"{cache_dir}/scene_{i}.mp4"
        if os.path.exists(expected_anchor) and os.path.exists(expected_clip):
            current_input_img = expected_anchor
            start_index       = i + 1
            generated_clips.append(expected_clip)
        else:
            break
    if start_index > 0:
        print(f"   Resuming from shot {start_index + 1} (using cache)")

    prev_shot_success = True
    for i in tqdm(range(start_index, len(storyboard)), desc="Generating PRO shots"):
        scene     = storyboard[i]
        shot_data = scene["shot_data"]

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        strength = calculate_adaptive_strength(
            shot_data, scene.get("prev_shot"), prev_shot_success) if i > 0 else 0.0

        max_retries = 3
        success     = False
        attempt     = 0

        while attempt < max_retries and not success:
            attempt      += 1
            current_seed  = base_seed + (i * 100) + (attempt * 197)

            try:
                print(f"\n\U0001f4cd Shot {i+1}/{len(storyboard)} (Attempt {attempt})")
                # Build per-shot LoRA stack including camera LoRA if specified
                _cam_lora_file = scene.get("camera_lora")
                if _cam_lora_file and _cam_lora_file not in (None, "None", ""):
                    _shot_lora_stack = [
                        {"on": True,  "lora": _cam_lora_file, "guard": True, "strength": CAMERA_LORA_STRENGTH},
                    ] + [{"on": False, "lora": "None", "guard": False, "strength": 1.0}] * 9
                    _shot_lora_json = json.dumps(_shot_lora_stack)
                else:
                    _shot_lora_stack = LORA_STACK
                    _shot_lora_json  = LORA_STACK_JSON
                clip_path = generate_pro(
                    user_input           = scene["prompt"],
                    image_path           = current_input_img,
                    positive_prompt      = scene["prompt"],
                    negative_prompt      = build_negative_prompt_enhanced(),
                    seed                 = current_seed,
                    image_strength       = strength,
                    frames               = FRAMES,
                    width                = WIDTH,
                    height               = HEIGHT,
                    fps                  = FPS,
                    lora_stack           = _shot_lora_stack,
                    lora_stack_json      = _shot_lora_json,
                    output_prefix        = f"{project}_shot_{i+1:02d}",
                    bypass_easy_prompt   = True,
                )
                if clip_path:
                    next_anchor = extract_overlap_anchor_enhanced(
                        clip_path, output_folder=input_dir,
                        scene_idx=i, overlap=OVERLAP_FRAMES)
                    if next_anchor:
                        cached_clip = f"{cache_dir}/scene_{i}.mp4"
                        shutil.copy(clip_path, cached_clip)
                        generated_clips.append(cached_clip)
                        current_input_img = next_anchor
                        success           = True
                        prev_shot_success = True
                        print(f"   \u2705 Shot {i+1} complete!")
                    else:
                        print("   \u26a0\ufe0f Invalid anchor (retrying...)")
                        prev_shot_success = False
            except Exception as e:
                print(f"   \u274c Error on shot {i+1}: {e}")
                prev_shot_success = False

        if not success:
            print(f"\U0001f6d1 Failed to generate shot {i+1} after {max_retries} attempts. Stopping.")
            break

    if not generated_clips:
        print("\u274c No clips generated.")
        return None

    print(f"\n{'='*60}")
    print(f"\U0001f3ac Stitching final movie...")
    final_movie = stitch_videos_with_overlap_pro(
        generated_clips,
        output_filename=f"{project}_Complete_PRO.mp4",
        overlap_frames=OVERLAP_FRAMES)
    if final_movie:
        print(f"\n\U0001f389 MOVIE COMPLETE: {final_movie}")
        if SHOW_PREVIEWS:
            display_video(final_movie)
    return final_movie


print("\u2705 JSON storyboard runner ready.")


# ======================================================================
# CELL 14  --  run_storyboard() Simple Runner
# ======================================================================

# @title  { "single-column": true }
# @markdown ## 💥 14. Simple Storyboard Runner
# @markdown Each scene is a dict. When continuity enabled, last frame of
# @markdown scene N is automatically used as seed for scene N+1.

def run_storyboard(
    scenes:          List[Dict],
    use_continuity:  bool = None,
    tmp_dir:         str  = "/content/ComfyUI/input",
) -> List[Optional[str]]:
    """
    Run a list of scenes sequentially, optionally chaining last-frame continuity.

    Each scene dict supports these keys (all optional except user_input):
      user_input           -- story description for Easy Prompt
      image_path           -- seed image path (overridden by continuity if None)
      frames               -- frame count
      seed                 -- RNG seed
      output_prefix        -- filename prefix
      width / height       -- resolution (defaults to global WIDTH/HEIGHT)
      character_image_path -- character reference
      character_mode       -- i2v, anchor, both, none
      positive_prompt      -- manual prompt (used if BYPASS_EASY_PROMPT=True)
      negative_prompt      -- manual negative

    Returns list of output paths (None for failed scenes).
    """
    if use_continuity is None:
        use_continuity = USE_SCENE_CONTINUITY
    os.makedirs(tmp_dir, exist_ok=True)
    outputs     = []
    prev_output = None

    print("\U0001f3ac Storyboard Runner -- Starting")
    print(f"   Scenes    : {len(scenes)}")
    print(f"   Continuity: {use_continuity}")
    print("-" * 70)

    for i, scene in enumerate(scenes):
        scene_num = i + 1
        print(f"\n\U0001f3ac Scene {scene_num}/{len(scenes)}: {scene.get('output_prefix','Scene')}")
        print(f"   Input: {scene.get('user_input','')[:80]}...")

        _image_path = scene.get("image_path")
        if use_continuity and prev_output and _image_path is None:
            print(f"   Continuity: extracting last frame from scene {scene_num - 1}...")
            last_tensor = get_last_frame_tensor(prev_output)
            if last_tensor is not None:
                _cont_path = os.path.join(tmp_dir, f"_continuity_s{scene_num:02d}.jpg")
                pil_frame  = tensor_to_pil(last_tensor)
                pil_frame.save(_cont_path, "JPEG", quality=95)
                _image_path = _cont_path
                print(f"   \u2713 Continuity frame saved: {_cont_path}")
            else:
                print("   \u26a0\ufe0f  Could not extract last frame -- skipping continuity.")

        try:
            out = generate_pro(
                user_input           = scene.get("user_input", USER_INPUT),
                image_path           = _image_path,
                positive_prompt      = scene.get("positive_prompt", POSITIVE_PROMPT),
                negative_prompt      = scene.get("negative_prompt", NEGATIVE_PROMPT),
                width                = scene.get("width", WIDTH),
                height               = scene.get("height", HEIGHT),
                frames               = scene.get("frames", FRAMES),
                fps                  = scene.get("fps", FPS),
                seed                 = scene.get("seed", SEED),
                image_strength       = scene.get("image_strength", IMAGE_STRENGTH),
                character_image_path = scene.get("character_image_path", CHARACTER_IMAGE_PATH),
                character_strength   = scene.get("character_strength", CHARACTER_STRENGTH),
                character_mode       = scene.get("character_mode", CHARACTER_CONSISTENCY_MODE),
                character_name       = scene.get("character_name", CHARACTER_NAME),
                character_description= scene.get("character_description", CHARACTER_DESCRIPTION),
                output_prefix        = scene.get("output_prefix", OUTPUT_PREFIX),
            )
            outputs.append(out)
            prev_output = out
            print(f"   \u2705 Scene {scene_num} done -> {out}")
        except Exception as e:
            import traceback
            print(f"   \u274c Scene {scene_num} failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            outputs.append(None)
            prev_output = None

    print("\n" + "=" * 70)
    print("\U0001f3ac Storyboard Complete")
    print(f"   Total scenes : {len(scenes)}")
    print(f"   Successful   : {sum(1 for p in outputs if p)}")
    print(f"   Failed       : {sum(1 for p in outputs if not p)}")
    print("\n   Output paths:")
    for i, p in enumerate(outputs):
        status = "\u2705" if p else "\u274c"
        print(f"   {status} Scene {i+1}: {p or 'FAILED'}")
    print("=" * 70)
    return outputs


# Example SCENES list -- edit or extend
SCENES = [
    {
        "user_input"    : "a woman enters a dimly lit cafe, shaking rain from her coat, looks around",
        "image_path"    : CHARACTER_IMAGE_PATH,
        "frames"        : 97,
        "seed"          : SEED,
        "output_prefix" : f"Story01-{CHARACTER_NAME}",
        "character_image_path": CHARACTER_IMAGE_PATH,
        "character_mode": CHARACTER_CONSISTENCY_MODE,
    },
    {
        "user_input"    : "she sits at a window table, wraps her hands around a coffee cup, gazes out at the rain-streaked street",
        "image_path"    : None,
        "frames"        : 121,
        "seed"          : SEED + 1,
        "output_prefix" : f"Story02-{CHARACTER_NAME}",
        "character_image_path": CHARACTER_IMAGE_PATH,
        "character_mode": CHARACTER_CONSISTENCY_MODE,
    },
    {
        "user_input"    : "she notices something outside and leans forward, face half lit by neon glow",
        "image_path"    : None,
        "frames"        : 97,
        "seed"          : SEED + 2,
        "output_prefix" : f"Story03-{CHARACTER_NAME}",
        "character_image_path": CHARACTER_IMAGE_PATH,
        "character_mode": CHARACTER_CONSISTENCY_MODE,
    },
]

print("\u2705 Storyboard runner ready.")
print("   Edit SCENES list above, then set MODE='storyboard' in Cell 15.")


# ======================================================================
# CELL 15  --  Interactive Run Cell
# ======================================================================

# @title  { "single-column": true }
# @markdown ## 💥 15. Run
# @markdown Set MODE then run this cell. Seed auto-increments after success.

MODE = "single"  # @param ["single", "storyboard", "json_scene", "infinite"]
SEED_IMAGE_PATH = None  # @param {type:"string"}

# 12 Elena story beats for infinite mode
STORY_BEATS = [
    "Elena arrives at the entrance of a dimly lit urban apartment building at night, "
    "pushing through the glass door, rain dripping from her jacket",

    "She takes the elevator, watching the floor numbers tick upward, "
    "her reflection ghostly in the steel doors",

    "Elena unlocks her front door and steps inside the dark apartment, "
    "not turning on the lights, setting her keys on the counter quietly",

    "She moves to the kitchen, opens the fridge, stares blankly at the shelves, "
    "the cold blue light illuminating her face -- she is not hungry",

    "Elena notices a note on the kitchen table that was not there this morning, "
    "she picks it up slowly, her expression shifting from curiosity to alarm",

    "She grabs her phone from her bag, dials a number, presses it to her ear -- "
    "no answer. She tries again. Silence. She lowers the phone.",

    "Elena walks to the window and peers down at the wet street below, "
    "watching a black car parked at the curb with its engine running",

    "She goes to the bedroom wardrobe, pulls out a small travel bag, "
    "begins packing quickly -- clothes, passport, a laptop",

    "A sharp knock at the front door. Elena freezes mid-motion, "
    "bag half-packed, listening. Silence. Then another knock, louder.",

    "She approaches the door, looks through the peephole -- "
    "a man in a grey coat stands in the corridor, face turned away",

    "Elena slips out through the apartments service exit, "
    "moving quickly down the back stairwell, bag over her shoulder",

    "She emerges onto the rain-slicked alley behind the building, "
    "looks both ways, then runs toward the far end where a taxi waits with its light on",
]

_current_seed = SEED

try:
    if MODE == "single":
        output = generate_pro(
            user_input             = USER_INPUT,
            image_path             = IMAGE_PATH,
            positive_prompt        = POSITIVE_PROMPT,
            negative_prompt        = NEGATIVE_PROMPT,
            width                  = WIDTH,
            height                 = HEIGHT,
            frames                 = FRAMES,
            fps                    = FPS,
            seed                   = _current_seed,
            image_strength         = IMAGE_STRENGTH,
            character_image_path   = CHARACTER_IMAGE_PATH,
            character_strength     = CHARACTER_STRENGTH,
            character_mode         = CHARACTER_CONSISTENCY_MODE,
            character_name         = CHARACTER_NAME,
            character_description  = CHARACTER_DESCRIPTION,
            pass1_sigmas           = PASS1_SIGMAS,
            pass1_sampler          = PASS1_SAMPLER,
            pass1_cfg              = PASS1_CFG,
            pass2_sigmas           = PASS2_SIGMAS,
            pass2_sampler          = PASS2_SAMPLER,
            pass2_cfg              = PASS2_CFG,
            pass2_seed             = PASS2_SEED,
            pro_mode               = PRO_MODE,
            pro_steps              = PRO_STEPS,
            pro_scheduler          = PRO_SCHEDULER,
            pro_split_at           = PRO_SPLIT_AT,
            use_tiled_vae          = USE_TILED_VAE,
            tiled_spatial_tiles    = TILED_SPATIAL_TILES,
            tiled_spatial_overlap  = TILED_SPATIAL_OVERLAP,
            tiled_temporal_len     = TILED_TEMPORAL_LEN,
            tiled_temporal_overlap = TILED_TEMPORAL_OVERLAP,
            tiled_last_frame_fix   = TILED_LAST_FRAME_FIX,
            lora_stack             = LORA_STACK,
            lora_stack_json        = LORA_STACK_JSON,
            output_prefix          = OUTPUT_PREFIX,
        )
        if output:
            display_video(output)
        if AUTO_INCREMENT_SEED:
            SEED = _current_seed + 1
            print(f"\U0001f522 Next seed: {SEED}")

    elif MODE == "storyboard":
        print("\U0001f3ac Running storyboard mode...")
        storyboard_outputs = run_storyboard(
            scenes=SCENES, use_continuity=USE_SCENE_CONTINUITY)
        output = storyboard_outputs[-1] if storyboard_outputs else None
        if AUTO_INCREMENT_SEED:
            SEED = _current_seed + len(SCENES)
            print(f"\U0001f522 Next seed: {SEED}")

    elif MODE == "json_scene":
        print("\U0001f3ac Running JSON scene mode...")
        output = run_json_storyboard(SCENE_JSON)

    elif MODE == "infinite":
        print("\U0001f3ac Running Infinite Flow mode...")
        engine = InfiniteFlowEngine(
            character_bible = bible,
            llm_model       = LLM_MODEL,
            vision_model    = VISION_MODEL,
            width           = WIDTH,
            height          = HEIGHT,
            frames          = FRAMES,
            fps             = FPS,
            image_strength  = IMAGE_STRENGTH,
            creativity      = CREATIVITY,
            lora_triggers   = LORA_TRIGGERS,
            use_vision      = USE_VISION,
            use_tiled_vae   = USE_TILED_VAE,
            offline         = False,
            output_dir      = "/content/ComfyUI/output/infinite_flow",
            show_previews   = SHOW_PREVIEWS,
        )
        clip_paths = engine.run(
            story_beats     = STORY_BEATS,
            seed_image_path = SEED_IMAGE_PATH,
            base_seed       = _current_seed,
        )
        engine.save_bible("/content/character_bible.json")
        if len(clip_paths) > 1:
            final = engine.concat_all("infinite_final.mp4")
            print(f"\n\U0001f3ac Final video: {final}")
            if SHOW_PREVIEWS:
                display_video(final)
        output = clip_paths[-1] if clip_paths else None

    if DOWNLOAD_AFTER_GENERATE and output and os.path.exists(output):
        print("   \u2b07\ufe0f  Auto-downloading...")
        try:
            files.download(output)
        except Exception as e:
            print(f"   \u26a0\ufe0f  Download failed ({e})")

except KeyboardInterrupt:
    print("\n\u26a0\ufe0f  Interrupted -- partial output may be in /content/ComfyUI/output/")

except FileNotFoundError as e:
    print(f"\n\u274c Missing models: {e}")
    print("   Run Cell 2 to download, then retry Cell 15.")

except torch.cuda.OutOfMemoryError:
    cleanup_memory()
    print("\n\u274c CUDA Out of Memory")
    print(f"   Current settings: {WIDTH}x{HEIGHT}, {FRAMES} frames")
    print("   -- Suggested fixes --")
    print("   T4  (15 GB): WIDTH=768,  HEIGHT=512,  FRAMES=97")
    print("   L4  (24 GB): WIDTH=1024, HEIGHT=576,  FRAMES=161")
    print("   A100(40 GB): WIDTH=1280, HEIGHT=720,  FRAMES=241")
    print("   -- Also try --")
    print("   LLM_MODEL='3B'         in Cell 10  (reduce LLM VRAM footprint)")
    print("   USE_CHUNK_FF=True      in Cell 10  (chunk feedforward for T4)")
    print("   USE_TILED_VAE=True     in Cell 10  (tile VAE decode)")
    print("   TILED_SPATIAL_TILES=4  in Cell 10  (more tiles = less VRAM per tile)")
    print("   PRO_MODE=False         in Cell 10  (simpler sigma schedule)")

except RuntimeError as e:
    print(f"\n\u274c Runtime error: {e}")
    cleanup_memory()

except Exception as e:
    import traceback
    print(f"\n\u274c Error: {type(e).__name__}: {e}")
    traceback.print_exc()
    print("\n\U0001f4a1 Quick-fix reference:")
    print("   'UnetLoaderGGUF' not found       -> Cell 1: clone ComfyUI_GGUF")
    print("   'LTX2PromptArchitect' not found  -> Cell 1: clone LTX2EasyPrompt-LD")
    print("   'LTX2MasterLoaderLD' not found   -> Cell 1: clone LTX2-Master-Loader")
    print("   'LTXVImgToVideoInplace' missing  -> Cell 1: clone ComfyUI-LTXVideo")
    print("   'LTXVPreprocess' missing         -> Cell 1: clone ComfyUI-LTXVideo")
    print("   'LTXVCropGuides' missing         -> Cell 1: clone ComfyUI-LTXVideo")
    print("   'PathchSageAttentionKJ' missing  -> Cell 1: clone ComfyUI_KJNodes  (or set USE_SAGE_ATTENTION=False)")
    print("   'LTXVChunkFeedForward' missing   -> Cell 1: clone ComfyUI-LTXVideo  (or set USE_CHUNK_FF=False)")
    print("   'VHS_VideoCombine' missing       -> Cell 1: clone ComfyUI-VideoHelperSuite  (auto-fallback to CreateVideo)")
    print("   'ModelSamplingSD3' missing       -> set PRO_MODE=False in Cell 10")
    print("   'BasicScheduler' missing         -> set PRO_MODE=False in Cell 10")
    print("   DualCLIPLoader fp4 error         -> swap CLIP_NAME1 to fp8 in Cell 10")
    print("   Tiled VAE error                  -> set USE_TILED_VAE=False in Cell 10")
    print("   Deformed output in Pass 1        -> change SEED and re-run Cell 15")
    print("   Persistent deformation (3x same) -> change USER_INPUT / POSITIVE_PROMPT")
    print("   Character drift                  -> try CHARACTER_CONSISTENCY_MODE='both'")
    print("                                       or increase CHARACTER_STRENGTH")
