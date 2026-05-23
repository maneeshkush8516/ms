# -*- coding: utf-8 -*-
# ======================================================================
#  LTX STUDIO INFINITE FLOW ENGINE v1.0
#  Complete AI Video Production System - Google Colab Notebook
#
#  Features: Character Consistency, Easy Prompt Logic, T4 GPU optimization,
#            Voice Sync hooks, SVI-Pro chaining, Story-to-Prompt workflow
#
#  GPU Target: Google Colab T4 (15 GB VRAM)
#  Model: LTX-2 19B Q4_K_M GGUF (Kijai distilled)
#  Strategy: CLIP-first-then-UNet, aggressive cleanup between stages
# ======================================================================
#
# CELL ORDER:
#   Cell 1  - Environment Setup
#   Cell 2  - Model Downloads
#   Cell 3  - Core Imports & Utilities
#   Cell 4  - Character Bible & Consistency System
#   Cell 5  - Easy Prompt Engine
#   Cell 6  - SVI-Pro Infinite Flow Engine
#   Cell 7  - generate_clip() Function
#   Cell 8  - Configuration
#   Cell 9  - Story Input & Scene Definition
#   Cell 10 - Run Production


# ======================================================================
# CELL 1 - ENVIRONMENT SETUP
# ======================================================================

# @title { "single-column": true }
# @markdown ## 1. Prepare Environment & Install Custom Nodes
# @markdown Clones ComfyUI + all required custom nodes for the Infinite Flow Engine.

# -- Base Python packages --
!pip install torch torchvision torchaudio

%cd /content
from IPython.display import clear_output
clear_output()

!pip install -q torchsde einops diffusers accelerate nest_asyncio
!pip install -q av spandrel albumentations onnx opencv-python onnxruntime
!pip install -q imageio imageio-ffmpeg
!pip install -q "transformers>=4.43.0" accelerate qwen-vl-utils huggingface_hub

# -- ComfyUI (pinned branch) --
!git clone --branch ComfyUI_22_01_2026_v0.10.0 https://github.com/Isi-dev/ComfyUI.git
!pip install -r /content/ComfyUI/requirements.txt -q
clear_output()

# -- Custom nodes --
%cd /content/ComfyUI/custom_nodes

# Core KJNodes (pinned build)
!git clone --branch kj_1.2.6               https://github.com/Isi-dev/ComfyUI_KJNodes
# GGUF loader (UnetLoaderGGUF)
!git clone --branch ComfyUI_GGUF_22_01_2026 https://github.com/Isi-dev/ComfyUI_GGUF.git
# LTXVideo nodes (LTXVImgToVideoInplace, tiled VAE, etc.)
!git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git
# LTX2EasyPrompt-LD (LTX2PromptArchitect + LTX2VisionDescribe)
!git clone https://github.com/seanhan19911990-source/LTX2EasyPrompt-LD.git
# LTX2-Master-Loader (LTX2MasterLoaderLD 10-slot LoRA stacker)
!git clone https://github.com/seanhan19911990-source/LTX2-Master-Loader.git
# VideoHelperSuite (VHS_VideoCombine)
!git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git

# -- Install node requirements --
%cd /content/ComfyUI/custom_nodes/ComfyUI_KJNodes
!pip install -r requirements.txt -q

%cd /content/ComfyUI/custom_nodes/ComfyUI_GGUF
!pip install -r requirements.txt -q

%cd /content/ComfyUI/custom_nodes/ComfyUI-LTXVideo
!pip install -r requirements.txt -q 2>/dev/null || true

%cd /content/ComfyUI/custom_nodes/LTX2EasyPrompt-LD
!pip install -r requirements.txt -q 2>/dev/null || true

%cd /content/ComfyUI/custom_nodes/LTX2-Master-Loader
!pip install -r requirements.txt -q 2>/dev/null || true

# -- System tools --
import subprocess
subprocess.run(["apt-get", "-y", "install", "-qq", "aria2", "ffmpeg"],
               check=True, capture_output=True)

# -- Final setup --
%cd /content/ComfyUI
import os, sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, "/content/ComfyUI")

clear_output()
print("Environment setup complete.")
print("   Custom nodes installed:")
print("   - ComfyUI_KJNodes    (kj_1.2.6)")
print("   - ComfyUI_GGUF       (22_01_2026)")
print("   - ComfyUI-LTXVideo   (Lightricks)")
print("   - LTX2EasyPrompt-LD  (LoRa Daddy)")
print("   - LTX2-Master-Loader (LoRa Daddy)")
print("   - ComfyUI-VideoHelperSuite")



# ======================================================================
# CELL 2 - MODEL DOWNLOADS
# ======================================================================

# @title { "single-column": true }
# @markdown ## 2. Download All Model Weights
# @markdown Uses aria2c for fast parallel downloads. Skips cached files.

import os, subprocess
from pathlib import Path


def _dl(url: str, dest: str, filename: str = None) -> str:
    """aria2c download with skip-if-cached logic. Returns local filename."""
    Path(dest).mkdir(parents=True, exist_ok=True)
    fn = filename or url.split("/")[-1].split("?")[0]
    full = os.path.join(dest, fn)
    if os.path.exists(full) and os.path.getsize(full) > 1_000_000:
        print(f"  cached: {fn}")
        return fn
    cmd = ["aria2c", "--console-log-level=error", "--summary-interval=0",
           "--quiet", "-c", "-x", "16", "-s", "16", "-k", "1M",
           "-d", dest, "-o", fn, url]
    print(f"  downloading {fn}...", end=" ", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    print("done." if r.returncode == 0 else f"FAILED: {r.stderr.strip()}")
    return fn if r.returncode == 0 else None


# -- Source URLs --
KIJAI = "https://huggingface.co/Kijai/LTXV2_comfy/resolve/main"
KIJAI23 = "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main"
COMFYORG = "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files"
LIGHTRIX = "https://huggingface.co/Lightricks"
UNET_D = "/content/ComfyUI/models/unet"
TE_D = "/content/ComfyUI/models/text_encoders"
VAE_D = "/content/ComfyUI/models/vae"
UPD = "/content/ComfyUI/models/latent_upscale_models"
LR_D = "/content/ComfyUI/models/loras"

print("-- Core model downloads --")

# UNet: GGUF Q4_K_M distilled
_M_UNET = _dl(f"{KIJAI}/diffusion_models/ltx-2-19b-distilled_Q4_K_M.gguf", UNET_D)

# Text encoders - Gemma fp4 (RTX 5000 Blackwell; use fp8 for T4/A100)
_M_CLIP1 = _dl(f"{COMFYORG}/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors", TE_D)
# fp8 alternative for T4:
# _M_CLIP1 = _dl(f"{COMFYORG}/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors", TE_D)

# Embeddings connector (distilled)
_M_CLIP2 = _dl(
    f"{KIJAI}/text_encoders/ltx-2-19b-embeddings_connector_distill_bf16.safetensors", TE_D)

# VAEs
_M_VAE = _dl(f"{KIJAI}/VAE/LTX2_video_vae_bf16.safetensors", VAE_D)
_M_AVAE = _dl(f"{KIJAI}/VAE/LTX2_audio_vae_bf16.safetensors", VAE_D)
_M_TAE = _dl(f"{KIJAI23}/vae/taeltx2_3.safetensors", VAE_D)

# Spatial upscaler
_M_UP = _dl(
    f"{LIGHTRIX}/LTX-2/resolve/main/ltx-2-spatial-upscaler-x2-1.0.safetensors", UPD)

# LoRAs: IC + Camera Control
print("\n-- LoRA downloads --")
_LORA_URLS = {
    "Detailer": f"{LIGHTRIX}/LTX-2-19b-IC-LoRA-Detailer/resolve/main/ltx-2-19b-ic-lora-detailer.safetensors",
    "Canny": f"{LIGHTRIX}/LTX-2-19b-IC-LoRA-Canny-Control/resolve/main/ltx-2-19b-ic-lora-canny-control.safetensors",
    "Depth": f"{LIGHTRIX}/LTX-2-19b-IC-LoRA-Depth-Control/resolve/main/ltx-2-19b-ic-lora-depth-control.safetensors",
    "Pose": f"{LIGHTRIX}/LTX-2-19b-IC-LoRA-Pose-Control/resolve/main/ltx-2-19b-ic-lora-pose-control.safetensors",
    "Dolly-In": f"{LIGHTRIX}/LTX-2-19b-LoRA-Camera-Control-Dolly-In/resolve/main/ltx-2-19b-lora-camera-control-dolly-in.safetensors",
    "Dolly-Out": f"{LIGHTRIX}/LTX-2-19b-LoRA-Camera-Control-Dolly-Out/resolve/main/ltx-2-19b-lora-camera-control-dolly-out.safetensors",
    "Dolly-Left": f"{LIGHTRIX}/LTX-2-19b-LoRA-Camera-Control-Dolly-Left/resolve/main/ltx-2-19b-lora-camera-control-dolly-left.safetensors",
    "Dolly-Right": f"{LIGHTRIX}/LTX-2-19b-LoRA-Camera-Control-Dolly-Right/resolve/main/ltx-2-19b-lora-camera-control-dolly-right.safetensors",
    "Jib-Up": f"{LIGHTRIX}/LTX-2-19b-LoRA-Camera-Control-Jib-Up/resolve/main/ltx-2-19b-lora-camera-control-jib-up.safetensors",
    "Jib-Down": f"{LIGHTRIX}/LTX-2-19b-LoRA-Camera-Control-Jib-Down/resolve/main/ltx-2-19b-lora-camera-control-jib-down.safetensors",
    "Static": f"{LIGHTRIX}/LTX-2-19b-LoRA-Camera-Control-Static/resolve/main/ltx-2-19b-lora-camera-control-static.safetensors",
}
os.makedirs(LR_D, exist_ok=True)
for name, url in _LORA_URLS.items():
    r = _dl(url, LR_D)
    print(f"   {'done' if r else 'FAIL'}  {name}")

print("\nAll model files downloaded.")



# ======================================================================
# CELL 3 - CORE IMPORTS & UTILITIES
# ======================================================================

# @title { "single-column": true }
# @markdown ## 3. Imports, Helpers, Node Loader
# @markdown VRAM management, tensor helpers, video helpers, ComfyUI node loader.

import gc, re, json, time, shutil, warnings, asyncio, subprocess
import numpy as np
import torch
import cv2
from PIL import Image
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any, Union, Sequence, Mapping
from base64 import b64encode
from IPython.display import display, HTML, clear_output
from google.colab import files

warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, "/content/ComfyUI")

from nodes import NODE_CLASS_MAPPINGS, LoraLoaderModelOnly
import folder_paths


# -- VRAM management --

def aggressive_cleanup(label=""):
    """Aggressive VRAM cleanup: double gc + synchronize + empty_cache + ipc_collect."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()
    if label:
        _vram_print(label)


def _vram_free():
    """Quick VRAM free: gc + empty_cache + ipc_collect."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()


def _vram_print(tag=""):
    """Print current VRAM usage with visual bar."""
    if not torch.cuda.is_available():
        return
    u = torch.cuda.memory_allocated() / 1024**3
    t = torch.cuda.get_device_properties(0).total_memory / 1024**3
    filled = int(20 * u / t) if t else 0
    bar = "|" * filled + "." * (20 - filled)
    print(f"   VRAM [{bar}] {u:.1f}/{t:.1f} GB  {tag}")


def _vram_guard(min_free_gb=3.0):
    """Check available VRAM and force cleanup if below threshold."""
    if not torch.cuda.is_available():
        return 0.0
    free, total = torch.cuda.mem_get_info()
    free_gb = free / 1024**3
    if free_gb < min_free_gb:
        print(f"   [VRAM Guard] Only {free_gb:.1f} GB free, forcing cleanup...")
        aggressive_cleanup("VRAM guard pre-load")
        free, total = torch.cuda.mem_get_info()
        free_gb = free / 1024**3
        if free_gb < min_free_gb:
            print(f"   [VRAM Guard] Warning: still only {free_gb:.1f} GB free")
    return free_gb


# -- ComfyUI node output accessor --

def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Extract output at index from ComfyUI node result."""
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


# -- Tensor / image helpers --

def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL image to ComfyUI NHWC float tensor (1,H,W,3)."""
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """ComfyUI NHWC tensor to PIL Image."""
    if t.ndim == 4:
        t = t[0]
    return Image.fromarray((t.cpu().numpy() * 255).clip(0, 255).astype(np.uint8), "RGB")


def load_image_tensor(path: str) -> Optional[torch.Tensor]:
    """Load image file as ComfyUI NHWC tensor. Returns None if missing."""
    if not path or not os.path.exists(path):
        return None
    return pil_to_tensor(Image.open(path).convert("RGB"))


def get_last_frame_tensor(video_path: str) -> Optional[torch.Tensor]:
    """Extract last frame of video as NHWC float tensor (1,H,W,3)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n == 0:
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, n - 1)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(frame).float().unsqueeze(0) / 255.0


# -- Video helpers --

def display_video(path: str):
    """Display video inline in Colab."""
    if not path or not os.path.exists(path):
        print(f"   Video not found: {path}")
        return
    data = b64encode(open(path, "rb").read()).decode()
    display(HTML(
        '<video width=720 controls autoplay loop muted>'
        f'<source src="data:video/mp4;base64,{data}" type="video/mp4"></video>'))


def save_video_obj(video_obj, prefix="IFE") -> str:
    """Save a ComfyUI video object and return the output path."""
    from comfy_api.latest import Types
    w, h = video_obj.get_dimensions()
    folder, fname, ctr, _, _ = folder_paths.get_save_image_path(
        f"video/{prefix}", folder_paths.get_output_directory(), w, h)
    ext = Types.VideoContainer.get_extension("auto")
    path = os.path.join(folder, f"{fname}_{ctr:05}_.{ext}")
    video_obj.save_to(path, format=Types.VideoContainer("auto"),
                      codec="auto", metadata=None)
    return path


def concatenate_clips(clip_paths: List[str], output_path: str) -> str:
    """Concatenate video clips using ffmpeg concat demuxer."""
    valid = [p for p in clip_paths if p and os.path.exists(p)]
    if len(valid) < 2:
        return valid[0] if valid else None
    list_file = "/tmp/concat_list.txt"
    with open(list_file, "w") as f:
        for p in valid:
            f.write(f"file '{p}'\n")
    try:
        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                        "-i", list_file, "-c", "copy", output_path],
                       check=True, capture_output=True)
        print(f"   Concatenated {len(valid)} clips -> {output_path}")
        return output_path
    except Exception as e:
        print(f"   Concatenation failed ({e})")
        return None


# -- ComfyUI async node loader --

_NODES_LOADED = False


def _load_comfy_nodes():
    """Load all ComfyUI custom nodes (Jupyter/Colab safe)."""
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


# -- Audio VAE loader helper --

def _audio_vae(name: str):
    """Load audio VAE. Prefers VAELoaderKJ, falls back to VAELoader."""
    if "VAELoaderKJ" in NODE_CLASS_MAPPINGS:
        return NODE_CLASS_MAPPINGS["VAELoaderKJ"]().load_vae(
            vae_name=name, device="main_device", weight_dtype="fp16")
    return NODE_CLASS_MAPPINGS["VAELoader"]().load_vae(vae_name=name)


print("Core utilities ready.")



# ======================================================================
# CELL 4 - CHARACTER BIBLE & CONSISTENCY SYSTEM
# ======================================================================

# @title { "single-column": true }
# @markdown ## 4. Character Bible & VisionDescribe Engine
# @markdown Cross-scene character consistency and standalone vision analysis.


class CharacterBible:
    """
    Records named character attributes and serialises them as a prompt-injection
    block. The block is passed to EasyPromptEngine as character_bible so the LLM
    receives a hard [NON-NEGOTIABLE] constraint preventing attribute drift.
    """

    def __init__(self):
        self._chars: Dict[str, dict] = {}

    def add(self, name: str, **attributes):
        """Manually define a character with keyword attributes."""
        self._chars[name] = dict(attributes)

    def extract_from_description(self, name: str, description: str):
        """Store a raw VisionDescribe output under a character name."""
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
        """Save bible to JSON file."""
        with open(path, "w") as f:
            json.dump(self._chars, f, indent=2)
        print(f"   [Bible] Saved -> {path}")

    def load(self, path: str):
        """Load bible from JSON file."""
        with open(path) as f:
            self._chars = json.load(f)
        print(f"   [Bible] Loaded from {path}")

    def __repr__(self):
        return f"CharacterBible({self.names()})"


class VisionDescribeEngine:
    """
    Analyses an image and returns a 100-130 word scene description.
    Standalone Qwen2.5-VL - loads, describes, unloads immediately.
    """

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
        """Describe an image. Loads model, runs inference, unloads."""
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from huggingface_hub import snapshot_download
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            raise ImportError("[VisionDescribe] pip install qwen-vl-utils")

        _vram_guard(min_free_gb=3.0)

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
        print(f"   [VisionDescribe] Loading {self.model_key} ...")
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
        _vram_free()
        print(f"   [VisionDescribe] Done ({len(desc.split())} words).")
        return desc


print("Character Bible & VisionDescribe Engine ready.")



# ======================================================================
# CELL 5 - EASY PROMPT ENGINE
# ======================================================================

# @title { "single-column": true }
# @markdown ## 5. Easy Prompt Engine
# @markdown Full cinematic LLM expansion with character bible lock,
# @markdown pacing control, and mandatory quality keywords.

# -- Negative prompt builder --

_NEG_BASE = (
    "blurry, out of focus, low quality, worst quality, jpeg artifacts, "
    "static, no motion, frozen, duplicate, watermark, text, signature, "
    "poorly drawn, bad anatomy, deformed, disfigured, extra limbs, "
    "missing limbs, overexposed, underexposed, grainy, noise, flickering"
)


def _build_neg(result: str, user_input: str) -> str:
    """Build context-aware negative prompt based on scene content."""
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
    if any(w in c for w in ["pussy", "cock", "penis", "vagina", "nude", "naked"]):
        extras.append("censored, mosaic, pixelated, black bar, blurred genitals")
    return ", ".join([_NEG_BASE] + extras)


# -- Mandatory quality keywords appended to every generated prompt --
_MANDATORY_KEYWORDS = (
    "Ultra HDR, 3D intricate details, vibrant colors, realistic lighting, "
    "Dramatic Lighting, Enhanced Clarity, Brilliant Highlights, "
    "Hyperrealistic Detailing, cinematic"
)


class EasyPromptEngine:
    """
    Expands a simple story beat into a dense cinematic LTX-2 prompt.
    Loads the LLM, generates, cleans output, appends mandatory keywords,
    then unloads to free VRAM.
    """

    MODELS = {
        "3B": "huihui-ai/Llama-3.2-3B-Instruct-abliterated",
        "8B": "mlabonne/NeuralDaredevil-8B-abliterated",
        "14B": "huihui-ai/Huihui-Qwen3-14B-abliterated-v2",
    }

    SYSTEM_PROMPT = (
        "You are a cinematic prompt writer for LTX-2, an AI video generation model. "
        "Expand the user's idea into a rich, video-ready prompt.\n\n"
        "PRIORITY ORDER:\n"
        "1. Video style & genre (slow-burn thriller, documentary, editorial, action)\n"
        "2. Camera angle & shot type (low-angle close-up, bird's-eye wide, Dutch angle)\n"
        "3. Character description - age MUST be a specific number, body type, hair, skin, clothing\n"
        "4. Scene & environment (location, time of day, lighting, colour palette, atmosphere)\n"
        "5. Action & motion - continuous present-tense sequence\n"
        "6. Camera movement - prose only, no screenplay brackets\n"
        "7. Audio - max 2 ambient sounds, dialogue as inline prose with attribution\n\n"
        "RULES:\n"
        "- Present tense throughout.\n"
        "- 8-12 sentences of dense flowing prose - no bullet lists.\n"
        "- Fill the full token budget. Do not stop early.\n"
        "- Output ONLY the expanded prompt. No preamble. No trailing notes."
    )

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
        """Load the LLM into VRAM."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from huggingface_hub import snapshot_download
        key = self.model_size
        if self._model is not None and self._loaded_key == key:
            return
        if self._model is not None:
            self._unload()
        _vram_guard(min_free_gb=4.0)
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
        """Unload LLM from VRAM."""
        if self._model is not None:
            del self._model
        self._model = None
        self._tok = None
        self._loaded_key = None
        _vram_free()
        print("   [EasyPrompt] VRAM cleared.")

    def _stop_ids(self) -> List[int]:
        """Get stop token IDs for generation."""
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
        """Clean LLM output of artifacts."""
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
    ) -> Tuple[str, str]:
        """
        Returns (positive_prompt, negative_prompt).

        character_bible is injected as a hard [CHARACTER BIBLE - NON-NEGOTIABLE]
        block so the LLM cannot alter any locked attribute across clips.
        Mandatory quality keywords are appended to every generated prompt.
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

        # Pacing constraint
        ordinal = {2: "2nd", 3: "3rd"}.get(action_count, f"{action_count}th")
        if action_count > 1:
            pacing = (
                f"This clip is {real_seconds:.0f}s. Write EXACTLY {action_count} "
                f"distinct actions. HARD STOP after the {ordinal} action. "
                f"Write ~{token_budget} tokens."
            )
        else:
            pacing = (
                f"This clip is {real_seconds:.0f}s. Write EXACTLY 1 action. "
                f"HARD STOP after it. ~{token_budget} tokens."
            )

        # Character bible lock
        bible_clause = ""
        if character_bible.strip():
            bible_clause = (
                f"\n[CHARACTER BIBLE - NON-NEGOTIABLE: Every character attribute below "
                f"MUST remain exactly as described. Do NOT alter hair, age, skin, "
                f"clothing, or any other attribute. This overrides any inference:\n"
                f"{character_bible.strip()}\n]"
            )

        # Scene context
        if scene_context.strip():
            effective = (
                f"[SCENE CONTEXT FROM IMAGE - authoritative]\n"
                f"{scene_context.strip()}\n\n"
                f"[USER DIRECTION]\n{user_input.strip()}"
            )
        else:
            effective = user_input.strip()

        # LoRA trigger injection
        lora_clause = (f"\n[LORA: Begin prompt with: {lora_triggers.strip()}]"
                       if lora_triggers.strip() else "")

        user_content = effective + bible_clause + lora_clause + f"\n[PACING: {pacing}]"

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        is_qwen3 = "Qwen3" in self.MODELS.get(self.model_size, "")
        template_kwargs = {"enable_thinking": False} if is_qwen3 else {}
        raw = self._tok.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True,
            **template_kwargs)

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

        # Append mandatory quality keywords to every generated prompt
        result = result.rstrip(".") + ". " + _MANDATORY_KEYWORDS

        neg = _build_neg(result, user_input)

        if not self.keep_loaded:
            self._unload()

        print(f"   [EasyPrompt] Done ({len(result.split())} words).")
        return result, neg


print("Easy Prompt Engine ready.")



# ======================================================================
# CELL 6 - SVI-PRO INFINITE FLOW ENGINE
# ======================================================================

# @title { "single-column": true }
# @markdown ## 6. SVI-Pro Infinite Flow Engine
# @markdown Camera LoRA mapping, adaptive anchor, enhanced extraction,
# @markdown and the main InfiniteFlowEngine orchestrator.

# -- Camera LoRA Mapping --

CAMERA_LORA_MAPPING = {
    "dolly_forward": "ltx-2-19b-lora-camera-control-dolly-in.safetensors",
    "dolly_backward": "ltx-2-19b-lora-camera-control-dolly-out.safetensors",
    "dolly_left": "ltx-2-19b-lora-camera-control-dolly-left.safetensors",
    "dolly_right": "ltx-2-19b-lora-camera-control-dolly-right.safetensors",
    "pan_left": "ltx-2-19b-lora-camera-control-dolly-left.safetensors",
    "pan_right": "ltx-2-19b-lora-camera-control-dolly-right.safetensors",
    "tilt_up": "ltx-2-19b-lora-camera-control-jib-up.safetensors",
    "tilt_down": "ltx-2-19b-lora-camera-control-jib-down.safetensors",
    "zoom_in": "ltx-2-19b-lora-camera-control-dolly-in.safetensors",
    "zoom_out": "ltx-2-19b-lora-camera-control-dolly-out.safetensors",
    "static": "ltx-2-19b-lora-camera-control-static.safetensors",
}


def _build_lora_stack(ic_lora="detailer", ic_strength=0.4,
                      camera_lora_file=None, camera_strength=0.8):
    """Build 10-slot LoRA stack for LTX2MasterLoaderLD."""
    _IC_FILES = {
        "none": "None",
        "detailer": "ltx-2-19b-ic-lora-detailer.safetensors",
        "canny": "ltx-2-19b-ic-lora-canny-control.safetensors",
        "depth": "ltx-2-19b-ic-lora-depth-control.safetensors",
        "pose": "ltx-2-19b-ic-lora-pose-control.safetensors",
    }
    ic_file = _IC_FILES.get(ic_lora.lower(), "None")
    cam_file = camera_lora_file or "None"

    stack = [
        {"on": ic_file != "None", "lora": ic_file, "guard": False, "strength": ic_strength},
        {"on": cam_file != "None", "lora": cam_file, "guard": False, "strength": camera_strength},
    ]
    for _ in range(8):
        stack.append({"on": False, "lora": "None", "guard": False, "strength": 1.0})
    return stack


def _apply_loras(unet, lora_stack):
    """Apply LoRA stack. Tries LTX2MasterLoaderLD, falls back to manual loop."""
    active = [s for s in lora_stack
              if s.get("on") and s.get("lora") not in (None, "None", "")]
    if not active:
        return unet

    if "LTX2MasterLoaderLD" in NODE_CLASS_MAPPINGS:
        try:
            node = NODE_CLASS_MAPPINGS["LTX2MasterLoaderLD"]()
            result = getattr(node, node.FUNCTION)(
                model=unet, clip=None, stack_data=json.dumps(lora_stack))
            print(f"   [LoRA] {len(active)} slot(s) via LTX2MasterLoaderLD")
            return get_value_at_index(result, 0)
        except Exception as e:
            print(f"   [LoRA] MasterLoader failed ({e}), manual fallback")

    for slot in active:
        name, strength = slot["lora"], slot["strength"]
        try:
            unet = LoraLoaderModelOnly().load_lora_model_only(unet, name, strength)[0]
            print(f"   [LoRA] {name} @ {strength}")
        except Exception as e:
            print(f"   [LoRA] {name} failed: {e}")
    return unet


def calculate_adaptive_strength(shot_data, prev_shot_data,
                                prev_success=True,
                                anchor_high=0.85, anchor_low=0.70):
    """
    Calculate anchor strength based on motion intensity change between shots,
    character focus change, and previous shot success.
    """
    strength = anchor_high

    if prev_shot_data:
        motion_change = abs(
            shot_data.get("motion_intensity", 0.5) -
            prev_shot_data.get("motion_intensity", 0.5)
        )
        if motion_change > 0.4:
            strength -= 0.10
        elif motion_change < 0.2:
            strength += 0.05

        if shot_data.get("character_focus") != prev_shot_data.get("character_focus"):
            strength -= 0.05

    if not prev_success:
        strength -= 0.10

    strength = max(anchor_low, min(anchor_high, strength))
    return strength


def enhanced_anchor_extraction(video_path: str, output_folder: str = "/content/ComfyUI/input",
                               scene_idx: int = 0, overlap: int = 16) -> Optional[str]:
    """
    Enhanced anchor extraction with:
    - Multi-frame candidates near overlap point
    - Brightness validation (reject black frames)
    - Sharpness scoring via Laplacian variance
    Returns path to saved anchor image or None.
    """
    if not os.path.exists(video_path):
        print(f"   Video not found: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    candidates = [
        total_frames - overlap,
        total_frames - overlap - 2,
        total_frames - overlap + 2,
        total_frames - 1,
    ]

    best_frame = None
    best_score = 0.0

    for candidate_idx in candidates:
        if candidate_idx < 0 or candidate_idx >= total_frames:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, candidate_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(cv2.mean(gray)[0])
        if brightness < 5:
            continue

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(laplacian.var())
        score = brightness + sharpness * 0.1

        if score > best_score:
            best_score = score
            best_frame = frame

    cap.release()

    if best_frame is not None:
        os.makedirs(output_folder, exist_ok=True)
        filename = f"anchor_scene_{scene_idx:03d}.png"
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, best_frame)
        print(f"   Anchor extracted (quality score: {best_score:.2f})")
        return save_path
    else:
        print(f"   No valid anchor frame found")
        return None


class InfiniteFlowEngine:
    """
    Generates a sequence of clips where each clip is chained to the next
    via last-frame anchor extraction. Uses SVI-Pro pattern:
      1. Generate clip N via generate_clip()
      2. Extract last OVERLAP_FRAMES frames as anchor (enhanced_anchor_extraction)
      3. Use anchor as seed image for clip N+1 via LTXVImgToVideoInplace
      4. Adaptive anchor strength based on motion change
      5. Rolling scene context (last 2 expanded prompts)
      6. Cache-based resume (skip if clip already exists on disk)
      7. Progress tracking with time estimates
      8. Retry logic (3 attempts, progressive frame reduction on OOM)
    """

    def __init__(self, character_bible: Optional[CharacterBible] = None,
                 vision_engine: Optional[VisionDescribeEngine] = None,
                 prompt_engine: Optional[EasyPromptEngine] = None,
                 config: Optional[Dict] = None):
        self.bible = character_bible or CharacterBible()
        self.vision = vision_engine or VisionDescribeEngine("3B-fast")
        self.prompt_engine = prompt_engine or EasyPromptEngine("8B")
        cfg = config or {}

        self.width = cfg.get("WIDTH", 848)
        self.height = cfg.get("HEIGHT", 480)
        self.fps = cfg.get("FPS", 24)
        self.frames = cfg.get("FRAMES", 121)
        self.base_seed = cfg.get("BASE_SEED", 42)
        self.use_vision = cfg.get("USE_VISION", True)
        self.use_tiled_vae = cfg.get("USE_TILED_VAE", True)
        self.use_adaptive_strength = cfg.get("USE_ADAPTIVE_STRENGTH", True)
        self.anchor_high = cfg.get("ANCHOR_STRENGTH_HIGH", 0.85)
        self.anchor_low = cfg.get("ANCHOR_STRENGTH_LOW", 0.70)
        self.creativity = cfg.get("CREATIVITY", 0.9)
        self.overlap_frames = cfg.get("OVERLAP_FRAMES", 16)
        self.ic_lora = cfg.get("IC_LORA", "detailer")
        self.ic_lora_strength = cfg.get("IC_LORA_STRENGTH", 0.4)
        self.camera_lora_strength = cfg.get("CAMERA_LORA_STRENGTH", 0.8)
        self.tiled_stiles = cfg.get("TILED_SPATIAL_TILES", 2)
        self.tiled_tlen = cfg.get("TILED_TEMPORAL_LEN", 48)
        self.project_name = cfg.get("PROJECT_NAME", "LTX_Studio_InfiniteFlow")
        self.output_dir = cfg.get("OUTPUT_DIR", "/content/ComfyUI/output")
        self.show_previews = cfg.get("SHOW_PREVIEWS", True)
        self.auto_increment_seed = cfg.get("AUTO_INCREMENT_SEED", True)
        self.pass1_sigmas = cfg.get("PASS1_SIGMAS",
            "1., 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0")
        self.pass2_sigmas = cfg.get("PASS2_SIGMAS", "0.909375, 0.725, 0.421875, 0.0")
        self.pass1_sampler = cfg.get("PASS1_SAMPLER", "euler")
        self.pass2_sampler = cfg.get("PASS2_SAMPLER", "gradient_estimation")

        self._clip_paths: List[str] = []
        self._failed_shots: List[int] = []
        self._scene_history: List[str] = []

    def _rolling_ctx(self) -> str:
        """Return last 2 expanded prompts as rolling context."""
        return "\n".join(self._scene_history[-2:])

    def get_clip_paths(self) -> List[str]:
        """Return list of successfully generated clip paths."""
        return self._clip_paths

    def get_summary(self) -> Dict:
        """Return production summary as dict."""
        return {
            "total_shots": len(self._clip_paths) + len(self._failed_shots),
            "successful": len(self._clip_paths),
            "failed": len(self._failed_shots),
            "failed_shots": self._failed_shots,
            "clip_paths": self._clip_paths,
        }

    def run(self, storyboard: List[Dict], seed_image_path: Optional[str] = None,
            base_seed: Optional[int] = None) -> List[str]:
        """
        Process each entry in the storyboard list.
        Each entry is a dict with keys: prompt, camera_movement, motion_intensity, etc.
        Returns list of output .mp4 paths.
        """
        if base_seed is not None:
            self.base_seed = base_seed

        cache_dir = f"{self.output_dir}/{self.project_name}_cache"
        os.makedirs(cache_dir, exist_ok=True)
        input_dir = "/content/ComfyUI/input"
        os.makedirs(input_dir, exist_ok=True)

        current_image: Optional[torch.Tensor] = None
        prev_shot_data: Optional[Dict] = None
        prev_success = True

        # Load seed image
        if seed_image_path and os.path.exists(seed_image_path):
            current_image = load_image_tensor(seed_image_path)
            print(f"[IFE] Seed image loaded: {seed_image_path}")

            if not self.bible.has_characters() and self.use_vision:
                print("[IFE] Extracting Character Bible from seed image...")
                desc = self.vision.describe(current_image)
                _vram_free()
                self.bible.extract_from_description("Main Character", desc)
                self._scene_history.append(desc[:400])

        # Cache-based resume
        start_index = 0
        for i in range(len(storyboard)):
            cached_clip = f"{cache_dir}/scene_{i:03d}.mp4"
            if os.path.exists(cached_clip):
                self._clip_paths.append(cached_clip)
                start_index = i + 1
                lf = get_last_frame_tensor(cached_clip)
                if lf is not None:
                    current_image = lf
            else:
                break

        if start_index > 0:
            print(f"[IFE] Resuming from shot {start_index + 1} ({start_index} cached)")

        print(f"\n[IFE] === INFINITE FLOW ENGINE ===")
        print(f"[IFE] Shots: {len(storyboard)}  |  Size: {self.width}x{self.height}")
        print(f"[IFE] FPS: {self.fps}  |  Frames/shot: {self.frames}")
        print(f"[IFE] Bible: {self.bible.names() or 'empty'}")
        print(f"[IFE] Mode: {'I2V' if current_image is not None else 'T2V'}\n")

        production_start = time.time()

        for i in range(start_index, len(storyboard)):
            entry = storyboard[i]
            shot_num = i + 1
            beat_text = entry.get("prompt", entry.get("beat", ""))
            shot_data = entry.get("shot_data", entry)

            print(f"\n[IFE] --- Shot {shot_num}/{len(storyboard)} ---")
            print(f"[IFE]   {beat_text[:80]}...")

            # Adaptive strength
            if self.use_adaptive_strength and current_image is not None and i > 0:
                strength = calculate_adaptive_strength(
                    shot_data, prev_shot_data, prev_success,
                    self.anchor_high, self.anchor_low)
            else:
                strength = self.anchor_high if current_image is not None else 0.0

            # Vision describe on anchor (from beat 2 onward)
            scene_ctx = ""
            if i > 0 and current_image is not None and self.use_vision:
                try:
                    scene_ctx = self.vision.describe(current_image)
                    _vram_free()
                except Exception as e:
                    print(f"   [IFE] Vision failed ({e}), using rolling context")
                    scene_ctx = self._rolling_ctx()
            elif i == 0:
                scene_ctx = self._rolling_ctx()

            # EasyPrompt expand with CharacterBible lock
            beat_seed = self.base_seed + i * 1000 if self.auto_increment_seed else self.base_seed
            try:
                expanded_prompt, neg_prompt = self.prompt_engine.generate(
                    user_input=beat_text,
                    frame_count=self.frames,
                    creativity=self.creativity,
                    seed=beat_seed,
                    scene_context=scene_ctx,
                    character_bible=self.bible.to_prompt_block(),
                )
                _vram_free()
            except Exception as e:
                print(f"   [IFE] EasyPrompt failed ({e}), using raw beat text")
                expanded_prompt = beat_text + ". " + _MANDATORY_KEYWORDS
                neg_prompt = _NEG_BASE

            self._scene_history.append(expanded_prompt[:400])
            if len(self._scene_history) > 2:
                self._scene_history = self._scene_history[-2:]

            print(f"   [IFE] Prompt: {expanded_prompt[:120]}...")

            # Determine camera LoRA
            camera_movement = shot_data.get("camera_movement", "static")
            camera_lora_file = CAMERA_LORA_MAPPING.get(camera_movement)

            lora_stack = _build_lora_stack(
                ic_lora=self.ic_lora,
                ic_strength=self.ic_lora_strength,
                camera_lora_file=camera_lora_file,
                camera_strength=self.camera_lora_strength)

            # Retry loop (max 3 attempts)
            max_retries = 3
            success = False
            clip_path = None
            current_seed = beat_seed
            retry_frames = self.frames
            retry_width = self.width
            retry_height = self.height
            retry_tiled = self.use_tiled_vae

            for attempt in range(max_retries):
                try:
                    clip_path = generate_clip(
                        image_tensor=current_image,
                        prompt=expanded_prompt,
                        neg_prompt=neg_prompt,
                        width=retry_width,
                        height=retry_height,
                        frames=retry_frames,
                        fps=self.fps,
                        seed=current_seed,
                        image_strength=strength,
                        pass1_sigmas=self.pass1_sigmas,
                        pass1_sampler=self.pass1_sampler,
                        pass2_sigmas=self.pass2_sigmas,
                        pass2_sampler=self.pass2_sampler,
                        use_tiled_vae=retry_tiled,
                        tiled_stiles=self.tiled_stiles,
                        tiled_tlen=self.tiled_tlen,
                        lora_stack=lora_stack,
                        output_prefix=f"{self.project_name}_{shot_num:03d}",
                    )
                    if clip_path and os.path.exists(clip_path):
                        success = True
                        break
                except torch.cuda.OutOfMemoryError:
                    aggressive_cleanup("OOM recovery")
                    current_seed += 1
                    retry_frames = max(retry_frames - 24, 25)
                    if attempt >= 1:
                        retry_tiled = True
                        retry_width = max(retry_width - 128, 512)
                        retry_height = max(retry_height - 64, 320)
                    print(f"   [IFE] OOM, reducing to {retry_frames}f "
                          f"{retry_width}x{retry_height}, seed {current_seed}...")
                except Exception as e:
                    print(f"   [IFE] Error: {type(e).__name__}: {e}")
                    current_seed += 1

            if success and clip_path:
                cached_path = f"{cache_dir}/scene_{i:03d}.mp4"
                shutil.copy(clip_path, cached_path)
                self._clip_paths.append(cached_path)
                prev_success = True

                # Enhanced anchor extraction for next shot
                anchor_path = enhanced_anchor_extraction(
                    cached_path, output_folder=input_dir,
                    scene_idx=i, overlap=self.overlap_frames)
                if anchor_path:
                    current_image = load_image_tensor(anchor_path)
                else:
                    lf = get_last_frame_tensor(cached_path)
                    current_image = lf

                print(f"   [IFE] Shot {shot_num} DONE")
                if self.show_previews:
                    display_video(cached_path)
            else:
                self._failed_shots.append(shot_num)
                prev_success = False
                current_image = None
                print(f"   [IFE] Shot {shot_num} FAILED after {max_retries} attempts")

            prev_shot_data = shot_data

            # Progress tracking
            elapsed = time.time() - production_start
            completed = i - start_index + 1
            avg_per_shot = elapsed / completed if completed > 0 else 0
            remaining = avg_per_shot * (len(storyboard) - i - 1)
            print(f"   [IFE] Elapsed: {elapsed/60:.1f}min | "
                  f"Est. remaining: {remaining/60:.1f}min")

        print(f"\n[IFE] === {len(storyboard)} scenes processed ===")
        return self._clip_paths

    def concat_all(self, output_name: str = "full_video.mp4") -> Optional[str]:
        """Concatenate all generated clips into one final video."""
        out = os.path.join(self.output_dir, output_name)
        return concatenate_clips(self._clip_paths, out)


print("SVI-Pro Infinite Flow Engine ready.")



# ======================================================================
# CELL 7 - GENERATE CLIP FUNCTION
# ======================================================================

# @title { "single-column": true }
# @markdown ## 7. generate_clip() - Two-Pass LTX-2 19B GGUF Pipeline
# @markdown VRAM-safe CLIP-first-then-UNet pattern for T4 GPU.
# @markdown Supports I2V (image-to-video) and T2V (text-to-video) modes.


def generate_clip(
    image_tensor: Optional[torch.Tensor],
    prompt: str,
    neg_prompt: str,
    width: int = 848,
    height: int = 480,
    frames: int = 121,
    fps: int = 24,
    seed: int = 42,
    image_strength: float = 1.0,
    pass1_sigmas: str = "1., 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0",
    pass1_sampler: str = "euler",
    pass1_cfg: float = 1.0,
    pass2_sigmas: str = "0.909375, 0.725, 0.421875, 0.0",
    pass2_sampler: str = "gradient_estimation",
    pass2_cfg: float = 1.0,
    pass2_seed: int = 0,
    use_tiled_vae: bool = True,
    tiled_stiles: int = 2,
    tiled_soverlap: int = 8,
    tiled_tlen: int = 48,
    tiled_toverlap: int = 4,
    lora_stack: Optional[List[Dict]] = None,
    output_prefix: str = "IFE",
    unet_model: str = "ltx-2-19b-distilled_Q4_K_M.gguf",
    clip_name1: str = "gemma_3_12B_it_fp4_mixed.safetensors",
    clip_name2: str = "ltx-2-19b-embeddings_connector_distill_bf16.safetensors",
    vae_model: str = "LTX2_video_vae_bf16.safetensors",
    vae_audio_model: str = "LTX2_audio_vae_bf16.safetensors",
    upscaler_model: str = "ltx-2-spatial-upscaler-x2-1.0.safetensors",
) -> str:
    """
    Two-pass LTX-2 19B GGUF generation for one clip.
    T4-safe: CLIP is loaded, encodes text, then deleted before UNet loads.
    UNet is deleted after Pass 2, before decode.

    VRAM sequence (peak ~13 GB on T4):
      1. Load CLIP Gemma (~6-8 GB) -> encode text -> DELETE CLIP -> free
      2. Load UNet GGUF (~10 GB) -> apply LoRAs
      3. Load VAE_video + VAE_audio + upscaler
      4. Prepare latents (I2V if image provided, else T2V)
      5. Pass 1 (ManualSigmas + euler + CFGGuider)
      6. Pass 2 (spatial upscale + gradient_estimation) -> DELETE UNet
      7. Decode video (tiled or standard) + audio -> DELETE VAEs
      8. Save via CreateVideo + save_video_obj

    Returns: output video file path.
    """
    _load_comfy_nodes()

    img_bypass = image_tensor is None
    img_str = image_strength if not img_bypass else 0.0

    print(f"   [Clip] {'I2V' if not img_bypass else 'T2V'}  "
          f"{width}x{height}  {frames}f  seed={seed}")
    _vram_print("before clip")

    with torch.inference_mode():

        # ============================================================
        # STEP 1: Load CLIP -> encode text -> DELETE CLIP (frees 6-8 GB)
        # ============================================================
        print("   [Clip] Loading CLIP...")
        clip_ld = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
        try:
            clip_raw = get_value_at_index(
                clip_ld.load_clip(clip_name1=clip_name1, clip_name2=clip_name2,
                                  type="ltxv", device="default"), 0)
        except Exception as e:
            print(f"   fp4 failed ({e}), trying fp8...")
            fp8 = "gemma_3_12B_it_fp8_scaled.safetensors"
            clip_raw = get_value_at_index(
                clip_ld.load_clip(clip_name1=fp8, clip_name2=clip_name2,
                                  type="ltxv", device="default"), 0)

        cte = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cond_pos = cte.encode(text=prompt, clip=clip_raw)

        zero_out = NODE_CLASS_MAPPINGS["ConditioningZeroOut"]()
        cond_0 = zero_out.zero_out(conditioning=get_value_at_index(cond_pos, 0))

        ltxv_cn = NODE_CLASS_MAPPINGS["LTXVConditioning"]()
        cond = ltxv_cn.EXECUTE_NORMALIZED(
            frame_rate=float(fps),
            positive=get_value_at_index(cond_pos, 0),
            negative=get_value_at_index(cond_0, 0))

        # DELETE CLIP - frees ~6-8 GB for UNet
        del clip_raw
        _vram_free()
        _vram_print("CLIP deleted")

        # ============================================================
        # STEP 2: Load UNet + apply LoRAs
        # ============================================================
        print("   [Clip] Loading UNet (GGUF Q4_K_M)...")
        _vram_guard(min_free_gb=5.0)
        unet_ld = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
        unet = get_value_at_index(unet_ld.load_unet(unet_name=unet_model), 0)

        if lora_stack is None:
            lora_stack = _build_lora_stack()
        unet = _apply_loras(unet, lora_stack)
        _vram_print("UNet + LoRAs loaded")

        # ============================================================
        # STEP 3: Load VAEs + upscaler
        # ============================================================
        vae_ld = NODE_CLASS_MAPPINGS["VAELoader"]()
        vae_v = get_value_at_index(vae_ld.load_vae(vae_name=vae_model), 0)
        vae_a = get_value_at_index(_audio_vae(vae_audio_model), 0)
        uml = NODE_CLASS_MAPPINGS["LatentUpscaleModelLoader"]()
        up_mdl = get_value_at_index(uml.EXECUTE_NORMALIZED(model_name=upscaler_model), 0)

        # ============================================================
        # STEP 4: Prepare latents
        # ============================================================
        # Half-resolution: EmptyImage -> ResizeImageMaskNode x0.5 -> GetImageSize
        ei = NODE_CLASS_MAPPINGS["EmptyImage"]()
        full_i = ei.generate(width=width, height=height, batch_size=1, color=0)
        rimn = NODE_CLASS_MAPPINGS["ResizeImageMaskNode"]()
        half_i = rimn.EXECUTE_NORMALIZED(
            input=get_value_at_index(full_i, 0), scale_method="area",
            resize_type={"resize_type": "scale by multiplier", "multiplier": 0.5})
        gis = NODE_CLASS_MAPPINGS["GetImageSize"]()
        hsz = gis.EXECUTE_NORMALIZED(image=get_value_at_index(half_i, 0))
        hw, hh = get_value_at_index(hsz, 0), get_value_at_index(hsz, 1)

        eltxv = NODE_CLASS_MAPPINGS["EmptyLTXVLatentVideo"]()
        vid_lat = eltxv.EXECUTE_NORMALIZED(
            width=hw, height=hh, length=frames, batch_size=1)

        # I2V: condition the latent on the seed/anchor image
        if not img_bypass:
            rim = NODE_CLASS_MAPPINGS["ResizeImageMaskNode"]()
            ref = get_value_at_index(rim.EXECUTE_NORMALIZED(
                input=image_tensor, scale_method="lanczos",
                resize_type={"resize_type": "scale dimensions",
                             "width": width, "height": height, "crop": "center"}), 0)
            if "ResizeImagesByLongerEdge" in NODE_CLASS_MAPPINGS:
                rle = NODE_CLASS_MAPPINGS["ResizeImagesByLongerEdge"]()
                ref = get_value_at_index(rle.EXECUTE_NORMALIZED(
                    longer_edge=max(width, height), images=ref), 0)
            pp = NODE_CLASS_MAPPINGS["LTXVPreprocess"]()
            pp_img = get_value_at_index(pp.EXECUTE_NORMALIZED(
                img_compression=33, image=ref), 0)
            i2v = NODE_CLASS_MAPPINGS["LTXVImgToVideoInplace"]()
            vid_lat = (get_value_at_index(i2v.EXECUTE_NORMALIZED(
                strength=img_str, bypass=False,
                vae=vae_v, image=pp_img,
                latent=get_value_at_index(vid_lat, 0)), 0),)
        else:
            vid_lat = (get_value_at_index(vid_lat, 0),)

        # Audio latent + concat AV
        elalat = NODE_CLASS_MAPPINGS["LTXVEmptyLatentAudio"]()
        aud_lat = elalat.EXECUTE_NORMALIZED(
            frames_number=frames, frame_rate=fps, batch_size=1, audio_vae=vae_a)
        catav = NODE_CLASS_MAPPINGS["LTXVConcatAVLatent"]()
        av_lat = get_value_at_index(catav.EXECUTE_NORMALIZED(
            video_latent=vid_lat[0],
            audio_latent=get_value_at_index(aud_lat, 0)), 0)

        # ============================================================
        # STEP 5: Pass 1 (ManualSigmas + euler + CFGGuider)
        # ============================================================
        print("   [Clip] Pass 1...")
        _vram_print()
        ms = NODE_CLASS_MAPPINGS["ManualSigmas"]()
        ks = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
        rn = NODE_CLASS_MAPPINGS["RandomNoise"]()
        cfg_node = NODE_CLASS_MAPPINGS["CFGGuider"]()
        sca = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()

        g1 = cfg_node.EXECUTE_NORMALIZED(
            cfg=pass1_cfg, model=unet,
            positive=get_value_at_index(cond, 0),
            negative=get_value_at_index(cond, 1))
        o1 = sca.EXECUTE_NORMALIZED(
            noise=get_value_at_index(rn.EXECUTE_NORMALIZED(noise_seed=seed), 0),
            guider=get_value_at_index(g1, 0),
            sampler=get_value_at_index(ks.EXECUTE_NORMALIZED(sampler_name=pass1_sampler), 0),
            sigmas=get_value_at_index(ms.EXECUTE_NORMALIZED(sigmas=pass1_sigmas), 0),
            latent_image=av_lat)
        p1_av = get_value_at_index(o1, 0)
        del g1
        _vram_free()
        print("   [Clip] Pass 1 done")

        # ============================================================
        # STEP 6: Pass 2 (LTXVLatentUpsampler 2x + gradient_estimation)
        # ============================================================
        print("   [Clip] Pass 2 (upscale + refine)...")
        sep = NODE_CLASS_MAPPINGS["LTXVSeparateAVLatent"]()
        s1 = sep.EXECUTE_NORMALIZED(av_latent=p1_av)
        vl1, al1 = get_value_at_index(s1, 0), get_value_at_index(s1, 1)

        crop = NODE_CLASS_MAPPINGS["LTXVCropGuides"]()
        cr = crop.EXECUTE_NORMALIZED(
            positive=get_value_at_index(cond, 0),
            negative=get_value_at_index(cond, 1),
            latent=vl1)
        g2 = cfg_node.EXECUTE_NORMALIZED(
            cfg=pass2_cfg, model=unet,
            positive=get_value_at_index(cr, 0),
            negative=get_value_at_index(cr, 1))

        ltxvup = NODE_CLASS_MAPPINGS["LTXVLatentUpsampler"]()
        up = ltxvup.upsample_latent(
            samples=get_value_at_index(cr, 2),
            upscale_model=up_mdl, vae=vae_v)
        del up_mdl
        _vram_free()

        av2 = get_value_at_index(catav.EXECUTE_NORMALIZED(
            video_latent=get_value_at_index(up, 0), audio_latent=al1), 0)

        o2 = sca.EXECUTE_NORMALIZED(
            noise=get_value_at_index(rn.EXECUTE_NORMALIZED(noise_seed=pass2_seed), 0),
            guider=get_value_at_index(g2, 0),
            sampler=get_value_at_index(
                ks.EXECUTE_NORMALIZED(sampler_name=pass2_sampler), 0),
            sigmas=get_value_at_index(ms.EXECUTE_NORMALIZED(sigmas=pass2_sigmas), 0),
            latent_image=av2)
        p2_den = get_value_at_index(o2, 1)

        # DELETE UNet - big VRAM release
        del g2, unet
        _vram_free()
        _vram_print("UNet deleted")
        print("   [Clip] Pass 2 done")

        # ============================================================
        # STEP 7: Decode video + audio
        # ============================================================
        s2 = sep.EXECUTE_NORMALIZED(av_latent=p2_den)
        vl_f = get_value_at_index(s2, 0)
        al_f = get_value_at_index(s2, 1)

        decoded = None
        if use_tiled_vae:
            try:
                td = NODE_CLASS_MAPPINGS["LTXVSpatioTemporalTiledVAEDecode"]()
                decoded = get_value_at_index(td.EXECUTE_NORMALIZED(
                    vae=vae_v, latents=vl_f,
                    spatial_tiles=tiled_stiles, spatial_overlap=tiled_soverlap,
                    temporal_tile_length=tiled_tlen, temporal_overlap=tiled_toverlap,
                    last_frame_fix=False, working_device="auto",
                    working_dtype="auto"), 0)
                print("   [Clip] Tiled VAE decode done")
            except (KeyError, Exception) as e:
                print(f"   [Clip] Tiled VAE skipped ({type(e).__name__}), standard decode")
                use_tiled_vae = False

        if not use_tiled_vae or decoded is None:
            vd = NODE_CLASS_MAPPINGS["VAEDecode"]()
            decoded = get_value_at_index(vd.decode(samples=vl_f, vae=vae_v), 0)

        # DELETE video VAE
        del vae_v
        _vram_free()

        aud_d = NODE_CLASS_MAPPINGS["LTXVAudioVAEDecode"]()
        audio = aud_d.EXECUTE_NORMALIZED(samples=al_f, audio_vae=vae_a)

        # DELETE audio VAE
        del vae_a
        _vram_free()

        # ============================================================
        # STEP 8: Save via CreateVideo
        # ============================================================
        cv_node = NODE_CLASS_MAPPINGS["CreateVideo"]()
        vid_obj = cv_node.EXECUTE_NORMALIZED(
            fps=fps, images=decoded,
            audio=get_value_at_index(audio, 0))
        path = save_video_obj(get_value_at_index(vid_obj, 0), prefix=output_prefix)
        _vram_print("after save")
        print(f"   [Clip] Saved: {path}")
        return path


print("generate_clip() ready.")



# ======================================================================
# CELL 8 - CONFIGURATION
# ======================================================================

# @title { "single-column": true }
# @markdown ## 8. Configuration
# @markdown All pipeline settings with Colab form annotations.

# @markdown ### Project Settings
PROJECT_NAME = "LTX_Studio_InfiniteFlow"  # @param {type:"string"}
WIDTH = 848  # @param {type:"integer"}
HEIGHT = 480  # @param {type:"integer"}
FPS = 24  # @param {type:"integer"}
FRAMES = 121  # @param {type:"integer"}

# @markdown ### AI Engine Settings
LLM_MODEL = "3B"  # @param ["3B", "8B", "14B"]
VISION_MODEL = "3B-fast"  # @param ["3B-fast", "7B-nsfw"]
CREATIVITY = 0.9  # @param {type:"number"}
USE_VISION = True  # @param {type:"boolean"}

# @markdown ### Continuity & Anchor Settings
OVERLAP_FRAMES = 16  # @param {type:"integer"}
ANCHOR_STRENGTH_HIGH = 0.85  # @param {type:"number"}
ANCHOR_STRENGTH_LOW = 0.70  # @param {type:"number"}
USE_ADAPTIVE_STRENGTH = True  # @param {type:"boolean"}

# @markdown ### LoRA Settings
IC_LORA = "detailer"  # @param ["none", "detailer", "canny", "depth", "pose"]
IC_LORA_STRENGTH = 0.4  # @param {type:"number"}
CAMERA_LORA_STRENGTH = 0.8  # @param {type:"number"}

# @markdown ### Memory / Tiled VAE
USE_TILED_VAE = True  # @param {type:"boolean"}
TILED_SPATIAL_TILES = 2  # @param {type:"integer"}
TILED_TEMPORAL_LEN = 48  # @param {type:"integer"}

# @markdown ### Character Settings
CHARACTER_IMAGE_PATH = ""  # @param {type:"string"}
CHARACTER_STRENGTH = 1.0  # @param {type:"number"}

# @markdown ### Sampling Parameters
PASS1_SIGMAS = "1., 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"  # @param {type:"string"}
PASS2_SIGMAS = "0.909375, 0.725, 0.421875, 0.0"  # @param {type:"string"}
PASS1_SAMPLER = "euler"  # @param {type:"string"}
PASS2_SAMPLER = "gradient_estimation"  # @param {type:"string"}

# @markdown ### Output Settings
OUTPUT_DIR = "/content/ComfyUI/output"  # @param {type:"string"}
SHOW_PREVIEWS = True  # @param {type:"boolean"}
DOWNLOAD_AFTER_GENERATE = False  # @param {type:"boolean"}

# @markdown ### Seed Settings
BASE_SEED = 42  # @param {type:"integer"}
AUTO_INCREMENT_SEED = True  # @param {type:"boolean"}


print("Configuration ready.")
print(f"   Project: {PROJECT_NAME}  |  {WIDTH}x{HEIGHT} @ {FPS}fps")
print(f"   Frames: {FRAMES}  |  LLM: {LLM_MODEL}  |  Vision: {VISION_MODEL}")
print(f"   IC LoRA: {IC_LORA} ({IC_LORA_STRENGTH})  |  Camera: {CAMERA_LORA_STRENGTH}")
print(f"   Tiled VAE: {USE_TILED_VAE}  |  Adaptive: {USE_ADAPTIVE_STRENGTH}")
print(f"   Seed: {BASE_SEED}  |  Auto-increment: {AUTO_INCREMENT_SEED}")



# ======================================================================
# CELL 9 - STORY INPUT & SCENE DEFINITION
# ======================================================================

# @title { "single-column": true }
# @markdown ## 9. Story Input & Scene Definition
# @markdown Enter your story and parse it into scene beats.
# @markdown The Whispering Cave 12-shot storyboard is included as an example.

# @markdown ### Story Input
STORY_INPUT = """Two children, Shiv and Vandana, discover a mysterious glowing cave
deep in an enchanted forest. They must overcome their fears and venture inside,
guided by a magical treasure map that pulses with golden light. The forest is
alive with bioluminescent plants and ancient trees. Their friendship and courage
are tested as whispers emerge from the darkness within."""  # @param {type:"string"}


def parse_story_to_scenes(story_text: str, prompt_engine: EasyPromptEngine,
                          bible: CharacterBible, num_scenes: int = 12) -> List[Dict]:
    """
    Parse a story description into individual scene beats using EasyPromptEngine.
    Each scene gets a camera movement, motion intensity, and expanded prompt
    with character bible lock and mandatory keywords.

    Args:
        story_text: Raw story description
        prompt_engine: EasyPromptEngine instance for prompt expansion
        bible: CharacterBible for consistency
        num_scenes: Target number of scenes to generate

    Returns:
        List of scene dicts with keys: prompt, camera_movement, motion_intensity, shot_data
    """
    # Default camera progression for narrative structure
    camera_sequence = [
        "dolly_forward", "tilt_up", "static", "zoom_in",
        "dolly_forward", "static", "dolly_forward", "zoom_in",
        "dolly_forward", "tilt_up", "dolly_forward", "static",
    ]

    # Default motion intensity curve (tension arc)
    motion_curve = [
        0.6, 0.4, 0.3, 0.5, 0.7, 0.2, 0.5, 0.8, 0.4, 0.5, 0.6, 0.3
    ]

    scenes = []
    story_lines = [l.strip() for l in story_text.strip().split("\n") if l.strip()]
    story_combined = " ".join(story_lines)

    # Simple scene splitting: divide story into roughly equal parts
    words = story_combined.split()
    words_per_scene = max(5, len(words) // num_scenes)

    scene_beats = []
    for i in range(num_scenes):
        start = i * words_per_scene
        end = start + words_per_scene if i < num_scenes - 1 else len(words)
        beat = " ".join(words[start:end])
        if beat.strip():
            scene_beats.append(beat)

    # If we got fewer beats than requested, pad with repetitions
    while len(scene_beats) < num_scenes:
        scene_beats.append(scene_beats[-1] if scene_beats else story_combined)

    for idx, beat in enumerate(scene_beats[:num_scenes]):
        cam_idx = idx % len(camera_sequence)
        motion_idx = idx % len(motion_curve)

        shot_data = {
            "action": beat,
            "camera_movement": camera_sequence[cam_idx],
            "motion_intensity": motion_curve[motion_idx],
            "character_focus": "both",
            "time": f"{idx*4}-{(idx+1)*4}s",
        }

        scenes.append({
            "prompt": beat,
            "beat": beat,
            "camera_movement": camera_sequence[cam_idx],
            "motion_intensity": motion_curve[motion_idx],
            "shot_data": shot_data,
            "index": idx,
        })

    return scenes


# -- Example Scene JSON: Whispering Cave 12-shot storyboard --

EXAMPLE_SCENE_JSON = {
    "scene_id": "scene_whispering_cave",
    "project_name": "Whispering_Cave",
    "duration_seconds": 48,
    "video_style": "3D Pixar cartoon style, cinematic animation, consistent character design",
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
            "desc": "12-year-old Indian boy",
            "detailed_appearance": {
                "face": "Round friendly face, large expressive brown eyes, small nose",
                "hair": "Messy jet-black hair with natural volume",
                "clothing": "Bright yellow t-shirt with orange trim, blue denim shorts, red sneakers",
                "build": "Slim athletic build, average height for age",
                "skin_tone": "Warm medium brown skin tone",
                "accessories": "Holding an old weathered treasure map"
            },
            "personality_traits": "Curious, slightly nervous, excited",
            "voice_characteristics": "Young boy voice, Hindi speaker"
        },
        {
            "name": "Vandana",
            "desc": "12-year-old Indian girl",
            "detailed_appearance": {
                "face": "Heart-shaped face, determined eyes, defined eyebrows",
                "hair": "Long black hair in high ponytail with red scrunchie",
                "clothing": "Denim dungarees over white t-shirt, pink backpack, brown boots",
                "build": "Athletic build, slightly taller than Shiv",
                "skin_tone": "Medium brown skin tone with golden undertones",
                "accessories": "Heavy brass flashlight in right hand"
            },
            "personality_traits": "Brave, protective, natural leader",
            "voice_characteristics": "Confident girl voice, Hindi speaker"
        }
    ],
    "story_action": {
        "shots": [
            {"time": "0-4s", "camera": "Wide tracking shot dolly forward", "camera_movement": "dolly_forward", "motion_intensity": 0.6, "action": "Shiv and Vandana walk deeper into forest. Sunlight fades to emerald glow.", "character_focus": "both", "emotion": "curious_cautious", "visual_effects": "Light transition, atmospheric particles"},
            {"time": "4-8s", "camera": "Close-up low angle on feet", "camera_movement": "tilt_up", "motion_intensity": 0.4, "action": "Close-up of Shiv red sneakers on glowing blue moss. Moss pulses with bioluminescence.", "character_focus": "Shiv_feet", "emotion": "wonder", "visual_effects": "Glowing moss reaction"},
            {"time": "8-12s", "camera": "Low-angle upward tilt", "camera_movement": "tilt_up", "motion_intensity": 0.3, "action": "Camera tilts up to reveal massive ancient trees. Vandana looks up in awe.", "character_focus": "Vandana", "emotion": "awe_mixed_fear", "visual_effects": "Dramatic shadows, creeping vines"},
            {"time": "12-16s", "camera": "Medium shot zoom in", "camera_movement": "zoom_in", "motion_intensity": 0.5, "action": "Vandana stops and touches a glowing vine. It reacts with rippling light.", "character_focus": "Vandana", "emotion": "curious_alert", "visual_effects": "Glowing vine interaction"},
            {"time": "16-20s", "camera": "POV handheld", "camera_movement": "static", "motion_intensity": 0.7, "action": "POV: Treasure map vibrates and glows. Lines pulse with golden light. Hands tremble.", "character_focus": "Shiv_hands", "emotion": "excited_nervous", "visual_effects": "Map glowing, magical symbols"},
            {"time": "20-24s", "camera": "Extreme close-up face", "camera_movement": "static", "motion_intensity": 0.2, "action": "Extreme close-up of Shiv wide eyes. He hears ghostly whisper. Hair blows in cold draft.", "character_focus": "Shiv_face", "emotion": "frightened_alert", "visual_effects": "Eye reflection, hair movement"},
            {"time": "24-28s", "camera": "Wide establishing through trees", "camera_movement": "dolly_forward", "motion_intensity": 0.5, "action": "Wide shot reveals dark cave mouth in limestone cliff. Mist pours out. Both frozen.", "character_focus": "both", "emotion": "ominous_discovery", "visual_effects": "Atmospheric mist, depth of field"},
            {"time": "28-32s", "camera": "Fast zoom to cave", "camera_movement": "zoom_in", "motion_intensity": 0.8, "action": "Rapid zoom toward cave. Darkness swirls inside. Blue-green lights flicker deep within.", "character_focus": "environment", "emotion": "threatening_mysterious", "visual_effects": "Swirling darkness, ethereal lights"},
            {"time": "32-36s", "camera": "Medium two-shot push in", "camera_movement": "dolly_forward", "motion_intensity": 0.4, "action": "Shiv grabs Vandana arm. Both stare at cave. Vandana expression wavers.", "character_focus": "both", "emotion": "fear_determination", "visual_effects": "Emotional expressions, tense atmosphere"},
            {"time": "36-40s", "camera": "Close-up tilt up to face", "camera_movement": "tilt_up", "motion_intensity": 0.5, "action": "Vandana reaches into backpack. Pulls out brass flashlight. Clicks it on, beam cuts mist.", "character_focus": "Vandana", "emotion": "brave_resolved", "visual_effects": "Flashlight beam, volumetric lighting"},
            {"time": "40-44s", "camera": "Hero shot low angle", "camera_movement": "dolly_forward", "motion_intensity": 0.6, "action": "Vandana steps toward cave with flashlight. Shiv gulps, nods, follows close behind.", "character_focus": "both", "emotion": "courageous_supportive", "visual_effects": "Heroic lighting, dust kicked up"},
            {"time": "44-48s", "camera": "Dramatic silhouette wide", "camera_movement": "static", "motion_intensity": 0.3, "action": "Final wide: Two children at cave mouth, silhouetted against forest light. Wind blows hair. They hold hands, look into abyss.", "character_focus": "both_silhouette", "emotion": "unity_facing_unknown", "visual_effects": "Perfect silhouette, rim lighting, wind effects"}
        ]
    },
    "dialogue_with_timing": [
        {"time": 6, "character": "Shiv", "dialogue": "Vandana... did you hear that? Someone was calling my name.", "emotion": "fearful_questioning", "voice_direction": "Whispered, trembling", "lip_sync_emphasis": "high"},
        {"time": 14, "character": "Vandana", "dialogue": "It is just the wind, Shiv. Do not be scared, I am here.", "emotion": "reassuring", "voice_direction": "Calm, confident", "lip_sync_emphasis": "high"},
        {"time": 25, "character": "The Cave", "dialogue": "*Breathing sound*... come inside...", "emotion": "eerie", "voice_direction": "Hollow echo", "lip_sync_emphasis": "none"},
        {"time": 33, "character": "Shiv", "dialogue": "The map is shaking! We are at the right place.", "emotion": "excited_scared", "voice_direction": "Voice rising", "lip_sync_emphasis": "high"},
        {"time": 41, "character": "Vandana", "dialogue": "Turn on your torch. There is no turning back now.", "emotion": "determined", "voice_direction": "Firm voice", "lip_sync_emphasis": "high"}
    ],
    "audio": {
        "background_music": "Low ambient humming building into tense mystical choir",
        "environment_sfx": "Dry leaves crunching, hollow whispers, flashlight clicking, eerie wind",
        "voice_processing": "Natural reverb for forest, slight echo near cave"
    },
    "motion_guidance": {
        "global_motion": "Steady forward progression toward cave, building tension",
        "character_motion": "Realistic walking pace, natural body language",
        "camera_motion": "Cinematic movements, smooth transitions"
    }
}


def build_storyboard_from_scene_json(scene_json: Dict) -> List[Dict]:
    """Convert a scene JSON (like EXAMPLE_SCENE_JSON) into a storyboard list
    compatible with InfiniteFlowEngine.run()."""
    storyboard = []
    shots = scene_json.get("story_action", {}).get("shots", [])

    for idx, shot in enumerate(shots):
        storyboard.append({
            "prompt": shot.get("action", ""),
            "beat": shot.get("action", ""),
            "camera_movement": shot.get("camera_movement", "static"),
            "motion_intensity": shot.get("motion_intensity", 0.5),
            "shot_data": shot,
            "index": idx,
        })

    return storyboard


print("Story Input & Scene Definition ready.")
print(f"   Example storyboard: {len(EXAMPLE_SCENE_JSON['story_action']['shots'])} shots")
print(f"   Story input: {len(STORY_INPUT.split())} words")



# ======================================================================
# CELL 10 - RUN PRODUCTION
# ======================================================================

# @title { "single-column": true }
# @markdown ## 10. Run Production
# @markdown Executes the full Infinite Flow Engine pipeline.
# @markdown Uses the example storyboard by default, or parses STORY_INPUT.

# -- Instantiate engines --
bible = CharacterBible()
vision_engine = VisionDescribeEngine(model_key=VISION_MODEL)
prompt_engine = EasyPromptEngine(model_size=LLM_MODEL, keep_loaded=False)

# -- Auto-populate CharacterBible from image if provided --
if CHARACTER_IMAGE_PATH and os.path.exists(CHARACTER_IMAGE_PATH):
    print("[Production] Loading character image for bible extraction...")
    char_img = load_image_tensor(CHARACTER_IMAGE_PATH)
    if char_img is not None:
        desc = vision_engine.describe(char_img)
        _vram_free()
        bible.extract_from_description("Main Character", desc)
        print(f"[Production] Bible populated: {desc[:150]}...")
        del char_img

# -- Build storyboard --
# Use example scene JSON by default; switch to parsed story if STORY_INPUT is custom
use_example = True
if STORY_INPUT.strip() and "whispering" not in STORY_INPUT.lower()[:50]:
    print("[Production] Parsing custom story into scenes...")
    storyboard = parse_story_to_scenes(
        STORY_INPUT, prompt_engine, bible, num_scenes=12)
    use_example = False
else:
    print("[Production] Using Whispering Cave example storyboard...")
    storyboard = build_storyboard_from_scene_json(EXAMPLE_SCENE_JSON)

    # Auto-populate bible from example scene characters
    if not bible.has_characters():
        for char in EXAMPLE_SCENE_JSON.get("main_characters", []):
            appearance = char.get("detailed_appearance", {})
            desc_parts = [
                f"{char['desc']}.",
                f"Face: {appearance.get('face', '')}.",
                f"Hair: {appearance.get('hair', '')}.",
                f"Clothing: {appearance.get('clothing', '')}.",
                f"Build: {appearance.get('build', '')}.",
                f"Skin: {appearance.get('skin_tone', '')}.",
            ]
            bible.extract_from_description(char["name"], " ".join(desc_parts))

# -- Build configuration dict --
production_config = {
    "PROJECT_NAME": PROJECT_NAME,
    "WIDTH": WIDTH,
    "HEIGHT": HEIGHT,
    "FPS": FPS,
    "FRAMES": FRAMES,
    "BASE_SEED": BASE_SEED,
    "AUTO_INCREMENT_SEED": AUTO_INCREMENT_SEED,
    "USE_VISION": USE_VISION,
    "USE_TILED_VAE": USE_TILED_VAE,
    "USE_ADAPTIVE_STRENGTH": USE_ADAPTIVE_STRENGTH,
    "ANCHOR_STRENGTH_HIGH": ANCHOR_STRENGTH_HIGH,
    "ANCHOR_STRENGTH_LOW": ANCHOR_STRENGTH_LOW,
    "OVERLAP_FRAMES": OVERLAP_FRAMES,
    "CREATIVITY": CREATIVITY,
    "IC_LORA": IC_LORA,
    "IC_LORA_STRENGTH": IC_LORA_STRENGTH,
    "CAMERA_LORA_STRENGTH": CAMERA_LORA_STRENGTH,
    "TILED_SPATIAL_TILES": TILED_SPATIAL_TILES,
    "TILED_TEMPORAL_LEN": TILED_TEMPORAL_LEN,
    "SHOW_PREVIEWS": SHOW_PREVIEWS,
    "OUTPUT_DIR": OUTPUT_DIR,
    "PASS1_SIGMAS": PASS1_SIGMAS,
    "PASS2_SIGMAS": PASS2_SIGMAS,
    "PASS1_SAMPLER": PASS1_SAMPLER,
    "PASS2_SAMPLER": PASS2_SAMPLER,
}

# -- Instantiate and run InfiniteFlowEngine --
engine = InfiniteFlowEngine(
    character_bible=bible,
    vision_engine=vision_engine,
    prompt_engine=prompt_engine,
    config=production_config,
)

try:
    seed_path = CHARACTER_IMAGE_PATH if CHARACTER_IMAGE_PATH else None
    clip_paths = engine.run(
        storyboard=storyboard,
        seed_image_path=seed_path,
        base_seed=BASE_SEED,
    )

    # -- Final concatenation --
    final_path = None
    if len(clip_paths) >= 2:
        final_name = f"{PROJECT_NAME}_final.mp4"
        final_path = engine.concat_all(final_name)
        if final_path and os.path.exists(final_path):
            print(f"\n[Production] Final video: {final_path}")
            if SHOW_PREVIEWS:
                display_video(final_path)
    elif clip_paths:
        final_path = clip_paths[0]
        print(f"\n[Production] Single clip: {final_path}")
        if SHOW_PREVIEWS:
            display_video(final_path)

    # -- Summary report --
    summary = engine.get_summary()
    print(f"\n{'=' * 60}")
    print("PRODUCTION SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total shots  : {summary['total_shots']}")
    print(f"  Successful   : {summary['successful']}")
    print(f"  Failed       : {summary['failed']}")
    if summary['failed_shots']:
        print(f"  Failed shots : {summary['failed_shots']}")
    print(f"  Output clips : {len(summary['clip_paths'])}")
    if final_path:
        print(f"  Final video  : {final_path}")
    print(f"{'=' * 60}")

    # -- Download if requested --
    if DOWNLOAD_AFTER_GENERATE and clip_paths:
        print("\nDownloading clips...")
        for p in clip_paths:
            if os.path.exists(p):
                files.download(p)
        if final_path and os.path.exists(final_path):
            files.download(final_path)

except KeyboardInterrupt:
    print("\nProduction interrupted. Cached clips are saved for resume.")

except torch.cuda.OutOfMemoryError:
    aggressive_cleanup("OOM")
    print("\nCUDA Out of Memory.")
    print("Suggested fixes:")
    print(f"  Current: {WIDTH}x{HEIGHT}, {FRAMES} frames")
    print("  - Switch LLM_MODEL = '3B' (saves ~6 GB during prompt phase)")
    print("  - Reduce FRAMES to 97 (saves ~1 GB of latent space)")
    print("  - Reduce WIDTH/HEIGHT: 704x448 is the next T4-safe step")
    print("  - Set USE_TILED_VAE = True if not already")
    print("  - Set USE_VISION = False (skips Qwen VL load entirely)")

except Exception as e:
    import traceback
    print(f"\nError: {type(e).__name__}: {e}")
    traceback.print_exc()
    print("\nTroubleshooting:")
    print("  UnetLoaderGGUF missing         -> Cell 1: ComfyUI_GGUF not cloned")
    print("  LTXVCropGuides missing          -> Cell 1: ComfyUI-LTXVideo not cloned")
    print("  LTXVSpatioTemporalTiledVAEDecode -> Cell 1: ComfyUI-LTXVideo not cloned")
    print("  DualCLIPLoader fp4 error        -> Use fp8: edit _M_CLIP1 in Cell 2")
    print("  qwen_vl_utils missing           -> Cell 1: pip install qwen-vl-utils")
    print("  Deformed output (Pass 1)        -> Change BASE_SEED and re-run")
    print("  Missing nodes                   -> Re-run Cell 1 (environment setup)")
