# -*- coding: utf-8 -*-
# ======================================================================
#  LTX STUDIO PIPELINE v1.0
#  Complete AI Video Production System - Google Colab Notebook
#
#  Combines: LTX_PRO + InfiniteFlow_12Scene + Infinite_Flow_PRO_v2
#  Features: Character Consistency, Scene Continuity, Motion Control,
#            Voice Sync, Adaptive Anchor, Multi-Character Bible
#
#  GPU Target: Google Colab T4 (15 GB VRAM)
#  Model: LTX-2 19B Q4_K_M GGUF (Kijai distilled)
#  Strategy: CLIP-first-then-UNet, aggressive cleanup between stages
# ======================================================================
#
# CELL ORDER:
#   Cell 1 - Environment Setup
#   Cell 2 - Model Downloads
#   Cell 3 - Core Imports & Utilities
#   Cell 4 - Character Bible & Consistency System
#   Cell 5 - Scene Schema & Storyboard Builder
#   Cell 6 - Motion, Camera & Voice Systems
#   Cell 7 - Production Engine (generate_clip + StudioProductionEngine)
#   Cell 8 - Configuration & Example Scene
#   Cell 9 - Run


# ======================================================================
# CELL 1 - ENVIRONMENT SETUP
# ======================================================================

# @title { "single-column": true }
# @markdown ## 1. Prepare Environment & Install Custom Nodes
# @markdown Clones ComfyUI + all required custom nodes.

# -- Base Python packages --
get_ipython().system('pip install torch torchvision torchaudio')

get_ipython().run_line_magic('cd', '/content')
from IPython.display import clear_output
clear_output()

get_ipython().system('pip install -q torchsde einops diffusers accelerate nest_asyncio')
get_ipython().system('pip install -q av spandrel albumentations onnx opencv-python onnxruntime')
get_ipython().system('pip install -q imageio imageio-ffmpeg')
get_ipython().system('pip install -q "transformers>=4.43.0" accelerate qwen-vl-utils huggingface_hub')

# -- ComfyUI (pinned branch) --
get_ipython().system('git clone --branch ComfyUI_22_01_2026_v0.10.0 https://github.com/Isi-dev/ComfyUI.git')
get_ipython().system('pip install -r /content/ComfyUI/requirements.txt -q')
clear_output()

# -- Custom nodes --
get_ipython().run_line_magic('cd', '/content/ComfyUI/custom_nodes')

get_ipython().system('git clone --branch kj_1.2.6 https://github.com/Isi-dev/ComfyUI_KJNodes')
get_ipython().system('git clone --branch ComfyUI_GGUF_22_01_2026 https://github.com/Isi-dev/ComfyUI_GGUF.git')
get_ipython().system('git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git')
get_ipython().system('git clone https://github.com/seanhan19911990-source/LTX2EasyPrompt-LD.git')
get_ipython().system('git clone https://github.com/seanhan19911990-source/LTX2-Master-Loader.git')
get_ipython().system('git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git')

# -- Install node requirements --
get_ipython().run_line_magic('cd', '/content/ComfyUI/custom_nodes/ComfyUI_KJNodes')
get_ipython().system('pip install -r requirements.txt -q')

get_ipython().run_line_magic('cd', '/content/ComfyUI/custom_nodes/ComfyUI_GGUF')
get_ipython().system('pip install -r requirements.txt -q')

get_ipython().run_line_magic('cd', '/content/ComfyUI/custom_nodes/ComfyUI-LTXVideo')
get_ipython().system('pip install -r requirements.txt -q 2>/dev/null || true')

get_ipython().run_line_magic('cd', '/content/ComfyUI/custom_nodes/LTX2EasyPrompt-LD')
get_ipython().system('pip install -r requirements.txt -q 2>/dev/null || true')

get_ipython().run_line_magic('cd', '/content/ComfyUI/custom_nodes/LTX2-Master-Loader')
get_ipython().system('pip install -r requirements.txt -q 2>/dev/null || true')

# -- System tools --
import subprocess
subprocess.run(["apt-get", "-y", "install", "-qq", "aria2", "ffmpeg"],
               check=True, capture_output=True)

# -- Final setup --
get_ipython().run_line_magic('cd', '/content/ComfyUI')
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


def model_download(url, dest_dir, filename=None, silent=True):
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


# -- Source URLs --
KIJAI = "https://huggingface.co/Kijai/LTXV2_comfy/resolve/main"
KIJAI23 = "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main"
COMFYORG = "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files"
LIGHTRIX = "https://huggingface.co/Lightricks"

print("-- Core model downloads --")

# UNet GGUF
_M_UNET = model_download(
    f"{KIJAI}/diffusion_models/ltx-2-19b-distilled_Q4_K_M.gguf",
    "/content/ComfyUI/models/unet")

# Text encoders
_M_CLIP1 = model_download(
    f"{COMFYORG}/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",
    "/content/ComfyUI/models/text_encoders")

_M_CLIP2 = model_download(
    f"{KIJAI}/text_encoders/ltx-2-19b-embeddings_connector_distill_bf16.safetensors",
    "/content/ComfyUI/models/text_encoders")

# VAEs
_M_VAE = model_download(
    f"{KIJAI}/VAE/LTX2_video_vae_bf16.safetensors",
    "/content/ComfyUI/models/vae")

_M_AVAE = model_download(
    f"{KIJAI}/VAE/LTX2_audio_vae_bf16.safetensors",
    "/content/ComfyUI/models/vae")

_M_TAE = model_download(
    f"{KIJAI23}/vae/taeltx2_3.safetensors",
    "/content/ComfyUI/models/vae")

# Spatial upscaler
_M_UP = model_download(
    f"{LIGHTRIX}/LTX-2/resolve/main/ltx-2-spatial-upscaler-x2-1.0.safetensors",
    "/content/ComfyUI/models/latent_upscale_models")

# LoRAs
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
_LORA_DIR = "/content/ComfyUI/models/loras"
os.makedirs(_LORA_DIR, exist_ok=True)
for name, url in _LORA_URLS.items():
    model_download(url, _LORA_DIR)

print("\nAll model files downloaded.")


# ======================================================================
# CELL 3 - CORE IMPORTS & UTILITIES
# ======================================================================

# @title { "single-column": true }
# @markdown ## 3. Imports, Helpers, Node Loader

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


# -- ComfyUI node output accessor --

def get_value_at_index(obj, index):
    """Extract output at index from ComfyUI node result."""
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


# -- Tensor / image helpers --

def pil_to_tensor(img):
    """PIL image to ComfyUI NHWC float tensor (1,H,W,3)."""
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def tensor_to_pil(t):
    """ComfyUI NHWC tensor to PIL Image."""
    if t.ndim == 4:
        t = t[0]
    return Image.fromarray((t.cpu().numpy() * 255).clip(0, 255).astype(np.uint8), "RGB")


def load_image_tensor(path):
    """Load image file as ComfyUI NHWC tensor. Returns None if missing."""
    if not path or not os.path.exists(path):
        return None
    return pil_to_tensor(Image.open(path).convert("RGB"))


def get_last_frame_tensor(video_path):
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

def display_video(path):
    """Display video inline in Colab."""
    if not path or not os.path.exists(path):
        print(f"   Video not found: {path}")
        return
    data = b64encode(open(path, "rb").read()).decode()
    display(HTML(
        '<video width=720 controls autoplay loop muted>'
        f'<source src="data:video/mp4;base64,{data}" type="video/mp4"></video>'))


def save_video_obj(video_obj, prefix="LTX_Studio"):
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


def concatenate_clips(clip_paths, output_path):
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

def _audio_vae(name):
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
# @markdown ## 4. Character Bible, VisionDescribe, EasyPrompt Engines
# @markdown Provides cross-scene character consistency and cinematic prompt expansion.


class CharacterBible:
    """
    Records named character attributes and serialises them as a prompt-injection
    block. The block is passed to EasyPromptEngine as character_bible so the LLM
    receives a hard [NON-NEGOTIABLE] constraint preventing attribute drift.
    """

    def __init__(self):
        self._chars = {}

    def add(self, name, **attributes):
        """Manually define a character with keyword attributes."""
        self._chars[name] = dict(attributes)

    def extract_from_description(self, name, description):
        """Store a raw VisionDescribe output under a character name."""
        self._chars[name] = {"_raw": description.strip()}

    def to_prompt_block(self):
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

    def has_characters(self):
        return bool(self._chars)

    def names(self):
        return list(self._chars.keys())

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self._chars, f, indent=2)
        print(f"   [Bible] Saved -> {path}")

    def load(self, path):
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

    def __init__(self, model_key="3B-fast", offline=False):
        self.model_key = model_key
        self.offline = offline

    def describe(self, image):
        """Describe an image. Loads model, runs inference, unloads."""
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


# -- Negative prompt builder --

_NEG_BASE = (
    "blurry, out of focus, low quality, worst quality, jpeg artifacts, "
    "static, no motion, frozen, duplicate, watermark, text, signature, "
    "poorly drawn, bad anatomy, deformed, disfigured, extra limbs, "
    "missing limbs, overexposed, underexposed, grainy, noise, flickering"
)


def _build_neg(result, user_input):
    """Build context-aware negative prompt."""
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
    if any(w in c for w in ["night", "dark", "moonlight", "dimly lit"]):
        extras.append("overexposed, bright daylight, blown highlights")
    elif any(w in c for w in ["daylight", "sunny", "golden hour", "bright"]):
        extras.append("underexposed, dark shadows, black crush")
    if any(w in c for w in ["two women", "two men", "two people", "couple"]):
        extras.append("merged bodies, fused figures, incorrect number of people")
    return ", ".join([_NEG_BASE] + extras)


class EasyPromptEngine:
    """
    Expands a simple story beat into a dense cinematic LTX-2 prompt.
    Loads the LLM, generates, cleans output, then unloads to free VRAM.
    """

    MODELS = {
        "8B": "mlabonne/NeuralDaredevil-8B-abliterated",
        "3B": "huihui-ai/Llama-3.2-3B-Instruct-abliterated",
        "14B": "huihui-ai/Huihui-Qwen3-14B-abliterated-v2",
    }

    SYSTEM_PROMPT = (
        "You are a cinematic prompt writer for LTX-2, an AI video generation model. "
        "Expand the user's idea into a rich, video-ready prompt.\n\n"
        "PRIORITY ORDER:\n"
        "1. Video style & genre\n"
        "2. Camera angle & shot type\n"
        "3. Character description (age MUST be a specific number)\n"
        "4. Scene & environment\n"
        "5. Action & motion (continuous present-tense)\n"
        "6. Camera movement (prose only, no brackets)\n"
        "7. Audio (max 2 ambient sounds, dialogue as inline prose)\n\n"
        "RULES:\n"
        "- Present tense throughout.\n"
        "- 8-12 sentences of dense flowing prose.\n"
        "- Fill the full token budget. Do not stop early.\n"
        "- Output ONLY the expanded prompt. No preamble."
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

    def __init__(self, model_size="8B", offline=False, keep_loaded=False):
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
        _vram_free()
        print("   [EasyPrompt] VRAM cleared.")

    def _stop_ids(self):
        delims = ["assistant", "user", "system", "<|eot_id|>", "<|end_of_turn|>",
                  "<|im_end|>", "<end_of_turn>", "[/INST]", "### Human", "### Assistant"]
        ids = [self._tok.eos_token_id]
        for s in delims:
            enc = self._tok.encode(s, add_special_tokens=False)
            if enc and enc[0] not in ids:
                ids.append(enc[0])
        return [i for i in dict.fromkeys(ids) if i is not None]

    @staticmethod
    def _clean(text):
        text = text.strip()
        for pattern, repl in EasyPromptEngine._CLEAN_RE:
            text = pattern.sub(repl, text)
        text = re.sub(r"\s*[\(\[]\s*$", "", text)
        return text.strip()

    def generate(self, user_input, frame_count=121, creativity=0.9,
                 seed=-1, scene_context="", lora_triggers="",
                 character_bible=""):
        """
        Returns (positive_prompt, negative_prompt).

        character_bible is injected as a hard [CHARACTER BIBLE - NON-NEGOTIABLE]
        block so the LLM cannot alter any locked attribute across clips.
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
        neg = _build_neg(result, user_input)

        if not self.keep_loaded:
            self._unload()

        print(f"   [EasyPrompt] Done ({len(result.split())} words).")
        return result, neg


print("Character Bible & Consistency System ready.")


# ======================================================================
# CELL 5 - SCENE SCHEMA & STORYBOARD BUILDER
# ======================================================================

# @title { "single-column": true }
# @markdown ## 5. Scene Schema & Storyboard Builder
# @markdown Comprehensive JSON scene schema and prompt construction from structured data.

# -- JSON Scene Schema Documentation --
# The scene JSON structure supports:
#   scene_id, project_name, duration_seconds, video_style
#   environment: location, time, weather, mood, lighting, color_palette
#   main_characters: name, desc, detailed_appearance (face, hair, clothing,
#       build, skin_tone, accessories), lora_path, personality_traits,
#       voice_characteristics
#   story_action.shots: time, camera, camera_movement, motion_intensity,
#       action, character_focus, emotion, visual_effects
#   dialogue_with_timing: time, character, dialogue, english_translation,
#       emotion, voice_direction, lip_sync_emphasis
#   audio: background_music, environment_sfx, voice_processing
#   motion_guidance: global_motion, character_motion, camera_motion


def build_character_prompt_detailed(character_data):
    """Build highly detailed character description for consistency."""
    char_name = character_data["name"]
    appearance = character_data["detailed_appearance"]

    prompt = f"{char_name}: "
    prompt += f"{appearance['face']}, "
    prompt += f"{appearance['hair']}, "
    prompt += f"wearing {appearance['clothing']}, "
    prompt += f"{appearance['build']}, {appearance['skin_tone']}, "
    prompt += f"{appearance['accessories']}. "

    # Repeat key features for emphasis
    prompt += (
        f"ALWAYS MAINTAIN: {char_name} has "
        f"{appearance['face'].split(',')[0]}, "
        f"{appearance['hair'].split(',')[0]}, "
        f"{appearance['clothing'].split(',')[0]}. "
    )
    return prompt


def get_character_consistency_prefix(scene_json):
    """Generate character consistency prefix for ALL prompts."""
    char_prompts = []
    for char in scene_json["main_characters"]:
        char_prompts.append(build_character_prompt_detailed(char))

    consistency_prompt = "CHARACTER CONSISTENCY CRITICAL: " + " | ".join(char_prompts)
    consistency_prompt += (
        " | MAINTAIN EXACT SAME CHARACTER APPEARANCE THROUGHOUT ENTIRE SCENE. "
        "NO MORPHING. NO STYLE CHANGES."
    )
    return consistency_prompt


def build_shot_prompt_pro(shot, json_data, shot_index, prev_shot_success=True):
    """
    PRO prompt builder combining character, action, camera, motion,
    environment, audio, VFX, and emotion into one dense prompt.
    """
    try:
        times = shot["time"].replace("s", "").split("-")
        start_s = int(times[0])
        end_s = int(times[1])
    except Exception:
        start_s, end_s = 0, 5

    # Character consistency (highest priority)
    character_prompt = get_character_consistency_prefix(json_data)

    # Shot action & camera
    action_prompt = f"SHOT {shot_index + 1}: {shot['action']}. "
    camera_prompt = f"CAMERA: {shot['camera']}. "

    # Motion guidance
    motion_intensity = shot.get("motion_intensity", 0.5)
    if motion_intensity < 0.3:
        motion_desc = "minimal motion, subtle movements, mostly static"
    elif motion_intensity < 0.6:
        motion_desc = "moderate motion, natural movements, steady pace"
    else:
        motion_desc = "dynamic motion, pronounced movements, energetic action"
    motion_prompt = f"MOTION: {motion_desc}. CAMERA MOVE: {shot.get('camera_movement', 'static')}. "

    # Environment
    env = json_data["environment"]
    env_prompt = (
        f"ENVIRONMENT: {env['location']}. LIGHTING: {env['lighting']}. "
        f"TIME: {env['time']}. WEATHER: {env['weather']}. "
        f"MOOD: {env['mood']}. COLORS: {env['color_palette']}. "
    )

    # Style
    style_prompt = f"STYLE: {json_data['video_style']}. "

    # Audio & dialogue
    audio_config = json_data.get("audio", {})
    audio_prompt = f"AUDIO: {audio_config.get('background_music', '')}. "
    dialogue_lines = []
    for entry in json_data.get("dialogue_with_timing", []):
        if start_s <= entry.get("time", -1) < end_s:
            char = entry["character"]
            text = entry.get("dialogue", "")
            emotion = entry.get("emotion", "neutral")
            lip_sync = entry.get("lip_sync_emphasis", "medium")
            if lip_sync == "high":
                dialogue_lines.append(
                    f"LIP SYNC: {char} speaks '{text}' with {emotion} emotion. "
                    f"Mouth movements MUST match dialogue.")
            else:
                dialogue_lines.append(f"{char} says '{text}' ({emotion}).")
    if dialogue_lines:
        audio_prompt += " ".join(dialogue_lines) + " "

    # VFX & emotion
    vfx_prompt = f"VFX: {shot.get('visual_effects', 'natural')}. "
    emotion_prompt = f"EMOTION: {shot.get('emotion', 'neutral')}. FOCUS: {shot.get('character_focus', 'scene')}. "

    # Assemble in priority order
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


def parse_storyboard_from_json(scene_json):
    """Convert JSON scene schema into an ordered STORYBOARD list."""
    storyboard = []
    shots = scene_json.get("story_action", {}).get("shots", [])

    for idx, shot in enumerate(shots):
        prev_shot = shots[idx - 1] if idx > 0 else None
        full_prompt = build_shot_prompt_pro(shot, scene_json, idx)

        storyboard.append({
            "id": f"shot_{idx + 1:02d}",
            "prompt": full_prompt,
            "shot_data": shot,
            "prev_shot": prev_shot,
            "index": idx,
        })

    return storyboard


print("Scene Schema & Storyboard Builder ready.")


# ======================================================================
# CELL 6 - MOTION, CAMERA & VOICE SYSTEMS
# ======================================================================

# @title { "single-column": true }
# @markdown ## 6. Motion, Camera & Voice Systems
# @markdown Camera LoRA mapping, motion guidance, voice sync, adaptive strength,
# @markdown enhanced anchor extraction.

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


def get_motion_guidance_prompt(shot):
    """Build motion-specific guidance prompt with intensity descriptors."""
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

    for key in CAMERA_LORA_MAPPING:
        if key in camera_movement:
            return key, CAMERA_LORA_MAPPING[key]

    # Default to static if no match
    if "static" in CAMERA_LORA_MAPPING:
        return "static", CAMERA_LORA_MAPPING["static"]
    return None, None


def get_dialogue_for_shot_enhanced(start_time, end_time, dialogue_list):
    """Enhanced dialogue injection with lip sync emphasis support."""
    lines = []
    for entry in dialogue_list:
        if start_time <= entry.get("time", -1) < end_time:
            char = entry["character"]
            text = entry.get("dialogue", "")
            emotion = entry.get("emotion", "neutral")
            voice_direction = entry.get("voice_direction", "")
            lip_sync = entry.get("lip_sync_emphasis", "medium")

            if lip_sync == "high":
                lines.append(
                    f"LIP SYNC CRITICAL: {char} speaks '{text}' with {emotion} emotion. "
                    f"{voice_direction}. Mouth movements MUST match dialogue exactly.")
            elif lip_sync == "medium":
                lines.append(
                    f"{char} says '{text}' with {emotion} emotion. {voice_direction}.")
            else:
                lines.append(f"AUDIO EFFECT: {text}")

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


def calculate_adaptive_strength(shot, prev_shot, prev_shot_success,
                                anchor_high=0.85, anchor_low=0.70):
    """
    Calculate anchor strength based on:
    - Motion intensity change between shots
    - Character focus change
    - Previous shot success rate
    """
    strength = anchor_high

    if prev_shot:
        # Adjust based on motion intensity change
        motion_change = abs(
            shot.get("motion_intensity", 0.5) -
            prev_shot.get("motion_intensity", 0.5)
        )
        if motion_change > 0.4:
            strength -= 0.10
        elif motion_change < 0.2:
            strength += 0.05

        # Adjust based on character focus change
        if shot.get("character_focus") != prev_shot.get("character_focus"):
            strength -= 0.05

    # Adjust based on previous shot success
    if not prev_shot_success:
        strength -= 0.10

    # Clamp to reasonable range
    strength = max(anchor_low, min(anchor_high, strength))
    return strength


def enhanced_anchor_extraction(video_path, output_folder="/content/ComfyUI/input",
                               scene_idx=0, overlap=16):
    """
    Enhanced anchor extraction with:
    - Multi-frame candidates
    - Brightness validation
    - Sharpness scoring (Laplacian variance)
    """
    if not os.path.exists(video_path):
        print(f"   Video not found: {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Try multiple candidate frames near the overlap point
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

        # Score frame quality
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(cv2.mean(gray)[0])

        # Reject if too dark
        if brightness < 5:
            continue

        # Sharpness via Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(laplacian.var())

        # Combined score
        score = brightness + sharpness * 0.1

        if score > best_score:
            best_score = score
            best_frame = frame

    cap.release()

    if best_frame is not None:
        os.makedirs(output_folder, exist_ok=True)
        filename = f"anchor_scene_{scene_idx}.png"
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, best_frame)
        print(f"   Anchor extracted (quality score: {best_score:.2f})")
        return save_path
    else:
        print(f"   No valid anchor frame found")
        return None


def build_negative_prompt_enhanced():
    """Build comprehensive negative prompt covering character, motion, audio issues."""
    char_neg = (
        "character morphing, face changing, inconsistent character design, "
        "different clothing in same scene, style shift, character replacement, "
    )
    motion_neg = (
        "motion blur artifacts, jittery movement, unnatural animation, "
        "robotic motion, floating characters, "
    )
    audio_neg = (
        "desynchronized lips, mouth not moving during speech, "
        "frozen face during dialogue, mismatched audio, "
    )
    quality_neg = (
        "compression artifacts, pixelation, banding, color shifts, "
        "lighting inconsistency, flickering, "
    )
    base_neg = (
        "blurry, distorted, low quality, bad anatomy, text, watermark, "
        "ugly, deformed, glitch, morphing artifacts, extra limbs"
    )
    return char_neg + motion_neg + audio_neg + quality_neg + base_neg


print("Motion, Camera & Voice Systems ready.")


# ======================================================================
# CELL 7 - PRODUCTION ENGINE
# ======================================================================

# @title { "single-column": true }
# @markdown ## 7. Production Engine (generate_clip + StudioProductionEngine)
# @markdown Two-pass LTX-2 19B GGUF pipeline and full production orchestrator.

# -- LoRA stack builder --

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


# -- generate_clip: Two-pass LTX-2 19B GGUF pipeline --

def generate_clip(
    image_tensor,
    prompt,
    neg_prompt,
    width=848,
    height=480,
    frames=121,
    fps=24,
    seed=42,
    image_strength=1.0,
    pass1_sigmas="1., 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0",
    pass1_sampler="euler",
    pass1_cfg=1.0,
    pass2_sigmas="0.909375, 0.725, 0.421875, 0.0",
    pass2_sampler="gradient_estimation",
    pass2_cfg=1.0,
    pass2_seed=0,
    use_tiled_vae=True,
    tiled_stiles=2,
    tiled_soverlap=8,
    tiled_tlen=48,
    tiled_toverlap=4,
    lora_stack=None,
    output_prefix="LTX_Studio",
    unet_model="ltx-2-19b-distilled_Q4_K_M.gguf",
    clip_name1="gemma_3_12B_it_fp4_mixed.safetensors",
    clip_name2="ltx-2-19b-embeddings_connector_distill_bf16.safetensors",
    vae_model="LTX2_video_vae_bf16.safetensors",
    vae_audio_model="LTX2_audio_vae_bf16.safetensors",
    upscaler_model="ltx-2-spatial-upscaler-x2-1.0.safetensors",
):
    """
    Two-pass LTX-2 19B GGUF generation for one clip.
    T4-safe: CLIP is loaded, encodes text, then deleted before UNet loads.
    UNet is deleted after Pass 2, before decode.

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
            rim2 = NODE_CLASS_MAPPINGS["ResizeImageMaskNode"]()
            ri2 = rim2.EXECUTE_NORMALIZED(
                input=image_tensor, scale_method="lanczos",
                resize_type={"resize_type": "scale dimensions",
                             "width": hw * 2, "height": hh * 2, "crop": "center"})
            ppn = NODE_CLASS_MAPPINGS["LTXVPreprocess"]()
            pp = get_value_at_index(ppn.EXECUTE_NORMALIZED(
                img_compression=33, image=get_value_at_index(ri2, 0)), 0)
            i2v = NODE_CLASS_MAPPINGS["LTXVImgToVideoInplace"]()
            vid_lat = (get_value_at_index(i2v.EXECUTE_NORMALIZED(
                strength=img_str, bypass=False,
                vae=vae_v, image=pp,
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
            sampler=get_value_at_index(ks.EXECUTE_NORMALIZED(sampler_name=pass2_sampler), 0),
            sigmas=get_value_at_index(ms.EXECUTE_NORMALIZED(sigmas=pass2_sigmas), 0),
            latent_image=av2)
        p2_den = get_value_at_index(o2, 1)

        # Delete UNet - big VRAM release
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

        del vae_v
        _vram_free()

        aud_d = NODE_CLASS_MAPPINGS["LTXVAudioVAEDecode"]()
        audio = aud_d.EXECUTE_NORMALIZED(samples=al_f, audio_vae=vae_a)
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


# -- StudioProductionEngine: Full production orchestrator --

class StudioProductionEngine:
    """
    Orchestrates the full LTX Studio production pipeline.
    Iterates through shots from a scene JSON, using VisionDescribeEngine for
    anchor analysis, EasyPromptEngine for prompt expansion with CharacterBible lock,
    generate_clip() for rendering, and last-frame chaining between shots.

    Features:
    - Adaptive anchor strength per shot
    - Retry logic (max 3 attempts with seed variation)
    - Cache-based resume (saves each clip to disk)
    - Progress tracking with time estimates
    - Final video stitching via ffmpeg
    """

    def __init__(self, character_bible=None, vision_engine=None,
                 prompt_engine=None, scene_json=None, config=None):
        self.bible = character_bible or CharacterBible()
        self.vision = vision_engine or VisionDescribeEngine("3B-fast")
        self.prompt_engine = prompt_engine or EasyPromptEngine("8B")
        self.scene_json = scene_json or {}
        self.config = config or {}

        # Configuration with defaults
        self.width = self.config.get("WIDTH", 848)
        self.height = self.config.get("HEIGHT", 480)
        self.fps = self.config.get("FPS", 24)
        self.frames = self.config.get("FRAMES", 121)
        self.base_seed = self.config.get("BASE_SEED", 42)
        self.use_vision = self.config.get("USE_VISION", True)
        self.use_tiled_vae = self.config.get("USE_TILED_VAE", True)
        self.use_adaptive_strength = self.config.get("USE_ADAPTIVE_STRENGTH", True)
        self.use_motion_loras = self.config.get("USE_MOTION_LORAS", True)
        self.use_voice_sync = self.config.get("USE_VOICE_SYNC", True)
        self.anchor_high = self.config.get("ANCHOR_STRENGTH_HIGH", 0.85)
        self.anchor_low = self.config.get("ANCHOR_STRENGTH_LOW", 0.70)
        self.creativity = self.config.get("CREATIVITY", 0.9)
        self.seed_image_path = self.config.get("SEED_IMAGE_PATH", None)
        self.project_name = self.config.get("PROJECT_NAME", "LTX_Studio")
        self.output_dir = self.config.get("OUTPUT_DIR", "/content/ComfyUI/output")
        self.show_previews = self.config.get("SHOW_PREVIEWS", True)

        # Internal state
        self._clip_paths = []
        self._failed_shots = []

    def run(self):
        """
        Main production loop. Iterates through all shots in the storyboard,
        generating clips with retry logic and scene chaining.
        Returns list of output paths (None for failed shots).
        """
        storyboard = parse_storyboard_from_json(self.scene_json)
        if not storyboard:
            print("No shots found in scene JSON.")
            return []

        cache_dir = f"{self.output_dir}/{self.project_name}_cache"
        os.makedirs(cache_dir, exist_ok=True)
        input_dir = "/content/ComfyUI/input"
        os.makedirs(input_dir, exist_ok=True)

        outputs = []
        current_image = None
        prev_shot_success = True

        # Load seed image if provided
        if self.seed_image_path and os.path.exists(self.seed_image_path):
            current_image = load_image_tensor(self.seed_image_path)
            print(f"[Studio] Seed image loaded: {self.seed_image_path}")

            # Auto-populate CharacterBible from seed image
            if not self.bible.has_characters() and self.use_vision:
                print("[Studio] Extracting character from seed image...")
                desc = self.vision.describe(current_image)
                _vram_free()
                self.bible.extract_from_description("Main Character", desc)

        # Cache-based resume: check for existing clips
        start_index = 0
        for i in range(len(storyboard)):
            cached_clip = f"{cache_dir}/scene_{i:02d}.mp4"
            if os.path.exists(cached_clip):
                outputs.append(cached_clip)
                start_index = i + 1
                lf = get_last_frame_tensor(cached_clip)
                if lf is not None:
                    current_image = lf
            else:
                break

        if start_index > 0:
            print(f"[Studio] Resuming from shot {start_index + 1} ({start_index} cached)")

        print(f"\n[Studio] === PRODUCTION START ===")
        print(f"[Studio] Shots: {len(storyboard)}  |  Size: {self.width}x{self.height}")
        print(f"[Studio] FPS: {self.fps}  |  Frames/shot: {self.frames}")
        print(f"[Studio] Bible: {self.bible.names() or 'empty'}")
        print(f"[Studio] Mode: {'I2V' if current_image is not None else 'T2V'}\n")

        production_start = time.time()

        for i in range(start_index, len(storyboard)):
            entry = storyboard[i]
            shot_data = entry["shot_data"]
            shot_num = i + 1

            print(f"\n[Studio] --- Shot {shot_num}/{len(storyboard)} ---")
            print(f"[Studio] {shot_data.get('action', '')[:80]}...")

            # Calculate adaptive strength
            if self.use_adaptive_strength and i > 0:
                strength = calculate_adaptive_strength(
                    shot_data, entry.get("prev_shot"),
                    prev_shot_success,
                    self.anchor_high, self.anchor_low)
            else:
                strength = self.anchor_high if current_image is not None else 0.0

            # Vision describe on anchor image (from beat 2 onward)
            scene_ctx = ""
            if i > 0 and current_image is not None and self.use_vision:
                print(f"   [Studio] Vision Describe on anchor...")
                try:
                    scene_ctx = self.vision.describe(current_image)
                    _vram_free()
                except Exception as e:
                    print(f"   [Studio] Vision failed ({e}), continuing without context")

            # EasyPrompt expand with CharacterBible lock
            print(f"   [Studio] EasyPrompt expand...")
            beat_seed = self.base_seed + i * 1000
            try:
                expanded_prompt, neg_prompt = self.prompt_engine.generate(
                    user_input=shot_data.get("action", ""),
                    frame_count=self.frames,
                    creativity=self.creativity,
                    seed=beat_seed,
                    scene_context=scene_ctx,
                    character_bible=self.bible.to_prompt_block(),
                )
                _vram_free()
            except Exception as e:
                print(f"   [Studio] EasyPrompt failed ({e}), using raw prompt")
                expanded_prompt = entry["prompt"]
                neg_prompt = build_negative_prompt_enhanced()

            print(f"   [Studio] Prompt: {expanded_prompt[:120]}...")

            # Determine camera LoRA
            camera_lora_file = None
            if self.use_motion_loras:
                _, camera_lora_file = get_camera_lora_for_shot(shot_data)

            lora_stack = _build_lora_stack(
                ic_lora="detailer", ic_strength=0.4,
                camera_lora_file=camera_lora_file,
                camera_strength=0.8)

            # Retry loop (max 3 attempts with seed variation)
            max_retries = 3
            success = False
            clip_path = None
            current_seed = beat_seed

            for attempt in range(max_retries):
                try:
                    print(f"   [Studio] Generating (attempt {attempt + 1}, seed={current_seed})...")
                    clip_path = generate_clip(
                        image_tensor=current_image,
                        prompt=expanded_prompt,
                        neg_prompt=neg_prompt,
                        width=self.width,
                        height=self.height,
                        frames=self.frames,
                        fps=self.fps,
                        seed=current_seed,
                        image_strength=strength,
                        use_tiled_vae=self.use_tiled_vae,
                        lora_stack=lora_stack,
                        output_prefix=f"{self.project_name}_{shot_num:02d}",
                    )
                    if clip_path and os.path.exists(clip_path):
                        success = True
                        break
                except torch.cuda.OutOfMemoryError:
                    aggressive_cleanup("OOM recovery")
                    current_seed += 1
                    print(f"   [Studio] OOM, retrying with seed {current_seed}...")
                except Exception as e:
                    print(f"   [Studio] Error: {type(e).__name__}: {e}")
                    current_seed += 1

                if attempt < max_retries - 1:
                    current_seed += 1

            if success and clip_path:
                # Cache the clip
                cached_path = f"{cache_dir}/scene_{i:02d}.mp4"
                shutil.copy(clip_path, cached_path)
                outputs.append(cached_path)
                prev_shot_success = True

                # Enhanced anchor extraction for next shot
                anchor_path = enhanced_anchor_extraction(
                    cached_path, output_folder=input_dir,
                    scene_idx=i, overlap=16)
                if anchor_path:
                    current_image = load_image_tensor(anchor_path)
                else:
                    lf = get_last_frame_tensor(cached_path)
                    current_image = lf

                print(f"   [Studio] Shot {shot_num} DONE")

                if self.show_previews:
                    display_video(cached_path)
            else:
                outputs.append(None)
                self._failed_shots.append(shot_num)
                prev_shot_success = False
                current_image = None
                print(f"   [Studio] Shot {shot_num} FAILED after {max_retries} attempts")

            # Progress tracking
            elapsed = time.time() - production_start
            completed = i - start_index + 1
            avg_per_shot = elapsed / completed if completed > 0 else 0
            remaining = avg_per_shot * (len(storyboard) - i - 1)
            print(f"   [Studio] Elapsed: {elapsed/60:.1f}min | Est. remaining: {remaining/60:.1f}min")

        # Store clip paths
        self._clip_paths = [p for p in outputs if p]

        # Final concatenation
        final_path = None
        if len(self._clip_paths) >= 2:
            final_path = f"{self.output_dir}/{self.project_name}_final.mp4"
            concatenate_clips(self._clip_paths, final_path)
            if final_path and os.path.exists(final_path):
                print(f"\n[Studio] Final video: {final_path}")
                if self.show_previews:
                    display_video(final_path)

        # Summary report
        total_elapsed = time.time() - production_start
        print(f"\n[Studio] === PRODUCTION COMPLETE ===")
        print(f"[Studio] Total shots: {len(storyboard)}")
        print(f"[Studio] Successful: {len(self._clip_paths)}")
        print(f"[Studio] Failed: {len(self._failed_shots)}")
        print(f"[Studio] Duration: {total_elapsed/60:.1f} minutes")
        if final_path:
            print(f"[Studio] Output: {final_path}")

        return outputs

    def get_clip_paths(self):
        """Return list of successfully generated clip paths."""
        return self._clip_paths

    def get_summary(self):
        """Return production summary as dict."""
        return {
            "total_shots": len(parse_storyboard_from_json(self.scene_json)),
            "successful": len(self._clip_paths),
            "failed": len(self._failed_shots),
            "failed_shots": self._failed_shots,
            "clip_paths": self._clip_paths,
        }


print("Production Engine ready (generate_clip + StudioProductionEngine).")



# ======================================================================
# CELL 8 - CONFIGURATION & EXAMPLE SCENE
# ======================================================================

# @title { "single-column": true }
# @markdown ## 8. Configuration & Example Scene
# @markdown All parameters and the Whispering Cave 12-shot storyboard.

# -- Project Configuration --
PROJECT_NAME = "Whispering_Cave"
WIDTH = 848
HEIGHT = 480
FPS = 24
FRAMES = 121

# -- Engine Settings --
LLM_MODEL = "8B"
VISION_MODEL = "3B-fast"
CREATIVITY = 0.9
USE_VISION = True
USE_TILED_VAE = True
SEED_IMAGE_PATH = None
BASE_SEED = 42

# -- Continuity Settings --
OVERLAP_FRAMES = 16
ANCHOR_STRENGTH_HIGH = 0.85
ANCHOR_STRENGTH_LOW = 0.70
USE_ADAPTIVE_STRENGTH = True

# -- Feature Toggles --
USE_CHARACTER_LORAS = True
USE_MOTION_LORAS = True
USE_VOICE_SYNC = True
GENERATE_SUBTITLES = True
SHOW_PREVIEWS = True


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
            "lora_path": None,
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
            "lora_path": None,
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
        {"time": 6, "character": "Shiv", "dialogue": "Vandana... did you hear that? Someone was calling my name.", "english_translation": "Vandana... did you hear that?", "emotion": "fearful_questioning", "voice_direction": "Whispered, trembling", "lip_sync_emphasis": "high"},
        {"time": 14, "character": "Vandana", "dialogue": "It is just the wind, Shiv. Do not be scared, I am here.", "english_translation": "It is just the wind.", "emotion": "reassuring", "voice_direction": "Calm, confident", "lip_sync_emphasis": "high"},
        {"time": 25, "character": "The Cave", "dialogue": "*Breathing sound*... come inside...", "english_translation": "come inside", "emotion": "eerie", "voice_direction": "Hollow echo", "lip_sync_emphasis": "none"},
        {"time": 33, "character": "Shiv", "dialogue": "The map is shaking! We are at the right place.", "english_translation": "The map is shaking!", "emotion": "excited_scared", "voice_direction": "Voice rising", "lip_sync_emphasis": "high"},
        {"time": 41, "character": "Vandana", "dialogue": "Turn on your torch. There is no turning back now.", "english_translation": "No turning back.", "emotion": "determined", "voice_direction": "Firm voice", "lip_sync_emphasis": "high"}
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

print("Configuration & Example Scene ready.")
print(f"   Project: {PROJECT_NAME}  |  {WIDTH}x{HEIGHT} @ {FPS}fps")
print(f"   Frames: {FRAMES}  |  LLM: {LLM_MODEL}  |  Vision: {VISION_MODEL}")
print(f"   Example storyboard: {len(EXAMPLE_SCENE_JSON["story_action"]["shots"])} shots")


# ======================================================================
# CELL 9 - RUN
# ======================================================================

# @title { "single-column": true }
# @markdown ## 9. Run Production
# @markdown Executes the full pipeline on the configured scene.

# -- Instantiate engines --
bible = CharacterBible()

vision_engine = VisionDescribeEngine(model_key=VISION_MODEL)
prompt_engine = EasyPromptEngine(model_size=LLM_MODEL, keep_loaded=False)

# -- Build configuration dict --
production_config = {
    "PROJECT_NAME": PROJECT_NAME,
    "WIDTH": WIDTH,
    "HEIGHT": HEIGHT,
    "FPS": FPS,
    "FRAMES": FRAMES,
    "BASE_SEED": BASE_SEED,
    "USE_VISION": USE_VISION,
    "USE_TILED_VAE": USE_TILED_VAE,
    "USE_ADAPTIVE_STRENGTH": USE_ADAPTIVE_STRENGTH,
    "USE_MOTION_LORAS": USE_MOTION_LORAS,
    "USE_VOICE_SYNC": USE_VOICE_SYNC,
    "ANCHOR_STRENGTH_HIGH": ANCHOR_STRENGTH_HIGH,
    "ANCHOR_STRENGTH_LOW": ANCHOR_STRENGTH_LOW,
    "CREATIVITY": CREATIVITY,
    "SEED_IMAGE_PATH": SEED_IMAGE_PATH,
    "SHOW_PREVIEWS": SHOW_PREVIEWS,
    "OUTPUT_DIR": "/content/ComfyUI/output",
}

# -- Instantiate production engine --
engine = StudioProductionEngine(
    character_bible=bible,
    vision_engine=vision_engine,
    prompt_engine=prompt_engine,
    scene_json=EXAMPLE_SCENE_JSON,
    config=production_config,
)

# -- Execute production loop --
try:
    outputs = engine.run()

    # -- Summary --
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
    print(f"{'=' * 60}")

    # Display final video if available
    clip_paths = engine.get_clip_paths()
    if len(clip_paths) >= 2:
        final_path = f"/content/ComfyUI/output/{PROJECT_NAME}_final.mp4"
        if os.path.exists(final_path):
            print(f"\n  Final video: {final_path}")
            display_video(final_path)
    elif clip_paths:
        print(f"\n  Single clip: {clip_paths[0]}")
        display_video(clip_paths[0])

except KeyboardInterrupt:
    print("\nProduction interrupted. Cached clips are saved for resume.")

except torch.cuda.OutOfMemoryError:
    aggressive_cleanup("OOM")
    print("\nCUDA Out of Memory.")
    print(f"  Current: {WIDTH}x{HEIGHT}, {FRAMES} frames")
    print("  Try: WIDTH=768, HEIGHT=512, FRAMES=97 for T4")

except Exception as e:
    import traceback
    print(f"\nError: {type(e).__name__}: {e}")
    traceback.print_exc()
