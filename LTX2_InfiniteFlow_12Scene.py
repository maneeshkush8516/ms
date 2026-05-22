# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  LTX-2 LD  —  INFINITE FLOW ENGINE  (12-Scene Edition)                 ║
# ║                                                                          ║
# ║  SECTION A  VisionDescribeEngine   (standalone Qwen2.5-VL, no wrapper)  ║
# ║  SECTION B  EasyPromptEngine       (full cinematic LLM expansion)        ║
# ║  SECTION C  CharacterBible         (cross-scene consistency lock)        ║
# ║  SECTION D  generate_clip()        (two-pass LTX-2 19B GGUF pipeline)   ║
# ║  SECTION E  InfiniteFlowEngine     (12-beat orchestrator + chaining)     ║
# ║                                                                          ║
# ║  GPU target : Google Colab T4 (15 GB)                                   ║
# ║  Model      : LTX-2 19B Q4_K_M GGUF (Kijai distilled)                  ║
# ║  Strategy   : load → use → unload every heavyweight model in sequence   ║
# ╚══════════════════════════════════════════════════════════════════════════╝


# ══════════════════════════════════════════════════════════════════════════
# CELL 1  —  ENVIRONMENT SETUP  (run once per Colab session)
# ══════════════════════════════════════════════════════════════════════════
# @title { "single-column": true }
# @markdown ## 💥 1. Install Environment

!pip install torch torchvision torchaudio
%cd /content
from IPython.display import clear_output
clear_output()

!pip install -q torchsde einops diffusers accelerate nest_asyncio
!pip install -q av spandrel albumentations onnx opencv-python onnxruntime
!pip install -q imageio imageio-ffmpeg
!pip install -q "transformers>=4.43.0" accelerate qwen-vl-utils huggingface_hub

# Pinned ComfyUI build (matches reference notebook)
!git clone --branch ComfyUI_22_01_2026_v0.10.0 https://github.com/Isi-dev/ComfyUI.git
!pip install -r /content/ComfyUI/requirements.txt -q
clear_output()

%cd /content/ComfyUI/custom_nodes

# Pinned custom nodes
!git clone --branch kj_1.2.6               https://github.com/Isi-dev/ComfyUI_KJNodes
!git clone --branch ComfyUI_GGUF_22_01_2026 https://github.com/Isi-dev/ComfyUI_GGUF.git

# Official LTX nodes (tiled VAE + AV latent helpers)
!git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git

# LoRa Daddy nodes (Master Loader + Easy Prompt as ComfyUI nodes if needed)
!git clone https://github.com/seanhan19911990-source/LTX2EasyPrompt-LD.git
!git clone https://github.com/seanhan19911990-source/LTX2-Master-Loader.git

# Video helper suite
!git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git

# Install node requirements
%cd /content/ComfyUI/custom_nodes/ComfyUI_KJNodes  &&  pip install -r requirements.txt -q
%cd /content/ComfyUI/custom_nodes/ComfyUI_GGUF     &&  pip install -r requirements.txt -q
for d in ComfyUI-LTXVideo LTX2EasyPrompt-LD LTX2-Master-Loader; do
    req="/content/ComfyUI/custom_nodes/$d/requirements.txt"
    [ -f "$req" ] && pip install -r "$req" -q
done

import subprocess
subprocess.run(["apt-get","-y","install","-qq","aria2","ffmpeg"],
               check=True, capture_output=True)

%cd /content/ComfyUI
import os, sys
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, "/content/ComfyUI")
clear_output()
print("✅ Environment ready. All custom nodes installed.")


# ══════════════════════════════════════════════════════════════════════════
# CELL 2  —  MODEL DOWNLOADS  (run once; skips cached files)
# ══════════════════════════════════════════════════════════════════════════
# @title { "single-column": true }
# @markdown ## 💥 2. Download Model Weights

import os, subprocess
from pathlib import Path

def _dl(url: str, dest: str, filename: str = None) -> str:
    Path(dest).mkdir(parents=True, exist_ok=True)
    fn   = filename or url.split("/")[-1].split("?")[0]
    full = os.path.join(dest, fn)
    if os.path.exists(full) and os.path.getsize(full) > 1_000_000:
        print(f"  ↳ cached  {fn}"); return fn
    cmd = ["aria2c","--console-log-level=error","--summary-interval=0","--quiet",
           "-c","-x","16","-s","16","-k","1M","-d",dest,"-o",fn, url]
    print(f"  ↓ {fn}…", end=" ", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    print("done." if r.returncode == 0 else f"FAILED: {r.stderr.strip()}")
    return fn if r.returncode == 0 else None

KIJAI   = "https://huggingface.co/Kijai/LTXV2_comfy/resolve/main"
CORG    = "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files"
LTX_HF  = "https://huggingface.co/Lightricks"
UNET_D  = "/content/ComfyUI/models/unet"
TE_D    = "/content/ComfyUI/models/text_encoders"
VAE_D   = "/content/ComfyUI/models/vae"
UPD     = "/content/ComfyUI/models/latent_upscale_models"
LR_D    = "/content/ComfyUI/models/loras"

print("── Core models ──────────────────────────────────────────────────────")
_M_UNET  = _dl(f"{KIJAI}/diffusion_models/ltx-2-19b-distilled_Q4_K_M.gguf",  UNET_D)
# fp4 = Blackwell RTX 5000; switch to fp8 line for T4/A100
_M_CLIP1 = _dl(f"{CORG}/text_encoders/gemma_3_12B_it_fp4_mixed.safetensors",  TE_D)
# _M_CLIP1 = _dl(f"{CORG}/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors", TE_D)
_M_CLIP2 = _dl(f"{KIJAI}/text_encoders/ltx-2-19b-embeddings_connector_distill_bf16.safetensors", TE_D)
_M_VAE   = _dl(f"{KIJAI}/VAE/LTX2_video_vae_bf16.safetensors",                VAE_D)
_M_AVAE  = _dl(f"{KIJAI}/VAE/LTX2_audio_vae_bf16.safetensors",                VAE_D)
_M_TAEV  = _dl("https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/taeltx2_3.safetensors", VAE_D)
_M_UP    = _dl(f"{LTX_HF}/LTX-2/resolve/main/ltx-2-spatial-upscaler-x2-1.0.safetensors", UPD)

print("\n── LoRAs ────────────────────────────────────────────────────────────")
_LORA_URLS = {
    "Detailer":    f"{LTX_HF}/LTX-2-19b-IC-LoRA-Detailer/resolve/main/ltx-2-19b-ic-lora-detailer.safetensors",
    "Canny":       f"{LTX_HF}/LTX-2-19b-IC-LoRA-Canny-Control/resolve/main/ltx-2-19b-ic-lora-canny-control.safetensors",
    "Depth":       f"{LTX_HF}/LTX-2-19b-IC-LoRA-Depth-Control/resolve/main/ltx-2-19b-ic-lora-depth-control.safetensors",
    "Pose":        f"{LTX_HF}/LTX-2-19b-IC-LoRA-Pose-Control/resolve/main/ltx-2-19b-ic-lora-pose-control.safetensors",
    "Dolly-In":    f"{LTX_HF}/LTX-2-19b-LoRA-Camera-Control-Dolly-In/resolve/main/ltx-2-19b-lora-camera-control-dolly-in.safetensors",
    "Dolly-Out":   f"{LTX_HF}/LTX-2-19b-LoRA-Camera-Control-Dolly-Out/resolve/main/ltx-2-19b-lora-camera-control-dolly-out.safetensors",
    "Dolly-Left":  f"{LTX_HF}/LTX-2-19b-LoRA-Camera-Control-Dolly-Left/resolve/main/ltx-2-19b-lora-camera-control-dolly-left.safetensors",
    "Dolly-Right": f"{LTX_HF}/LTX-2-19b-LoRA-Camera-Control-Dolly-Right/resolve/main/ltx-2-19b-lora-camera-control-dolly-right.safetensors",
    "Jib-Up":      f"{LTX_HF}/LTX-2-19b-LoRA-Camera-Control-Jib-Up/resolve/main/ltx-2-19b-lora-camera-control-jib-up.safetensors",
    "Jib-Down":    f"{LTX_HF}/LTX-2-19b-LoRA-Camera-Control-Jib-Down/resolve/main/ltx-2-19b-lora-camera-control-jib-down.safetensors",
    "Static":      f"{LTX_HF}/LTX-2-19b-LoRA-Camera-Control-Static/resolve/main/ltx-2-19b-lora-camera-control-static.safetensors",
}
os.makedirs(LR_D, exist_ok=True)
for name, url in _LORA_URLS.items():
    r = _dl(url, LR_D)
    print(f"   {'✅' if r else '❌'}  {name}")

print("\n✅ All models ready.")


# ══════════════════════════════════════════════════════════════════════════
# CELL 3  —  CORE IMPORTS & UTILITIES
# ══════════════════════════════════════════════════════════════════════════
# @title { "single-column": true }
# @markdown ## 💥 3. Imports, Helpers, Node Loader

import gc, re, json, time, shutil, warnings, asyncio
import numpy as np
import torch
import cv2
from PIL import Image
from pathlib import Path
from typing import Optional, List, Any, Union, Sequence, Mapping
from base64 import b64encode
from IPython.display import display, HTML, clear_output
from google.colab import files

warnings.filterwarnings("ignore")
sys.path.insert(0, "/content/ComfyUI")
from nodes import NODE_CLASS_MAPPINGS, LoraLoaderModelOnly
import folder_paths

# ── VRAM management ────────────────────────────────────────────────────────
def _vram_free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

def _vram_print(tag=""):
    if not torch.cuda.is_available(): return
    u = torch.cuda.memory_allocated() / 1024**3
    t = torch.cuda.get_device_properties(0).total_memory / 1024**3
    filled = int(20 * u / t) if t else 0
    bar = "█" * filled + "░" * (20 - filled)
    print(f"   💾 VRAM [{bar}] {u:.1f}/{t:.1f} GB  {tag}")

# ── Tensor / image helpers ─────────────────────────────────────────────────
def get_value_at_index(obj: Union[Sequence, Mapping], idx: int) -> Any:
    try:    return obj[idx]
    except KeyError: return obj["result"][idx]

def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)          # (1,H,W,3)

def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    if t.ndim == 4: t = t[0]
    return Image.fromarray((t.cpu().numpy()*255).clip(0,255).astype(np.uint8),"RGB")

def load_image_tensor(path: str) -> Optional[torch.Tensor]:
    if not path or not os.path.exists(path): return None
    return pil_to_tensor(Image.open(path).convert("RGB"))

def get_last_frame_tensor(video_path: str) -> Optional[torch.Tensor]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n == 0: return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, n - 1)
    ok, frame = cap.read(); cap.release()
    if not ok: return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(frame).float().unsqueeze(0) / 255.0

# ── Video helpers ──────────────────────────────────────────────────────────
def display_video(path: str):
    if not path or not os.path.exists(path): return
    data = b64encode(open(path,"rb").read()).decode()
    display(HTML(
        '<video width=720 controls autoplay loop muted>'
        f'<source src="data:video/mp4;base64,{data}" type="video/mp4"></video>'))

def save_video_obj(video_obj, prefix="IFE") -> str:
    from comfy_api.latest import Types
    w, h = video_obj.get_dimensions()
    folder, fname, ctr, _, _ = folder_paths.get_save_image_path(
        f"video/{prefix}", folder_paths.get_output_directory(), w, h)
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
    subprocess.run(["ffmpeg","-y","-f","concat","-safe","0","-i",list_file,
                    "-c","copy", output_path], check=True, capture_output=True)
    print(f"   ✓ Concatenated → {output_path}")
    return output_path

# ── ComfyUI async node loader ──────────────────────────────────────────────
_NODES_LOADED = False
def _load_comfy_nodes():
    global _NODES_LOADED
    if _NODES_LOADED: return
    import nest_asyncio
    from nodes import init_builtin_extra_nodes, init_external_custom_nodes
    async def _l():
        failed = await init_builtin_extra_nodes()
        await init_external_custom_nodes()
        if failed: print(f"   ⚠  Node failures: {failed}")
    try:    asyncio.run(_l())
    except RuntimeError:
        nest_asyncio.apply()
        asyncio.get_event_loop().run_until_complete(_l())
    _NODES_LOADED = True

def _audio_vae(name: str):
    if "VAELoaderKJ" in NODE_CLASS_MAPPINGS:
        return NODE_CLASS_MAPPINGS["VAELoaderKJ"]().load_vae(
            vae_name=name, device="main_device", weight_dtype="fp16")
    return NODE_CLASS_MAPPINGS["VAELoader"]().load_vae(vae_name=name)

print("✅ Core utilities ready.")


# ══════════════════════════════════════════════════════════════════════════
# SECTION A  —  VISION DESCRIBE ENGINE
# Standalone Qwen2.5-VL wrapper. No ComfyUI node wrapper.
# Loads → describes → unloads immediately to free VRAM.
# ══════════════════════════════════════════════════════════════════════════

class VisionDescribeEngine:
    """Analyses an image and returns a 100-130 word scene description."""

    MODEL_OPTIONS = {
        "3B-fast": "huihui-ai/Qwen2.5-VL-3B-Instruct-abliterated",   # ~5 GB
        "7B-nsfw": "prithivMLmods/Qwen2.5-VL-7B-Abliterated-Caption-it",  # ~10 GB
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
        for s in ["<|im_end|>","<|endoftext|>"]:
            ids = tok.encode(s, add_special_tokens=False)
            if len(ids)==1 and ids[0] not in stop_ids: stop_ids.append(ids[0])

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


# ══════════════════════════════════════════════════════════════════════════
# SECTION B  —  EASY PROMPT ENGINE
# Full cinematic LLM expansion — no ComfyUI wrapper.
# Includes: pacing, character bible lock, dialogue control,
# explicit content tiers, sequence detection, full output cleaning.
# ══════════════════════════════════════════════════════════════════════════

# ── Negative prompt builder ────────────────────────────────────────────────
_NEG_BASE = (
    "blurry, out of focus, low quality, worst quality, jpeg artifacts, "
    "static, no motion, frozen, duplicate, watermark, text, signature, "
    "poorly drawn, bad anatomy, deformed, disfigured, extra limbs, "
    "missing limbs, overexposed, underexposed, grainy, noise, flickering"
)

def _build_neg(result: str, user_input: str) -> str:
    c = (result + " " + user_input).lower()
    extras = []
    if any(w in c for w in ["indoor","room","interior","bedroom","kitchen","office"]):
        extras.append("harsh outdoor lighting, direct sunlight")
    elif any(w in c for w in ["outdoor","street","beach","forest","park"]):
        extras.append("studio background, indoor lighting")
    if any(w in c for w in ["pussy","cock","penis","vagina","nude","naked","nipple","breast"]):
        extras.append("censored, mosaic, pixelated, black bar, blurred genitals")
    if any(w in c for w in ["close-up","close up","portrait","headshot"]):
        extras.append("wide angle distortion, fish eye")
    elif any(w in c for w in ["wide shot","wide angle","aerial","bird's-eye"]):
        extras.append("close-up, portrait crop")
    if any(w in c for w in ["night","dark","moonlight","dimly lit","candlelight"]):
        extras.append("overexposed, bright daylight, blown highlights")
    elif any(w in c for w in ["daylight","sunny","golden hour","bright"]):
        extras.append("underexposed, dark shadows, black crush")
    if any(w in c for w in ["two women","two men","two people","couple","both"]):
        extras.append("merged bodies, fused figures, incorrect number of people")
    return ", ".join([_NEG_BASE] + extras)


class EasyPromptEngine:
    """
    Expands a simple story beat into a dense cinematic LTX-2 prompt.
    Loads the LLM, generates, cleans output, then unloads to free VRAM.
    """

    MODELS = {
        "8B":  "mlabonne/NeuralDaredevil-8B-abliterated",      # ~10 GB
        "3B":  "huihui-ai/Llama-3.2-3B-Instruct-abliterated",  # ~4 GB (T4 safe)
        "14B": "huihui-ai/Huihui-Qwen3-14B-abliterated-v2",    # ~18 GB
    }

    SYSTEM_PROMPT = """You are a cinematic prompt writer for LTX-2, an AI video generation model. Expand the user's idea into a rich, video-ready prompt.

PRIORITY ORDER:
1. Video style & genre (slow-burn thriller, documentary, editorial, action blockbuster)
2. Camera angle & shot type (low-angle close-up, bird's-eye wide, Dutch angle medium)
3. Character description — age MUST be a specific number (e.g. "a 28-year-old woman"), body type, hair, skin, clothing. Use exact words from the user.
4. Scene & environment (location, time of day, lighting, colour palette, atmosphere)
5. Action & motion — continuous present-tense sequence.
6. Camera movement — prose only, no screenplay brackets like (HOLD) or (DOWN 10°).
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
        print(f"   [EasyPrompt] Loaded.")

    def _unload(self):
        if self._model is not None:
            try: self._model.to("cpu")
            except: pass
        self._model = None; self._tok = None; self._loaded_key = None
        _vram_free()
        print("   [EasyPrompt] VRAM cleared.")

    def _stop_ids(self) -> List[int]:
        delims = ["assistant","user","system","<|eot_id|>","<|end_of_turn|>",
                  "<|im_end|>","<end_of_turn>","[/INST]","### Human","### Assistant"]
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

        character_bible — injected as a hard [CHARACTER BIBLE — NON-NEGOTIABLE]
        block so the LLM cannot alter hair, age, clothing, or any locked attribute
        across clips. This is the primary mechanism for character consistency.
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

        # ── Pacing constraint ────────────────────────────────────────────
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

        # ── Character bible lock ─────────────────────────────────────────
        bible_clause = ""
        if character_bible.strip():
            bible_clause = (
                f"\n[CHARACTER BIBLE — NON-NEGOTIABLE: Every character attribute below "
                f"MUST remain exactly as described. Do NOT alter hair, age, skin, "
                f"clothing, or any other attribute. This overrides any inference:\n"
                f"{character_bible.strip()}\n]"
            )

        # ── Scene context (from Vision Describe) ─────────────────────────
        if scene_context.strip():
            effective = (
                f"[SCENE CONTEXT FROM IMAGE — authoritative, do not contradict]\n"
                f"{scene_context.strip()}\n\n"
                f"[USER DIRECTION — action, style, mood]\n{user_input.strip()}"
            )
        else:
            effective = user_input.strip()

        # ── LoRA trigger injection ────────────────────────────────────────
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


# ══════════════════════════════════════════════════════════════════════════
# SECTION C  —  CHARACTER BIBLE
# Serialisable cross-scene character consistency record.
# Injected verbatim into every EasyPromptEngine call.
# ══════════════════════════════════════════════════════════════════════════

class CharacterBible:
    """
    Records named character attributes and serialises them as a prompt-injection
    block. The block is passed to EasyPromptEngine as `character_bible` so the
    LLM receives a hard [NON-NEGOTIABLE] constraint preventing attribute drift
    across all 12 clips.

    Auto-populated from Vision Describe output on the seed image (recommended),
    or filled manually with `add()`.
    """

    def __init__(self):
        self._chars: dict = {}

    def add(self, name: str, **attributes):
        """Manually define a character."""
        self._chars[name] = dict(attributes)

    def extract_from_description(self, name: str, description: str):
        """
        Store a raw Vision Describe output under a character name.
        The full description is injected as-is — it already contains all
        relevant attributes in natural language.
        """
        self._chars[name] = {"_raw": description.strip()}

    def to_prompt_block(self) -> str:
        """Returns the injection string for EasyPromptEngine."""
        if not self._chars: return ""
        lines = []
        for name, attrs in self._chars.items():
            if "_raw" in attrs:
                lines.append(f"CHARACTER — {name}:\n{attrs['_raw']}")
            else:
                attr_str = "; ".join(f"{k}: {v}" for k, v in attrs.items())
                lines.append(f"CHARACTER — {name}: {attr_str}")
        return "\n\n".join(lines)

    def has_characters(self) -> bool:
        return bool(self._chars)

    def names(self) -> List[str]:
        return list(self._chars.keys())

    def save(self, path: str):
        with open(path, "w") as f: json.dump(self._chars, f, indent=2)
        print(f"   [Bible] Saved → {path}")

    def load(self, path: str):
        with open(path) as f: self._chars = json.load(f)
        print(f"   [Bible] Loaded from {path}")

    def __repr__(self):
        return f"CharacterBible({self.names()})"


# ══════════════════════════════════════════════════════════════════════════
# SECTION D  —  GENERATE CLIP  (two-pass LTX-2 19B GGUF pipeline)
#
# VRAM sequence (T4-safe, peak ~13 GB):
#   1. Load CLIP Gemma (~6-8 GB) → encode text → DELETE CLIP → free
#   2. Load UNet GGUF (~10 GB) → apply LoRAs (model-only, no CLIP needed)
#   3. Load VAE_video (~1 GB) + VAE_audio (~1 GB) + upscaler (~0.5 GB)
#   4. Prepare latents (I2V if image provided, else T2V)
#   5. Pass 1 (ManualSigmas + euler) → del guider
#   6. Pass 2 (spatial upscale + gradient_estimation) → del UNet + del guider
#   7. Decode video (tiled or standard) → del VAE_video
#   8. Decode audio → del VAE_audio
#   9. Save + return path
# ══════════════════════════════════════════════════════════════════════════

# ── LoRA stack (applied in Section D) ────────────────────────────────────
# Mirrors LTX2MasterLoaderLD node JSON format.
# Edit slot 1 filename / strength, enable slots 2-10 for camera LoRAs.
LD_LORA_STACK = [
    {"on": True,  "lora": "ltx-2-19b-ic-lora-detailer.safetensors", "guard": False, "strength": 0.4},
    {"on": False, "lora": "None", "guard": False, "strength": 1.0},   # camera LoRA slot
    {"on": False, "lora": "None", "guard": False, "strength": 1.0},
    {"on": False, "lora": "None", "guard": False, "strength": 1.0},
    {"on": False, "lora": "None", "guard": False, "strength": 1.0},
    {"on": False, "lora": "None", "guard": False, "strength": 1.0},
    {"on": False, "lora": "None", "guard": False, "strength": 1.0},
    {"on": False, "lora": "None", "guard": False, "strength": 1.0},
    {"on": False, "lora": "None", "guard": False, "strength": 1.0},
    {"on": False, "lora": "None", "guard": False, "strength": 1.0},
]
_LORA_STACK_JSON = json.dumps(LD_LORA_STACK)

def _apply_loras(unet, clip_model):
    """
    Apply LD_LORA_STACK to UNet.
    Tries LTX2MasterLoaderLD node first; falls back to LoraLoaderModelOnly loop.
    clip_model is passed to the node but unchanged for IC/camera LoRAs.
    """
    active = [s for s in LD_LORA_STACK
              if s.get("on") and s.get("lora") not in (None,"None","")]
    if not active:
        return unet, clip_model

    # LTX2MasterLoaderLD node (preferred — applies full stack in one call)
    if "LTX2MasterLoaderLD" in NODE_CLASS_MAPPINGS:
        try:
            node   = NODE_CLASS_MAPPINGS["LTX2MasterLoaderLD"]()
            result = getattr(node, node.FUNCTION)(
                model=unet, clip=clip_model, stack_data=_LORA_STACK_JSON)
            print(f"   [LoRA] ✓ {len(active)} slot(s) via LTX2MasterLoaderLD")
            return get_value_at_index(result, 0), clip_model
        except Exception as e:
            print(f"   [LoRA] MasterLoader failed ({e}) — manual fallback")

    # LoraLoaderModelOnly fallback
    for slot in active:
        name, strength, guard = slot["lora"], slot["strength"], slot.get("guard", False)
        try:
            unet = LoraLoaderModelOnly().load_lora_model_only(unet, name, strength)[0]
            print(f"   [LoRA] ✓  {name} @ {strength}")
        except Exception as e:
            print(f"   [LoRA] {'skip' if guard else 'fail'}  {name}: {e}")
    return unet, clip_model


def generate_clip(
    image_tensor:    Optional[torch.Tensor],  # (1,H,W,3) float32  or None
    prompt:          str,
    neg_prompt:      str,
    width:           int   = 768,
    height:          int   = 512,
    frames:          int   = 121,
    fps:             int   = 25,
    seed:            int   = 42,
    image_strength:  float = 1.0,
    pass1_sigmas:    str   = "1., 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0",
    pass1_sampler:   str   = "euler",
    pass1_cfg:       float = 1.0,
    pass2_sigmas:    str   = "0.909375, 0.725, 0.421875, 0.0",
    pass2_sampler:   str   = "gradient_estimation",
    pass2_cfg:       float = 1.0,
    pass2_seed:      int   = 0,
    use_tiled_vae:   bool  = True,
    tiled_stiles:    int   = 2,
    tiled_soverlap:  int   = 8,
    tiled_tlen:      int   = 48,
    tiled_toverlap:  int   = 4,
    output_prefix:   str   = "IFE",
) -> str:
    """
    Two-pass LTX-2 19B GGUF generation for one clip.
    T4-safe: CLIP is deleted before UNet loads; UNet is deleted before decode.
    """
    _load_comfy_nodes()

    img_bypass = image_tensor is None
    img_str    = image_strength if not img_bypass else 0.0

    print(f"   [Clip] {'I2V' if not img_bypass else 'T2V'}  "
          f"{width}×{height}  {frames}f  seed={seed}")
    _vram_print("before clip load")

    with torch.inference_mode():

        # ── STEP 1: CLIP → encode → delete ────────────────────────────────
        print("   [Clip] Loading CLIP…")
        clip_ld = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
        try:
            clip_raw = get_value_at_index(
                clip_ld.load_clip(clip_name1=_M_CLIP1, clip_name2=_M_CLIP2,
                                  type="ltxv", device="default"), 0)
        except Exception as e:
            print(f"   ⚠  fp4 failed ({e}), trying fp8…")
            fp8 = "gemma_3_12B_it_fp8_scaled.safetensors"
            clip_raw = get_value_at_index(
                clip_ld.load_clip(clip_name1=fp8, clip_name2=_M_CLIP2,
                                  type="ltxv", device="default"), 0)

        cte      = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cond_pos = cte.encode(text=prompt, clip=clip_raw)
        zero_out = NODE_CLASS_MAPPINGS["ConditioningZeroOut"]()
        cond_0   = zero_out.zero_out(conditioning=get_value_at_index(cond_pos, 0))
        ltxv_cn  = NODE_CLASS_MAPPINGS["LTXVConditioning"]()
        cond     = ltxv_cn.EXECUTE_NORMALIZED(
            frame_rate=float(fps),
            positive=get_value_at_index(cond_pos, 0),
            negative=get_value_at_index(cond_0, 0))

        # Delete CLIP now — frees ~6-8 GB for UNet
        del clip_raw
        _vram_free()
        _vram_print("CLIP deleted")

        # ── STEP 2: UNet + LoRAs ───────────────────────────────────────────
        print("   [Clip] Loading UNet (GGUF Q4_K_M)…")
        unet_ld = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
        unet    = get_value_at_index(unet_ld.load_unet(unet_name=_M_UNET), 0)
        # Apply LoRA stack (model-only — no CLIP needed for IC/camera LoRAs)
        _dummy_clip = None  # LTX2MasterLoaderLD requires clip but IC LoRAs ignore it
        unet, _ = _apply_loras(unet, _dummy_clip)
        _vram_print("UNet + LoRAs loaded")

        # ── STEP 3: VAEs + upscaler ────────────────────────────────────────
        vae_ld  = NODE_CLASS_MAPPINGS["VAELoader"]()
        vae_v   = get_value_at_index(vae_ld.load_vae(vae_name=_M_VAE), 0)
        vae_a   = get_value_at_index(_audio_vae(_M_AVAE), 0)
        uml     = NODE_CLASS_MAPPINGS["LatentUpscaleModelLoader"]()
        up_mdl  = get_value_at_index(uml.EXECUTE_NORMALIZED(model_name=_M_UP), 0)

        # ── STEP 4: Latent preparation ─────────────────────────────────────
        # Half-resolution empty latent (EmptyImage → ×0.5 → GetImageSize)
        ei      = NODE_CLASS_MAPPINGS["EmptyImage"]()
        full_i  = ei.generate(width=width, height=height, batch_size=1, color=0)
        rimn    = NODE_CLASS_MAPPINGS["ResizeImageMaskNode"]()
        half_i  = rimn.EXECUTE_NORMALIZED(
            input=get_value_at_index(full_i, 0), scale_method="area",
            resize_type={"resize_type":"scale by multiplier","multiplier":0.5})
        gis     = NODE_CLASS_MAPPINGS["GetImageSize"]()
        hsz     = gis.EXECUTE_NORMALIZED(image=get_value_at_index(half_i, 0))
        hw, hh  = get_value_at_index(hsz, 0), get_value_at_index(hsz, 1)

        eltxv   = NODE_CLASS_MAPPINGS["EmptyLTXVLatentVideo"]()
        vid_lat = eltxv.EXECUTE_NORMALIZED(width=hw, height=hh, length=frames, batch_size=1)

        # I2V: condition the latent on the seed/last-frame image
        if not img_bypass:
            rim2 = NODE_CLASS_MAPPINGS["ResizeImageMaskNode"]()
            ri2  = rim2.EXECUTE_NORMALIZED(
                input=image_tensor, scale_method="lanczos",
                resize_type={"resize_type":"scale dimensions",
                             "width":hw*2,"height":hh*2,"crop":"center"})
            ppn  = NODE_CLASS_MAPPINGS["LTXVPreprocess"]()
            pp   = get_value_at_index(ppn.EXECUTE_NORMALIZED(
                img_compression=33, image=get_value_at_index(ri2, 0)), 0)
            i2v  = NODE_CLASS_MAPPINGS["LTXVImgToVideoInplace"]()
            vid_lat = (get_value_at_index(i2v.EXECUTE_NORMALIZED(
                strength=img_str, bypass=False,
                vae=vae_v, image=pp,
                latent=get_value_at_index(vid_lat, 0)), 0),)
        else:
            vid_lat = (get_value_at_index(vid_lat, 0),)

        # Audio latent + concat AV
        elalat  = NODE_CLASS_MAPPINGS["LTXVEmptyLatentAudio"]()
        aud_lat = elalat.EXECUTE_NORMALIZED(
            frames_number=frames, frame_rate=fps, batch_size=1, audio_vae=vae_a)
        catav   = NODE_CLASS_MAPPINGS["LTXVConcatAVLatent"]()
        av_lat  = get_value_at_index(catav.EXECUTE_NORMALIZED(
            video_latent=vid_lat[0],
            audio_latent=get_value_at_index(aud_lat, 0)), 0)

        # ── STEP 5: Pass 1 ────────────────────────────────────────────────
        print("   [Clip] Pass 1…")
        _vram_print()
        ms  = NODE_CLASS_MAPPINGS["ManualSigmas"]()
        ks  = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
        rn  = NODE_CLASS_MAPPINGS["RandomNoise"]()
        cfg = NODE_CLASS_MAPPINGS["CFGGuider"]()
        sca = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()

        g1  = cfg.EXECUTE_NORMALIZED(cfg=pass1_cfg, model=unet,
                                     positive=get_value_at_index(cond,0),
                                     negative=get_value_at_index(cond,1))
        o1  = sca.EXECUTE_NORMALIZED(
            noise=get_value_at_index(rn.EXECUTE_NORMALIZED(noise_seed=seed),0),
            guider=get_value_at_index(g1,0),
            sampler=get_value_at_index(ks.EXECUTE_NORMALIZED(sampler_name=pass1_sampler),0),
            sigmas=get_value_at_index(ms.EXECUTE_NORMALIZED(sigmas=pass1_sigmas),0),
            latent_image=av_lat)
        p1_av = get_value_at_index(o1, 0)
        del g1; _vram_free()
        print("   [Clip] Pass 1 ✓")

        # ── STEP 6: Pass 2 ────────────────────────────────────────────────
        print("   [Clip] Pass 2 (upscale + refine)…")
        sep   = NODE_CLASS_MAPPINGS["LTXVSeparateAVLatent"]()
        s1    = sep.EXECUTE_NORMALIZED(av_latent=p1_av)
        vl1, al1 = get_value_at_index(s1,0), get_value_at_index(s1,1)

        crop  = NODE_CLASS_MAPPINGS["LTXVCropGuides"]()
        cr    = crop.EXECUTE_NORMALIZED(positive=get_value_at_index(cond,0),
                                        negative=get_value_at_index(cond,1),
                                        latent=vl1)
        g2    = cfg.EXECUTE_NORMALIZED(cfg=pass2_cfg, model=unet,
                                       positive=get_value_at_index(cr,0),
                                       negative=get_value_at_index(cr,1))

        ltxvup = NODE_CLASS_MAPPINGS["LTXVLatentUpsampler"]()
        up     = ltxvup.upsample_latent(samples=get_value_at_index(cr,2),
                                        upscale_model=up_mdl, vae=vae_v)
        del up_mdl; _vram_free()

        av2   = get_value_at_index(catav.EXECUTE_NORMALIZED(
            video_latent=get_value_at_index(up,0), audio_latent=al1), 0)

        o2    = sca.EXECUTE_NORMALIZED(
            noise=get_value_at_index(rn.EXECUTE_NORMALIZED(noise_seed=pass2_seed),0),
            guider=get_value_at_index(g2,0),
            sampler=get_value_at_index(ks.EXECUTE_NORMALIZED(sampler_name=pass2_sampler),0),
            sigmas=get_value_at_index(ms.EXECUTE_NORMALIZED(sigmas=pass2_sigmas),0),
            latent_image=av2)
        p2_den = get_value_at_index(o2, 1)

        # BIG VRAM RELEASE — UNet is done
        del g2, unet; _vram_free()
        _vram_print("UNet deleted")
        print("   [Clip] Pass 2 ✓")

        # ── STEP 7: Decode ────────────────────────────────────────────────
        s2     = sep.EXECUTE_NORMALIZED(av_latent=p2_den)
        vl_f   = get_value_at_index(s2, 0)
        al_f   = get_value_at_index(s2, 1)

        decoded = None
        if use_tiled_vae:
            try:
                td     = NODE_CLASS_MAPPINGS["LTXVSpatioTemporalTiledVAEDecode"]()
                decoded = get_value_at_index(td.EXECUTE_NORMALIZED(
                    vae=vae_v, latents=vl_f,
                    spatial_tiles=tiled_stiles, spatial_overlap=tiled_soverlap,
                    temporal_tile_length=tiled_tlen, temporal_overlap=tiled_toverlap,
                    last_frame_fix=False, working_device="auto", working_dtype="auto"), 0)
                print("   [Clip] Tiled VAE ✓")
            except (KeyError, Exception) as e:
                print(f"   [Clip] Tiled VAE skipped ({type(e).__name__}) — standard decode")
                use_tiled_vae = False

        if not use_tiled_vae:
            vd     = NODE_CLASS_MAPPINGS["VAEDecode"]()
            decoded = get_value_at_index(vd.decode(samples=vl_f, vae=vae_v), 0)

        del vae_v; _vram_free()

        aud_d  = NODE_CLASS_MAPPINGS["LTXVAudioVAEDecode"]()
        audio  = aud_d.EXECUTE_NORMALIZED(samples=al_f, audio_vae=vae_a)
        del vae_a; _vram_free()

        # ── STEP 8: Save ─────────────────────────────────────────────────
        cv_node = NODE_CLASS_MAPPINGS["CreateVideo"]()
        vid_obj = cv_node.EXECUTE_NORMALIZED(
            fps=fps, images=decoded,
            audio=get_value_at_index(audio, 0))
        path = save_video_obj(get_value_at_index(vid_obj, 0), prefix=output_prefix)
        _vram_print("after save")
        return path


# ══════════════════════════════════════════════════════════════════════════
# SECTION E  —  INFINITE FLOW ENGINE
# 12-scene orchestrator with last-frame chaining + rolling scene history.
# ══════════════════════════════════════════════════════════════════════════

class InfiniteFlowEngine:
    """
    Generates a sequence of clips (up to 12 by default) where:
      1. VisionDescribeEngine analyses the current seed/last frame
      2. EasyPromptEngine expands the story beat into a cinematic prompt,
         injecting the CharacterBible as a hard consistency lock
      3. generate_clip() renders the clip with the two-pass LTX-2 pipeline
      4. get_last_frame_tensor() extracts the last frame for the next beat
      5. The clip is saved to output_dir; all paths are returned at the end

    Every heavyweight model is loaded → used → unloaded in strict sequence.
    Peak VRAM on T4: ~13 GB (UNet + VAEs + upscaler during Pass 2).
    """

    def __init__(
        self,
        character_bible:   Optional[CharacterBible] = None,
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

        self._vision_engine = VisionDescribeEngine(vision_model, offline)
        self._prompt_engine = EasyPromptEngine(llm_model, offline, keep_loaded=False)

        self._clip_paths:    List[str] = []
        self._scene_history: List[str] = []   # last 2 expanded prompts as rolling context

    # ── Rolling context (forward continuity) ──────────────────────────────
    def _rolling_ctx(self) -> str:
        return "\n".join(self._scene_history[-2:])

    # ── Single beat processing ─────────────────────────────────────────────
    def _process_beat(self, beat: str, current_image: Optional[torch.Tensor],
                      beat_idx: int, base_seed: int) -> tuple:
        beat_seed = base_seed + (beat_idx - 1) * 1000

        # Vision Describe — only from beat 2 onward (beat 1 uses seed image desc)
        if beat_idx > 1 and current_image is not None and self.use_vision:
            print(f"   [Beat {beat_idx}] Vision Describe…")
            scene_ctx = self._vision_engine.describe(current_image)
            _vram_free()
        else:
            scene_ctx = self._rolling_ctx()

        # Expand beat → cinematic prompt with character bible lock
        print(f"   [Beat {beat_idx}] EasyPrompt expand…")
        prompt, neg = self._prompt_engine.generate(
            user_input=beat,
            frame_count=self.frames,
            creativity=self.creativity,
            seed=beat_seed,
            scene_context=scene_ctx,
            lora_triggers=self.lora_triggers,
            character_bible=self.bible.to_prompt_block(),
        )
        _vram_free()

        # Store for next beat's rolling context
        self._scene_history.append(prompt[:400])
        if len(self._scene_history) > 2:
            self._scene_history = self._scene_history[-2:]

        return prompt, neg, beat_seed

    # ── Main run ───────────────────────────────────────────────────────────
    def run(
        self,
        story_beats:     List[str],
        seed_image_path: Optional[str] = None,
        base_seed:       int   = 42,
    ) -> List[str]:
        """
        Process each beat in story_beats.
        Returns list of output .mp4 paths (one per beat).
        """
        current_image: Optional[torch.Tensor] = None

        # ── Load seed image ────────────────────────────────────────────────
        if seed_image_path and os.path.exists(seed_image_path):
            current_image = load_image_tensor(seed_image_path)
            print(f"[IFE] Seed image loaded: {seed_image_path}")

            # Auto-populate CharacterBible from seed image on first run
            if not self.bible.has_characters() and self.use_vision:
                print("[IFE] Extracting Character Bible from seed image…")
                desc = self._vision_engine.describe(current_image)
                _vram_free()
                self.bible.extract_from_description("Main Character", desc)
                self._scene_history.append(desc)
                print(f"[IFE] Bible set:\n{desc[:200]}…\n")

        print(f"\n[IFE] ═══  INFINITE FLOW ENGINE  ═══")
        print(f"[IFE]  Beats    : {len(story_beats)}")
        print(f"[IFE]  Size     : {self.width}×{self.height}  {self.frames}f @ {self.fps}fps")
        print(f"[IFE]  LLM      : {self.llm_model}  Vision: {self.vision_model}")
        print(f"[IFE]  Bible    : {self.bible.names() or 'empty (T2V)'}")
        print(f"[IFE]  Mode     : {'I2V' if current_image is not None else 'T2V'}\n")

        for beat_idx, beat in enumerate(story_beats, 1):
            print(f"\n[IFE] ── Beat {beat_idx}/{len(story_beats)} ────────────────────────────────")
            print(f"[IFE]   {beat[:80]}{'…' if len(beat)>80 else ''}")

            # ── Phase 1: Prompt generation ─────────────────────────────────
            prompt, neg, beat_seed = self._process_beat(
                beat, current_image, beat_idx, base_seed)

            print(f"\n  EXPANDED ({len(prompt.split())}w): {prompt[:200]}…")
            print(f"  NEG:  {neg[:100]}…\n")

            # ── Phase 2: Video generation ──────────────────────────────────
            try:
                clip_path = generate_clip(
                    image_tensor   = current_image,
                    prompt         = prompt,
                    neg_prompt     = neg,
                    width          = self.width,
                    height         = self.height,
                    frames         = self.frames,
                    fps            = self.fps,
                    seed           = beat_seed,
                    image_strength = self.img_strength,
                    use_tiled_vae  = self.use_tiled_vae,
                    output_prefix  = f"IFE_{beat_idx:03d}",
                )
            except torch.cuda.OutOfMemoryError:
                _vram_free()
                print(f"   ❌ OOM on beat {beat_idx}. Retrying in T2V mode…")
                clip_path = generate_clip(
                    image_tensor   = None,
                    prompt         = prompt,
                    neg_prompt     = neg,
                    width          = self.width,
                    height         = self.height,
                    frames         = self.frames,
                    fps            = self.fps,
                    seed           = beat_seed + 1,
                    use_tiled_vae  = False,
                    output_prefix  = f"IFE_{beat_idx:03d}_retry",
                )

            # Copy to organised output dir
            dest = os.path.join(self.output_dir, f"scene_{beat_idx:03d}.mp4")
            shutil.copy2(clip_path, dest)
            self._clip_paths.append(dest)
            print(f"\n[IFE] ✅  Beat {beat_idx} → {dest}")

            # ── Extract last frame for chaining ───────────────────────────
            last_frame = get_last_frame_tensor(dest)
            current_image = last_frame if last_frame is not None else None
            if last_frame is not None:
                print(f"[IFE]    Last frame extracted for beat {beat_idx+1}.")
            else:
                print(f"[IFE]    ⚠  No last frame — next beat will be T2V.")

            # ── Preview ────────────────────────────────────────────────────
            if self.show_previews:
                print(f"\n  ▶ Scene {beat_idx}:")
                display_video(dest)

        print(f"\n[IFE] ═══  {len(story_beats)} scenes complete  ═══")
        print(f"[IFE]  Output dir: {self.output_dir}")
        return self._clip_paths

    def concat_all(self, output_name: str = "full_video.mp4") -> str:
        """Concatenate all generated clips into one final video."""
        out = os.path.join(self.output_dir, output_name)
        return concatenate_clips(self._clip_paths, out)

    def download_all(self):
        for p in self._clip_paths:
            if os.path.exists(p): files.download(p)

    def save_bible(self, path: str = "/content/character_bible.json"):
        self.bible.save(path)


print("✅ All 5 sections defined.")
print("   A: VisionDescribeEngine  B: EasyPromptEngine  C: CharacterBible")
print("   D: generate_clip()       E: InfiniteFlowEngine")


# ══════════════════════════════════════════════════════════════════════════
# CELL 4  —  CONFIGURATION  (edit before running Cell 5)
# ══════════════════════════════════════════════════════════════════════════
# @title { "single-column": true }
# @markdown ## 💥 4. Configure Your 12-Scene Video

# ── LLM / Vision ──────────────────────────────────────────────────────────
LLM_MODEL    = "8B"       # @param ["8B","3B","14B"]
# "8B"  → NeuralDaredevil  — best quality, ~10 GB (may OOM with video model back-to-back on T4)
# "3B"  → Llama 3.2        — safe on T4, ~4 GB, good quality
# "14B" → Qwen3 14B        — best, ~18 GB, A100 only

VISION_MODEL  = "3B-fast"  # @param ["3B-fast","7B-nsfw"]
CREATIVITY    = 0.9        # @param {type:"number"}  0.7 literal / 0.9 balanced / 1.1 creative
INVENT_DIALOGUE = True     # @param {type:"boolean"}
LORA_TRIGGERS  = ""        # @param {type:"string"}  e.g. "ohwx woman"
USE_VISION     = True      # @param {type:"boolean"}
SHOW_PREVIEWS  = True      # @param {type:"boolean"}

# ── Video settings ─────────────────────────────────────────────────────────
WIDTH          = 768       # @param {type:"integer"}
HEIGHT         = 512       # @param {type:"integer"}
FRAMES         = 121       # @param {type:"integer"}   ~4.8s @ 25fps
FPS            = 25        # @param {type:"integer"}
BASE_SEED      = 42        # @param {type:"integer"}
IMAGE_STRENGTH = 1.0       # @param {type:"number"}  I2V conditioning (1.0=strong)
USE_TILED_VAE  = True      # @param {type:"boolean"}

# ── Reference / seed image ─────────────────────────────────────────────────
SEED_IMAGE_PATH = None     # @param {type:"string"}
# Set to local path, e.g. "/content/ComfyUI/input/elena.jpg"
# Or run: SEED_IMAGE_PATH = upload_image()  # (Colab upload dialog)

# ── LoRA stack (edit slots here) ──────────────────────────────────────────
# To add a camera LoRA: set "on": True and paste the filename in "lora"
# e.g. "ltx-2-19b-lora-camera-control-dolly-in.safetensors"
# (already downloaded in Cell 2 to /content/ComfyUI/models/loras/)
LD_LORA_STACK[0] = {"on": True,  "lora": "ltx-2-19b-ic-lora-detailer.safetensors",
                    "guard": False, "strength": 0.4}
_LORA_STACK_JSON  = json.dumps(LD_LORA_STACK)

# ── Character Bible (manual override — or leave empty for auto-extraction) ─
# If SEED_IMAGE_PATH is set and USE_VISION=True, the bible is auto-populated.
# You can also fill it manually below for precise control.
bible = CharacterBible()

# Example manual entry (comment out to use auto-extraction):
# bible.add("Elena",
#     age=28, ethnicity="white British", hair="long auburn waves",
#     eyes="green", build="slender athletic",
#     clothes="cream linen blouse, dark skinny jeans, white trainers",
#     other="small scar above left eyebrow, silver hoop earrings")

# ── 12 Story Beats ─────────────────────────────────────────────────────────
# Each string = one video clip (~4.8s at default settings).
# Total runtime: ~57.6s of video (12 × 4.8s).
# Edit to match your series concept.
STORY_BEATS = [
    # SCENE 1 — Establish character + location
    "Elena arrives at the entrance of a dimly lit urban apartment building at night, "
    "pushing through the glass door, rain dripping from her jacket",

    # SCENE 2 — Interior entry
    "She takes the elevator, watching the floor numbers tick upward, "
    "her reflection ghostly in the steel doors",

    # SCENE 3 — Apartment arrival
    "Elena unlocks her front door and steps inside the dark apartment, "
    "not turning on the lights, setting her keys on the counter quietly",

    # SCENE 4 — Tension builds
    "She moves to the kitchen, opens the fridge, stares blankly at the shelves, "
    "the cold blue light illuminating her face — she is not hungry",

    # SCENE 5 — Discovery
    "Elena notices a note on the kitchen table that was not there this morning, "
    "she picks it up slowly, her expression shifting from curiosity to alarm",

    # SCENE 6 — Phone call
    "She grabs her phone from her bag, dials a number, presses it to her ear — "
    "no answer. She tries again. Silence. She lowers the phone.",

    # SCENE 7 — Window
    "Elena walks to the window and peers down at the wet street below, "
    "watching a black car parked at the curb with its engine running",

    # SCENE 8 — Decision
    "She goes to the bedroom wardrobe, pulls out a small travel bag, "
    "begins packing quickly — clothes, passport, a laptop",

    # SCENE 9 — Interrupted
    "A sharp knock at the front door. Elena freezes mid-motion, "
    "bag half-packed, listening. Silence. Then another knock, louder.",

    # SCENE 10 — Confrontation
    "She approaches the door, looks through the peephole — "
    "a man in a grey coat stands in the corridor, face turned away",

    # SCENE 11 — Escape
    "Elena slips out through the apartment's service exit, "
    "moving quickly down the back stairwell, bag over her shoulder",

    # SCENE 12 — Cliffhanger ending
    "She emerges onto the rain-slicked alley behind the building, "
    "looks both ways, then runs toward the far end where a taxi waits with its light on",
]

print("✅ Configuration ready.")
print(f"   Beats     : {len(STORY_BEATS)}")
print(f"   Video     : {WIDTH}×{HEIGHT}  {FRAMES}f @ {FPS}fps  (~{FRAMES/FPS:.1f}s per clip)")
print(f"   LLM       : {LLM_MODEL}  Vision: {VISION_MODEL}  Creativity: {CREATIVITY}")
print(f"   Seed image: {SEED_IMAGE_PATH or 'None (T2V mode)'}")
print(f"   Bible     : {bible or 'auto-extract from seed image'}")


# ══════════════════════════════════════════════════════════════════════════
# CELL 5  —  RUN
# ══════════════════════════════════════════════════════════════════════════
# @title { "single-column": true }
# @markdown ## 💥 5. Run 12-Scene Infinite Flow

def _run():
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
        base_seed       = BASE_SEED,
    )

    # Save character bible for reuse across sessions
    engine.save_bible("/content/character_bible.json")

    # Concatenate all 12 scenes into one final video
    if len(clip_paths) > 1:
        final = engine.concat_all("12_scene_final.mp4")
        print(f"\n🎬 Final video: {final}")
        if SHOW_PREVIEWS: display_video(final)

    print("\n📥 Downloading individual clips…")
    engine.download_all()

    return engine, clip_paths


try:
    engine, output_clips = _run()

except torch.cuda.OutOfMemoryError:
    _vram_free()
    print("\n❌ CUDA OOM — suggested fixes:")
    print("   • Switch LLM_MODEL = '3B'  (saves ~6 GB during prompt phase)")
    print("   • Reduce FRAMES to 97  (saves ~1 GB of latent space)")
    print("   • Reduce WIDTH/HEIGHT: 704×448 is the next T4-safe step")
    print("   • Set USE_TILED_VAE = True if not already")
    print("   • Set USE_VISION = False  (skips Qwen VL load entirely)")

except Exception as e:
    import traceback
    print(f"\n❌ {type(e).__name__}: {e}")
    traceback.print_exc()
    print("\n💡 Quick fixes:")
    print("   UnetLoaderGGUF missing         → Cell 1: ComfyUI_GGUF not cloned")
    print("   LTXVCropGuides missing          → Cell 1: ComfyUI-LTXVideo not cloned")
    print("   LTXVSpatioTemporalTiledVAEDecode → Cell 1: ComfyUI-LTXVideo not cloned")
    print("   DualCLIPLoader fp4 error        → Use fp8: edit _M_CLIP1 in Cell 2")
    print("   qwen_vl_utils missing           → Cell 1: pip install qwen-vl-utils")
    print("   Deformed output (Pass 1)        → Change BASE_SEED and re-run Cell 5")
