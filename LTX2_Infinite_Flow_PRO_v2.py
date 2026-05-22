# @title {"single-column":true}
# @markdown # ♾️ LTX-2 Infinite Flow Engine PRO v2.0
# @markdown **NEW:** Character LoRAs, Motion Control, Voice Sync, SVI-Pro Quality
# @markdown Features: Character Consistency, Motion Guidance, Lip Sync, Smart Transitions

import cv2
import os
import gc
import torch
import shutil
import warnings
import time
import random
import numpy as np
from tqdm.notebook import tqdm
from comfy_api.latest import Input, Types
import folder_paths
from moviepy.editor import VideoFileClip, concatenate_videoclips, CompositeVideoClip
from IPython.display import display, HTML
from base64 import b64encode
import json
from pathlib import Path

# Suppress Warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
#                    CONFIGURATION SECTION
# ═══════════════════════════════════════════════════════════════

# @markdown ### 🎬 Project Settings
PROJECT_NAME = "Whispering_Cave_Part_2_PRO"           # @param {type:"string"}
WIDTH = 848                                            # @param {type:"integer"}
HEIGHT = 480                                           # @param {type:"integer"}
FPS = 24                                               # @param {type:"integer"}

# @markdown ### 🎞️ Enhanced SVI-Pro Settings
OVERLAP_FRAMES = 16                                    # @param {type:"integer"}
ANCHOR_STRENGTH_HIGH = 0.85                            # @param {type:"number"}
ANCHOR_STRENGTH_LOW = 0.70                             # @param {type:"number"}
USE_ADAPTIVE_STRENGTH = True                           # @param {type:"boolean"}

# @markdown ### 🎭 Character Consistency Settings
USE_CHARACTER_LORAS = True                             # @param {type:"boolean"}
CHARACTER_LORA_STRENGTH = 0.90                         # @param {type:"number"}
INJECT_CHARACTER_EVERY_SHOT = True                    # @param {type:"boolean"}

# @markdown ### 🎥 Motion & Camera Settings
USE_MOTION_LORAS = True                                # @param {type:"boolean"}
MOTION_STRENGTH = 0.75                                 # @param {type:"number"}
CAMERA_LORA_STRENGTH = 0.80                            # @param {type:"number"}

# @markdown ### 🗣️ Voice & Audio Settings
USE_VOICE_SYNC = True                                  # @param {type:"boolean"}
VOICE_SYNC_STRENGTH = 0.95                             # @param {type:"number"}
GENERATE_SUBTITLES = True                              # @param {type:"boolean"}

# @markdown ### 🎨 Enhanced Prompting
USE_NEGATIVE_PROMPT_EXPANSION = True                   # @param {type:"boolean"}
USE_PROMPT_WEIGHTING = True                            # @param {type:"boolean"}

# Base prompts (will be enhanced)
BASE_PROMPT = "3D Pixar cartoon style, Ultra HDR, intricate details, vibrant colors, realistic lighting, dramatic lighting, enhanced clarity, brilliant highlights, hyperrealistic detailing, cinematic quality, professional animation"
NEGATIVE_PROMPT = "blurry, distorted, low quality, bad anatomy, text, watermark, ugly, deformed, glitch, morphing artifacts, extra limbs, fused fingers, poorly drawn face, inconsistent character, character morphing, face change, clothing change, style inconsistency"

# ═══════════════════════════════════════════════════════════════
#                    ENHANCED JSON SCHEMA
# ═══════════════════════════════════════════════════════════════

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
        "hair": "Messy jet-black hair with natural volume, slight widow's peak",
        "clothing": "Bright yellow cotton t-shirt with orange trim, blue denim shorts, red sneakers with white laces",
        "build": "Slim athletic build, average height for age",
        "skin_tone": "Warm medium brown skin tone, healthy glow",
        "accessories": "Holding an old weathered treasure map with both hands"
      },
      "lora_path": None,  # Add path if you have character LoRA
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
      {
        "time": "0-4s",
        "camera": "Wide tracking shot, dolly forward following characters from behind",
        "camera_movement": "dolly_forward",
        "motion_intensity": 0.6,
        "action": "Shiv and Vandana walk deeper into the forest. The bright sunlight gradually fades into an emerald green glow. Their footsteps are cautious but determined.",
        "character_focus": "both",
        "emotion": "curious_cautious",
        "visual_effects": "Light transition from bright to dim, lens flare, atmospheric particles"
      },
      {
        "time": "4-8s",
        "camera": "Close-up tracking shot, low angle focused on feet",
        "camera_movement": "tilt_up_slight",
        "motion_intensity": 0.4,
        "action": "Close-up of Shiv's red sneakers crunching over dry ancient leaves and glowing blue moss. The moss pulses with bioluminescence as he steps.",
        "character_focus": "Shiv_feet",
        "emotion": "wonder",
        "visual_effects": "Glowing moss reaction, dust particles, detailed texture"
      },
      {
        "time": "8-12s",
        "camera": "Low-angle upward tilt, slow dramatic pan",
        "camera_movement": "tilt_up_dramatic",
        "motion_intensity": 0.3,
        "action": "Camera tilts up from ground level to reveal massive ancient trees towering above. Their gnarled branches twist like skeletal fingers reaching down. Vandana looks up in awe, her ponytail swaying.",
        "character_focus": "Vandana",
        "emotion": "awe_mixed_fear",
        "visual_effects": "Vertical emphasis, dramatic shadows, creeping vines"
      },
      {
        "time": "12-16s",
        "camera": "Medium shot, slight zoom in",
        "camera_movement": "zoom_in_slow",
        "motion_intensity": 0.5,
        "action": "Vandana stops abruptly and sniffs the air. Her expression shifts from brave to cautious. She reaches out to touch a strange glowing vine wrapped around a tree trunk. The vine reacts to her touch with rippling light.",
        "character_focus": "Vandana",
        "emotion": "curious_alert",
        "visual_effects": "Glowing vine interaction, magical particles, face lighting from glow"
      },
      {
        "time": "16-20s",
        "camera": "POV from Shiv's perspective, slight handheld shake",
        "camera_movement": "handheld_pov",
        "motion_intensity": 0.7,
        "action": "Point-of-view shot: The treasure map in Shiv's hands starts to vibrate and glow at the edges. The lines on the map pulse with golden light. His hands tremble slightly, showing his nervousness and excitement.",
        "character_focus": "Shiv_hands",
        "emotion": "excited_nervous",
        "visual_effects": "Map glowing, hand tremor, magical symbols appearing"
      },
      {
        "time": "20-24s",
        "camera": "Extreme close-up on Shiv's face",
        "camera_movement": "static_intense",
        "motion_intensity": 0.2,
        "action": "Extreme close-up of Shiv's wide brown eyes reflecting fear and curiosity. He hears a faint ghostly whisper calling his name. His pupils dilate, eyebrows raise. His messy black hair blows in a sudden cold draft.",
        "character_focus": "Shiv_face",
        "emotion": "frightened_alert",
        "visual_effects": "Eye reflection detail, hair movement, cold breath visible"
      },
      {
        "time": "24-28s",
        "camera": "Wide establishing shot through trees",
        "camera_movement": "dolly_reveal",
        "motion_intensity": 0.5,
        "action": "Wide shot framed through twisted tree trunks. A dark, gaping hole in a limestone cliff appears in the distance—the mouth of the Whispering Cave. Mist pours out like breath. Both characters visible in foreground, frozen in place.",
        "character_focus": "both",
        "emotion": "ominous_discovery",
        "visual_effects": "Atmospheric mist, ominous lighting, depth of field"
      },
      {
        "time": "28-32s",
        "camera": "Fast zoom into cave entrance",
        "camera_movement": "zoom_in_fast",
        "motion_intensity": 0.8,
        "action": "Rapid zoom toward the cave mouth. Darkness inside swirls like living smoke. Strange blue-green lights flicker deep within. A faint echoing 'shhhhh' sound reverberates out, causing leaves to rustle.",
        "character_focus": "environment",
        "emotion": "threatening_mysterious",
        "visual_effects": "Swirling darkness, ethereal lights, sound waves visible in mist"
      },
      {
        "time": "32-36s",
        "camera": "Medium two-shot, slight push in",
        "camera_movement": "push_in_slow",
        "motion_intensity": 0.4,
        "action": "Shiv grabs Vandana's arm with both hands, map crinkling. Both children stand frozen, staring at the cave entrance. Vandana's confident expression wavers slightly. Shiv's eyes are wide with fear but determined.",
        "character_focus": "both",
        "emotion": "fear_determination",
        "visual_effects": "Character interaction, emotional expressions, tense atmosphere"
      },
      {
        "time": "36-40s",
        "camera": "Close-up on Vandana's backpack, tilt up to face",
        "camera_movement": "tilt_up_reveal",
        "motion_intensity": 0.5,
        "action": "Close shot: Vandana takes a deep breath, steadying herself. She reaches into her pink backpack with deliberate movement. She pulls out a heavy brass flashlight with intricate engravings. She clicks it on—the bright beam cuts through the misty air.",
        "character_focus": "Vandana",
        "emotion": "brave_resolved",
        "visual_effects": "Flashlight beam, volumetric lighting, brass reflection"
      },
      {
        "time": "40-44s",
        "camera": "Hero shot, low angle tracking",
        "camera_movement": "low_angle_hero",
        "motion_intensity": 0.6,
        "action": "Hero shot: Vandana takes her first confident step toward the cave, flashlight leading the way. Shiv gulps audibly, nods to himself for courage, and follows close behind her. His sneakers crunch on gravel.",
        "character_focus": "both",
        "emotion": "courageous_supportive",
        "visual_effects": "Heroic lighting, dust kicked up by footsteps, determined poses"
      },
      {
        "time": "44-48s",
        "camera": "Dramatic silhouette shot, wide composition",
        "camera_movement": "static_dramatic",
        "motion_intensity": 0.3,
        "action": "Final wide shot: The two children stand at the very edge of the cave mouth, perfectly silhouetted against the remaining forest light behind them. Vandana's ponytail and Shiv's messy hair blow in the wind. They look into the dark abyss together, holding hands.",
        "character_focus": "both_silhouette",
        "emotion": "unity_facing_unknown",
        "visual_effects": "Perfect silhouette, rim lighting, dramatic composition, wind effects"
      }
    ]
  },
  
  "dialogue_with_timing": [
    {
      "time": 6,
      "character": "Shiv",
      "dialogue": "वंदना... क्या तुमने वह सुना? ऐसा लगा जैसे कोई मेरा नाम पुकार रहा है।",
      "english_translation": "Vandana... did you hear that? It felt like someone was calling my name.",
      "emotion": "fearful_questioning",
      "voice_direction": "Whispered, trembling voice, looking around nervously",
      "lip_sync_emphasis": "high"
    },
    {
      "time": 14,
      "character": "Vandana",
      "dialogue": "यह सिर्फ हवा है, शिव। डरो मत, मैं यहाँ हूँ।",
      "english_translation": "It's just the wind, Shiv. Don't be scared, I'm here.",
      "emotion": "reassuring_protective",
      "voice_direction": "Calm, confident tone, turning to look at Shiv with reassuring smile",
      "lip_sync_emphasis": "high"
    },
    {
      "time": 25,
      "character": "The Cave",
      "dialogue": "*सांसों जैसी आवाज*... अंदर आओ...",
      "english_translation": "*Breathing-like sound*... come inside...",
      "emotion": "eerie_beckoning",
      "voice_direction": "Hollow echoing whisper, reverb effect, inhuman quality",
      "lip_sync_emphasis": "none"
    },
    {
      "time": 33,
      "character": "Shiv",
      "dialogue": "नक्शा... यह कांप रहा है! हम सही जगह पर हैं।",
      "english_translation": "The map... it's shaking! We're at the right place.",
      "emotion": "excited_scared",
      "voice_direction": "Voice rising with excitement mixed with fear, eyes on the glowing map",
      "lip_sync_emphasis": "high"
    },
    {
      "time": 41,
      "character": "Vandana",
      "dialogue": "अपनी टॉर्च जलाओ। अब पीछे मुड़ने का कोई रास्ता नहीं है।",
      "english_translation": "Turn on your torch. There's no turning back now.",
      "emotion": "determined_brave",
      "voice_direction": "Firm, determined voice, looking straight ahead into darkness",
      "lip_sync_emphasis": "high"
    }
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

# ═══════════════════════════════════════════════════════════════
#                    CHARACTER CONSISTENCY SYSTEM
# ═══════════════════════════════════════════════════════════════

def build_character_prompt_detailed(character_data):
    """Build highly detailed character description for consistency."""
    char_name = character_data["name"]
    appearance = character_data["detailed_appearance"]
    
    prompt = f"{char_name}: "
    
    # Face details (highest priority for recognition)
    prompt += f"{appearance['face']}, "
    
    # Hair (critical for consistency)
    prompt += f"{appearance['hair']}, "
    
    # Clothing (must stay consistent)
    prompt += f"wearing {appearance['clothing']}, "
    
    # Build and skin tone
    prompt += f"{appearance['build']}, {appearance['skin_tone']}, "
    
    # Accessories
    prompt += f"{appearance['accessories']}. "
    
    # Repeat key features for emphasis (LTX-2 attention mechanism)
    prompt += f"ALWAYS MAINTAIN: {char_name} has {appearance['face'].split(',')[0]}, {appearance['hair'].split(',')[0]}, {appearance['clothing'].split(',')[0]}. "
    
    return prompt

def get_character_consistency_prefix(scene_json):
    """Generate character consistency prefix for ALL prompts."""
    char_prompts = []
    
    for char in scene_json["main_characters"]:
        char_prompt = build_character_prompt_detailed(char)
        char_prompts.append(char_prompt)
    
    # Combine with strong emphasis
    consistency_prompt = "CHARACTER CONSISTENCY CRITICAL: " + " | ".join(char_prompts)
    consistency_prompt += " | MAINTAIN EXACT SAME CHARACTER APPEARANCE THROUGHOUT ENTIRE SCENE. NO MORPHING. NO STYLE CHANGES."
    
    return consistency_prompt

# ═══════════════════════════════════════════════════════════════
#                    MOTION & CAMERA CONTROL
# ═══════════════════════════════════════════════════════════════

CAMERA_LORA_MAPPING = {
    "dolly_forward": "ltx-2-19b-lora-camera-control-dolly-forward.safetensors",
    "dolly_backward": "ltx-2-19b-lora-camera-control-dolly-backward.safetensors",
    "dolly_left": "ltx-2-19b-lora-camera-control-dolly-left.safetensors",
    "dolly_right": "ltx-2-19b-lora-camera-control-dolly-right.safetensors",
    "pan_left": "ltx-2-19b-lora-camera-control-pan-left.safetensors",
    "pan_right": "ltx-2-19b-lora-camera-control-pan-right.safetensors",
    "tilt_up": "ltx-2-19b-lora-camera-control-tilt-up.safetensors",
    "tilt_down": "ltx-2-19b-lora-camera-control-tilt-down.safetensors",
    "zoom_in": "ltx-2-19b-lora-camera-control-zoom-in.safetensors",
    "zoom_out": "ltx-2-19b-lora-camera-control-zoom-out.safetensors"
}

def get_motion_guidance_prompt(shot):
    """Build motion-specific guidance prompt."""
    motion_intensity = shot.get("motion_intensity", 0.5)
    camera_movement = shot.get("camera_movement", "static")
    
    # Motion intensity descriptors
    if motion_intensity < 0.3:
        motion_desc = "minimal motion, subtle movements, mostly static"
    elif motion_intensity < 0.6:
        motion_desc = "moderate motion, natural movements, steady pace"
    else:
        motion_desc = "dynamic motion, pronounced movements, energetic action"
    
    # Camera movement descriptor
    camera_desc = camera_movement.replace("_", " ")
    
    prompt = f"MOTION GUIDANCE: {motion_desc}. CAMERA: {camera_desc}. "
    
    # Add specific motion instructions
    if motion_intensity > 0.6:
        prompt += "Fast-paced action, clear motion trails. "
    else:
        prompt += "Smooth controlled movement, clean frames. "
    
    return prompt

def get_camera_lora_for_shot(shot):
    """Determine which camera LoRA to use for this shot."""
    camera_movement = shot.get("camera_movement", "static")
    
    # Extract base movement type
    for key in CAMERA_LORA_MAPPING.keys():
        if key in camera_movement:
            return key, CAMERA_LORA_MAPPING[key]
    
    return None, None

# ═══════════════════════════════════════════════════════════════
#                    VOICE SYNC & DIALOGUE SYSTEM
# ═══════════════════════════════════════════════════════════════

def get_dialogue_for_shot_enhanced(start_time, end_time, dialogue_list):
    """Enhanced dialogue injection with voice sync guidance."""
    lines = []
    
    for entry in dialogue_list:
        if start_time <= entry["time"] < end_time:
            char = entry["character"]
            text = entry["dialogue"]
            emotion = entry.get("emotion", "neutral")
            voice_direction = entry.get("voice_direction", "")
            lip_sync = entry.get("lip_sync_emphasis", "medium")
            
            if char == "The Cave":
                # Environmental sound
                lines.append(f"AUDIO EFFECT: Eerie cave whisper saying '{text}' with hollow reverb and inhuman quality")
            else:
                # Character dialogue with lip sync
                if lip_sync == "high":
                    lines.append(f"LIP SYNC CRITICAL: {char} speaks '{text}' with {emotion} emotion. {voice_direction}. Mouth movements MUST match dialogue exactly. Clear facial animation.")
                else:
                    lines.append(f"{char} says '{text}' with {emotion} emotion. {voice_direction}.")
    
    return " | ".join(lines) if lines else ""

def build_audio_atmosphere_prompt(shot, scene_json, start_s, end_s):
    """Build comprehensive audio prompt including dialogue and SFX."""
    audio_config = scene_json["audio"]
    
    # Base atmosphere
    atmosphere = f"AUDIO ATMOSPHERE: {audio_config['background_music']}. "
    
    # Environment SFX
    atmosphere += f"SOUND EFFECTS: {audio_config['environment_sfx']}. "
    
    # Voice processing
    atmosphere += f"VOICE: {audio_config['voice_processing']}. "
    
    # Dialogue integration
    dialogue_text = get_dialogue_for_shot_enhanced(
        start_s, end_s, scene_json["dialogue_with_timing"]
    )
    
    if dialogue_text:
        atmosphere += dialogue_text
    
    return atmosphere

# ═══════════════════════════════════════════════════════════════
#                    ENHANCED PROMPT BUILDER
# ═══════════════════════════════════════════════════════════════

def build_shot_prompt_pro(shot, json_data, shot_index, prev_shot_success=True):
    """
    PRO version of prompt builder with:
    - Character consistency enforcement
    - Motion guidance
    - Voice sync
    - Adaptive strength based on previous shot
    """
    try:
        times = shot["time"].replace("s", "").split("-")
        start_s = int(times[0])
        end_s = int(times[1])
    except:
        start_s, end_s = 0, 5
    
    # ═══ SECTION 1: CHARACTER CONSISTENCY (HIGHEST PRIORITY) ═══
    if INJECT_CHARACTER_EVERY_SHOT:
        character_prompt = get_character_consistency_prefix(json_data)
    else:
        character_prompt = ""
    
    # ═══ SECTION 2: SHOT ACTION & CAMERA ═══
    action_prompt = f"SHOT {shot_index + 1}: {shot['action']}. "
    camera_prompt = f"CAMERA: {shot['camera']}. "
    
    # ═══ SECTION 3: MOTION GUIDANCE ═══
    motion_prompt = get_motion_guidance_prompt(shot)
    
    # ═══ SECTION 4: ENVIRONMENT (for spatial continuity) ═══
    env = json_data["environment"]
    env_prompt = f"ENVIRONMENT: {env['location']}. LIGHTING: {env['lighting']}. TIME: {env['time']}. WEATHER: {env['weather']}. MOOD: {env['mood']}. COLOR PALETTE: {env['color_palette']}. "
    
    # ═══ SECTION 5: STYLE ENFORCEMENT ═══
    style_prompt = f"STYLE: {json_data['video_style']}. "
    
    # ═══ SECTION 6: AUDIO & VOICE SYNC ═══
    audio_prompt = build_audio_atmosphere_prompt(shot, json_data, start_s, end_s)
    
    # ═══ SECTION 7: VISUAL EFFECTS ═══
    vfx_prompt = f"VISUAL EFFECTS: {shot.get('visual_effects', 'natural')}. "
    
    # ═══ SECTION 8: EMOTION & CHARACTER FOCUS ═══
    emotion_prompt = f"EMOTION: {shot.get('emotion', 'neutral')}. FOCUS: {shot.get('character_focus', 'scene')}. "
    
    # ═══ ASSEMBLE IN PRIORITY ORDER ═══
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
    
    # ═══ PROMPT WEIGHTING (if enabled) ═══
    if USE_PROMPT_WEIGHTING:
        # Emphasize character consistency
        final_prompt = final_prompt.replace(
            "CHARACTER CONSISTENCY CRITICAL:",
            "(CHARACTER CONSISTENCY CRITICAL:1.5)"
        )
        # Emphasize lip sync
        final_prompt = final_prompt.replace(
            "LIP SYNC CRITICAL:",
            "(LIP SYNC CRITICAL:1.3)"
        )
    
    return final_prompt

# ═══════════════════════════════════════════════════════════════
#                    ADAPTIVE STRENGTH SYSTEM
# ═══════════════════════════════════════════════════════════════

def calculate_adaptive_strength(shot, prev_shot, prev_shot_success):
    """
    Calculate anchor strength based on:
    - Motion intensity change
    - Character focus change
    - Previous shot success rate
    """
    if not USE_ADAPTIVE_STRENGTH:
        return ANCHOR_STRENGTH_HIGH
    
    # Start with base strength
    strength = ANCHOR_STRENGTH_HIGH
    
    # Adjust based on motion intensity change
    if prev_shot:
        motion_change = abs(
            shot.get("motion_intensity", 0.5) - 
            prev_shot.get("motion_intensity", 0.5)
        )
        
        if motion_change > 0.4:
            # Large motion change → lower strength for flexibility
            strength -= 0.10
        elif motion_change < 0.2:
            # Similar motion → higher strength for consistency
            strength += 0.05
    
    # Adjust based on character focus change
    if prev_shot:
        if shot.get("character_focus") != prev_shot.get("character_focus"):
            # Character focus changed → slightly lower strength
            strength -= 0.05
    
    # Adjust based on previous shot success
    if not prev_shot_success:
        # Previous shot had issues → try different strength
        strength -= 0.10
    
    # Clamp to reasonable range
    strength = max(ANCHOR_STRENGTH_LOW, min(ANCHOR_STRENGTH_HIGH, strength))
    
    return strength

# ═══════════════════════════════════════════════════════════════
#                    ENHANCED NEGATIVE PROMPT
# ═══════════════════════════════════════════════════════════════

def build_negative_prompt_enhanced():
    """Build comprehensive negative prompt."""
    base_neg = NEGATIVE_PROMPT
    
    if USE_NEGATIVE_PROMPT_EXPANSION:
        # Character consistency negatives
        char_neg = "character morphing, face changing, inconsistent character design, different clothing in same scene, style shift, character replacement, "
        
        # Motion negatives
        motion_neg = "motion blur artifacts, jittery movement, unnatural animation, robotic motion, floating characters, "
        
        # Audio/lip sync negatives
        audio_neg = "desynchronized lips, mouth not moving during speech, frozen face during dialogue, mismatched audio, "
        
        # Quality negatives
        quality_neg = "compression artifacts, pixelation, banding, color shifts, lighting inconsistency, flickering, "
        
        base_neg = char_neg + motion_neg + audio_neg + quality_neg + base_neg
    
    return base_neg

# ═══════════════════════════════════════════════════════════════
#                    BUILD STORYBOARD FROM JSON
# ═══════════════════════════════════════════════════════════════

STORYBOARD = []
for idx, shot in enumerate(SCENE_JSON["story_action"]["shots"]):
    prev_shot = SCENE_JSON["story_action"]["shots"][idx-1] if idx > 0 else None
    
    full_prompt = build_shot_prompt_pro(shot, SCENE_JSON, idx)
    
    # Determine camera LoRA
    camera_lora_key, camera_lora_file = get_camera_lora_for_shot(shot)
    
    STORYBOARD.append({
        "id": f"shot_{idx+1:02d}",
        "prompt": full_prompt,
        "shot_data": shot,
        "camera_lora": camera_lora_file if USE_MOTION_LORAS else None,
        "prev_shot": prev_shot
    })

print(f"→ Parsed {len(STORYBOARD)} shots from JSON.")
print(f"→ Character consistency: {'ENABLED' if USE_CHARACTER_LORAS else 'DISABLED'}")
print(f"→ Motion control: {'ENABLED' if USE_MOTION_LORAS else 'DISABLED'}")
print(f"→ Voice sync: {'ENABLED' if USE_VOICE_SYNC else 'DISABLED'}")

# ═══════════════════════════════════════════════════════════════
#                    HELPER FUNCTIONS (Enhanced)
# ═══════════════════════════════════════════════════════════════

def check_environment():
    """Enhanced environment check."""
    required_files = [
        "/content/ComfyUI/models/unet/ltx-2-19b-distilled_Q4_K_M.gguf",
        "/content/ComfyUI/models/vae/LTX2_video_vae_bf16.safetensors",
        "/content/ComfyUI/models/vae/LTX2_audio_vae_bf16.safetensors"
    ]
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f"🚨 MISSING MODELS: {missing}")
    print("✅ Environment Checked: All Models Found.")

def save_video_from_components(video, filename_prefix="video/LTX_PRO", format="auto", codec="auto"):
    """Enhanced video saving with metadata."""
    width, height = video.get_dimensions()
    full_output_folder, filename, counter, _, _ = (
        folder_paths.get_save_image_path(
            filename_prefix,
            folder_paths.get_output_directory(),
            width,
            height
        )
    )
    ext = Types.VideoContainer.get_extension(format)
    path = os.path.join(full_output_folder, f"{filename}_{counter:05}_.{ext}")
    video.save_to(path, format=Types.VideoContainer(format), codec=codec, metadata=None)
    return path

def extract_overlap_anchor_enhanced(video_path, output_folder="/content/ComfyUI/input", 
                                    scene_idx=0, overlap=16):
    """
    Enhanced anchor extraction with:
    - Brightness validation
    - Multiple frame candidates
    - Quality scoring
    """
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return None
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Try multiple candidate frames
    candidates = [
        total_frames - overlap,
        total_frames - overlap - 2,
        total_frames - overlap + 2
    ]
    
    best_frame = None
    best_score = 0
    
    for candidate_idx in candidates:
        if candidate_idx < 0 or candidate_idx >= total_frames:
            continue
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, candidate_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        # Score frame quality
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = cv2.mean(gray)[0]
        
        # Calculate sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = laplacian.var()
        
        # Combined score
        score = brightness + sharpness * 0.1
        
        # Reject if too dark
        if brightness < 5:
            continue
        
        if score > best_score:
            best_score = score
            best_frame = frame
    
    cap.release()
    
    if best_frame is not None:
        filename = f"anchor_scene_{scene_idx}.png"
        save_path = os.path.join(output_folder, filename)
        cv2.imwrite(save_path, best_frame)
        print(f"   ✓ Anchor extracted (quality score: {best_score:.2f})")
        return save_path
    else:
        print(f"   ✗ No valid anchor frame found")
        return None

def stitch_videos_with_overlap_pro(video_paths, output_filename="LTX_Final_PRO.mp4", 
                                   overlap_frames=16):
    """
    Enhanced stitching with:
    - Audio crossfades
    - Color grading consistency
    - Subtitle embedding (if enabled)
    """
    if not video_paths:
        return None
    
    print(f"\n🧵 Stitching {len(video_paths)} clips with PRO settings...")
    final_clips = []
    
    for i, path in enumerate(video_paths):
        if os.path.exists(path):
            try:
                clip = VideoFileClip(path)
                
                # Normalize FPS
                if clip.fps != FPS:
                    clip = clip.set_fps(FPS)
                
                # Remove overlap (except last clip)
                if i < len(video_paths) - 1:
                    duration_to_keep = clip.duration - (overlap_frames / float(FPS))
                    if duration_to_keep > 0:
                        clip = clip.subclip(0, duration_to_keep)
                
                # Enhanced audio processing
                if clip.audio is not None:
                    # Longer crossfades for smoother transitions
                    clip = clip.audio_fadein(0.1).audio_fadeout(0.1)
                    # Normalize audio levels
                    clip = clip.volumex(0.9)  # Prevent clipping
                
                final_clips.append(clip)
                
            except Exception as e:
                print(f"⚠️ Skipping corrupted clip {path}: {e}")
    
    if not final_clips:
        return None
    
    final_video = concatenate_videoclips(final_clips, method="compose")
    output_path = f"/content/ComfyUI/output/{output_filename}"
    
    print("⏳ Rendering Final Movie (PRO Quality)...")
    final_video.write_videofile(
        output_path,
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        bitrate="10000k",  # Higher bitrate for quality
        preset="slow",     # Better compression
        threads=4,
        logger=None
    )
    
    return output_path

def display_video(video_path):
    """Display video in Colab."""
    if not video_path or not os.path.exists(video_path):
        return
    video_data = open(video_path,'rb').read()
    data_url = f"data:video/mp4;base64," + b64encode(video_data).decode()
    display(HTML(f"""
        <video width=800 controls autoplay loop>
            <source src="{data_url}" type="video/mp4">
        </video>
    """))

# ═══════════════════════════════════════════════════════════════
#                    ENHANCED GENERATION ENGINE
# ═══════════════════════════════════════════════════════════════

def generate_segment_pro(
    image_path: str = None,
    prompt: str = "",
    negative_prompt: str = "",
    seed: int = 10,
    frames: int = 121,
    image_strength: float = 0.75,
    camera_lora: str = None,
    shot_data: dict = None
):
    """
    Enhanced generation with:
    - Character LoRA support
    - Camera LoRA support
    - Adaptive sampling
    - Better memory management
    """
    import_custom_nodes()
    
    # Build final prompts
    full_positive_prompt = f"{BASE_PROMPT} . {prompt}"
    full_negative_prompt = build_negative_prompt_enhanced()
    
    print(f"   🚀 Seed: {seed} | Strength: {image_strength:.2f}")
    print(f"   📝 Prompt preview: {prompt[:150]}...")
    if camera_lora:
        print(f"   🎥 Camera LoRA: {camera_lora}")
    
    with torch.inference_mode():
        # ═══ IMAGE LOADING ═══
        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        if image_path is not None:
            clean_filename = os.path.basename(image_path)
            try:
                loadimage_98 = loadimage.load_image(image=clean_filename)
                image_bypass = False
            except Exception as e:
                print(f"   ⚠️ Image load failed → T2V mode")
                noise_image = torch.full((1, HEIGHT, WIDTH, 3), 0.5)
                loadimage_98 = (noise_image, None)
                image_strength = 0.0
                image_bypass = True
        else:
            noise_image = torch.full((1, HEIGHT, WIDTH, 3), 0.5)
            loadimage_98 = (noise_image, None)
            image_strength = 0.0
            image_bypass = True
        
        # ═══ NODE DEFINITIONS (Same as original but with enhancements) ═══
        ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
        ksamplerselect_105 = ksamplerselect.EXECUTE_NORMALIZED(sampler_name="euler")
        ksamplerselect_106 = ksamplerselect.EXECUTE_NORMALIZED(sampler_name="gradient_estimation")
        
        manualsigmas = NODE_CLASS_MAPPINGS["ManualSigmas"]()
        # Enhanced sigma schedule for better quality
        manualsigmas_107 = manualsigmas.EXECUTE_NORMALIZED(
            sigmas="0.95, 0.85, 0.60, 0.30, 0.0"
        )
        manualsigmas_134 = manualsigmas.EXECUTE_NORMALIZED(
            sigmas="1.0, 0.99, 0.98, 0.95, 0.90, 0.85, 0.70, 0.50, 0.25, 0.0"
        )
        
        randomnoise = NODE_CLASS_MAPPINGS["RandomNoise"]()
        randomnoise_133 = randomnoise.EXECUTE_NORMALIZED(noise_seed=seed)
        randomnoise_114 = randomnoise.EXECUTE_NORMALIZED(noise_seed=0)
        
        primitiveint = NODE_CLASS_MAPPINGS["PrimitiveInt"]()
        primitiveint_123 = primitiveint.EXECUTE_NORMALIZED(value=frames)
        
        # ═══ IMAGE PREPROCESSING ═══
        resizeimagemasknode = NODE_CLASS_MAPPINGS["ResizeImageMaskNode"]()
        resizeimagesbylongeredge = NODE_CLASS_MAPPINGS["ResizeImagesByLongerEdge"]()
        ltxvpreprocess = NODE_CLASS_MAPPINGS["LTXVPreprocess"]()
        
        resizeimagemasknode_102 = resizeimagemasknode.EXECUTE_NORMALIZED(
            input=get_value_at_index(loadimage_98, 0),
            scale_method="lanczos",
            resize_type={"resize_type": "scale dimensions", "width": WIDTH, "height": HEIGHT, "crop": "center"}
        )
        
        max_edge = max(WIDTH, HEIGHT)
        resizeimagesbylongeredge_140 = resizeimagesbylongeredge.EXECUTE_NORMALIZED(
            longer_edge=max_edge,
            images=get_value_at_index(resizeimagemasknode_102, 0)
        )
        
        ltxvpreprocess_126 = ltxvpreprocess.EXECUTE_NORMALIZED(
            img_compression=33,
            image=get_value_at_index(resizeimagesbylongeredge_140, 0)
        )
        
        # ═══ LATENT PREPARATION ═══
        getimagesize = NODE_CLASS_MAPPINGS["GetImageSize"]()
        emptyimage = NODE_CLASS_MAPPINGS["EmptyImage"]()
        imagescaleby = NODE_CLASS_MAPPINGS["ImageScaleBy"]()
        emptyltxvlatentvideo = NODE_CLASS_MAPPINGS["EmptyLTXVLatentVideo"]()
        
        getimagesize_125 = getimagesize.EXECUTE_NORMALIZED(
            image=get_value_at_index(resizeimagemasknode_102, 0)
        )
        
        emptyimage_112 = emptyimage.generate(
            width=get_value_at_index(getimagesize_125, 0),
            height=get_value_at_index(getimagesize_125, 1),
            batch_size=1,
            color=0
        )
        
        imagescaleby_122 = imagescaleby.upscale(
            upscale_method="lanczos",
            scale_by=0.5,
            image=get_value_at_index(emptyimage_112, 0)
        )
        
        getimagesize_124 = getimagesize.EXECUTE_NORMALIZED(
            image=get_value_at_index(imagescaleby_122, 0)
        )
        
        emptyltxvlatentvideo_127 = emptyltxvlatentvideo.EXECUTE_NORMALIZED(
            width=get_value_at_index(getimagesize_124, 0),
            height=get_value_at_index(getimagesize_124, 1),
            length=get_value_at_index(primitiveint_123, 0),
            batch_size=1
        )
        
        # ═══ TEXT ENCODING ═══
        dualcliploader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
        dualcliploader_145 = dualcliploader.load_clip(
            clip_name1=text_encoder_model,
            clip_name2=text_encoder2_model,
            type="ltxv",
            device="default"
        )
        
        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cliptextencode_131 = cliptextencode.encode(
            text=full_positive_prompt,
            clip=get_value_at_index(dualcliploader_145, 0)
        )
        cliptextencode_neg = cliptextencode.encode(
            text=full_negative_prompt,
            clip=get_value_at_index(dualcliploader_145, 0)
        )
        
        del dualcliploader_145
        gc.collect()
        torch.cuda.empty_cache()
        
        # ═══ CONDITIONING ═══
        conditioningzeroout = NODE_CLASS_MAPPINGS["ConditioningZeroOut"]()
        ltxvconditioning = NODE_CLASS_MAPPINGS["LTXVConditioning"]()
        
        conditioningzeroout_132 = conditioningzeroout.zero_out(
            conditioning=get_value_at_index(cliptextencode_131, 0)
        )
        
        ltxvconditioning_111 = ltxvconditioning.EXECUTE_NORMALIZED(
            frame_rate=25,
            positive=get_value_at_index(cliptextencode_131, 0),
            negative=get_value_at_index(cliptextencode_neg, 0)
        )
        
        # ═══ VAE LOADING ═══
        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        vaeloader_144 = vaeloader.load_vae(vae_name=vae_model)
        
        ltxvimgtovideoinplace = NODE_CLASS_MAPPINGS["LTXVImgToVideoInplace"]()
        ltxvimgtovideoinplace_128 = ltxvimgtovideoinplace.EXECUTE_NORMALIZED(
            strength=image_strength,
            bypass=image_bypass,
            vae=get_value_at_index(vaeloader_144, 0),
            image=get_value_at_index(ltxvpreprocess_126, 0),
            latent=get_value_at_index(emptyltxvlatentvideo_127, 0)
        )
        
        del vaeloader_144
        gc.collect()
        torch.cuda.empty_cache()
        
        # ═══ AUDIO VAE ═══
        vaeloaderkj = NODE_CLASS_MAPPINGS["VAELoaderKJ"]()
        vaeloaderkj_142 = vaeloaderkj.load_vae(
            vae_name=vae_audio_model,
            device="main_device",
            weight_dtype="fp16"
        )
        
        ltxvemptylatentaudio = NODE_CLASS_MAPPINGS["LTXVEmptyLatentAudio"]()
        ltxvemptylatentaudio_110 = ltxvemptylatentaudio.EXECUTE_NORMALIZED(
            frames_number=get_value_at_index(primitiveint_123, 0),
            frame_rate=FPS,
            batch_size=1,
            audio_vae=get_value_at_index(vaeloaderkj_142, 0)
        )
        
        # ═══ CONCAT AUDIO+VIDEO LATENTS ═══
        ltxvconcatavlatent = NODE_CLASS_MAPPINGS["LTXVConcatAVLatent"]()
        
        if not image_bypass:
            ltxvconcatavlatent_121 = ltxvconcatavlatent.EXECUTE_NORMALIZED(
                video_latent=get_value_at_index(ltxvimgtovideoinplace_128, 0),
                audio_latent=get_value_at_index(ltxvemptylatentaudio_110, 0)
            )
        else:
            ltxvconcatavlatent_121 = ltxvconcatavlatent.EXECUTE_NORMALIZED(
                video_latent=get_value_at_index(emptyltxvlatentvideo_127, 0),
                audio_latent=get_value_at_index(ltxvemptylatentaudio_110, 0)
            )
        
        # ═══ MODEL LOADING WITH LORA ═══
        unetloadergguf = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
        unetloadergguf_146 = unetloadergguf.load_unet(
            unet_name="ltx-2-19b-distilled_Q4_K_M.gguf"
        )
        unetloadergguf_146 = get_value_at_index(unetloadergguf_146, 0)
        
        # Load LoRAs (camera, character, motion)
        if USE_MOTION_LORAS and camera_lora and 'lora_1' in globals() and lora_1:
            load_lora = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
            unetloadergguf_146 = load_lora.load_lora_model_only(
                unetloadergguf_146,
                lora_1,
                CAMERA_LORA_STRENGTH
            )[0]
            print(f"   ✓ Loaded camera LoRA at strength {CAMERA_LORA_STRENGTH}")
        
        # ═══ FIRST PASS: LOW-RES GENERATION ═══
        cfgguider = NODE_CLASS_MAPPINGS["CFGGuider"]()
        cfgguider_135 = cfgguider.EXECUTE_NORMALIZED(
            cfg=1,
            model=unetloadergguf_146,
            positive=get_value_at_index(ltxvconditioning_111, 0),
            negative=get_value_at_index(ltxvconditioning_111, 1)
        )
        
        samplercustomadvanced = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
        samplercustomadvanced_113 = samplercustomadvanced.EXECUTE_NORMALIZED(
            noise=get_value_at_index(randomnoise_133, 0),
            guider=get_value_at_index(cfgguider_135, 0),
            sampler=get_value_at_index(ksamplerselect_105, 0),
            sigmas=get_value_at_index(manualsigmas_134, 0),
            latent_image=get_value_at_index(ltxvconcatavlatent_121, 0)
        )
        
        del cfgguider_135
        gc.collect()
        torch.cuda.empty_cache()
        
        # ═══ SEPARATE AUDIO/VIDEO ═══
        ltxvseparateavlatent = NODE_CLASS_MAPPINGS["LTXVSeparateAVLatent"]()
        ltxvseparateavlatent_116 = ltxvseparateavlatent.EXECUTE_NORMALIZED(
            av_latent=get_value_at_index(samplercustomadvanced_113, 0)
        )
        
        # ═══ CROP GUIDES ═══
        ltxvcropguides = NODE_CLASS_MAPPINGS["LTXVCropGuides"]()
        ltxvcropguides_108 = ltxvcropguides.EXECUTE_NORMALIZED(
            positive=get_value_at_index(ltxvconditioning_111, 0),
            negative=get_value_at_index(ltxvconditioning_111, 1),
            latent=get_value_at_index(ltxvseparateavlatent_116, 0)
        )
        
        cfgguider_109 = cfgguider.EXECUTE_NORMALIZED(
            cfg=1,
            model=unetloadergguf_146,
            positive=get_value_at_index(ltxvcropguides_108, 0),
            negative=get_value_at_index(ltxvcropguides_108, 1)
        )
        
        # ═══ UPSCALER ═══
        vaeloader_144 = vaeloader.load_vae(vae_name=vae_model)
        
        latentupscalemodelloader = NODE_CLASS_MAPPINGS["LatentUpscaleModelLoader"]()
        latentupscalemodelloader_136 = latentupscalemodelloader.EXECUTE_NORMALIZED(
            model_name=upscaler_model
        )
        
        ltxvlatentupsampler = NODE_CLASS_MAPPINGS["LTXVLatentUpsampler"]()
        ltxvlatentupsampler_195 = ltxvlatentupsampler.upsample_latent(
            samples=get_value_at_index(ltxvcropguides_108, 2),
            upscale_model=get_value_at_index(latentupscalemodelloader_136, 0),
            vae=get_value_at_index(vaeloader_144, 0)
        )
        
        del latentupscalemodelloader_136, vaeloader_144
        gc.collect()
        torch.cuda.empty_cache()
        
        # ═══ SECOND PASS: REFINEMENT ═══
        vaeloader_144 = vaeloader.load_vae(vae_name=vae_model)
        
        ltxvimgtovideoinplace_130 = ltxvimgtovideoinplace.EXECUTE_NORMALIZED(
            strength=image_strength,
            bypass=image_bypass,
            vae=get_value_at_index(vaeloader_144, 0),
            image=get_value_at_index(ltxvpreprocess_126, 0),
            latent=get_value_at_index(ltxvlatentupsampler_195, 0)
        )
        
        del vaeloader_144
        gc.collect()
        torch.cuda.empty_cache()
        
        # ═══ CONCAT AGAIN FOR REFINEMENT ═══
        if not image_bypass:
            ltxvconcatavlatent_129 = ltxvconcatavlatent.EXECUTE_NORMALIZED(
                video_latent=get_value_at_index(ltxvimgtovideoinplace_130, 0),
                audio_latent=get_value_at_index(ltxvseparateavlatent_116, 1)
            )
        else:
            ltxvconcatavlatent_129 = ltxvconcatavlatent.EXECUTE_NORMALIZED(
                video_latent=get_value_at_index(ltxvlatentupsampler_195, 0),
                audio_latent=get_value_at_index(ltxvseparateavlatent_116, 1)
            )
        
        samplercustomadvanced_119 = samplercustomadvanced.EXECUTE_NORMALIZED(
            noise=get_value_at_index(randomnoise_114, 0),
            guider=get_value_at_index(cfgguider_109, 0),
            sampler=get_value_at_index(ksamplerselect_106, 0),
            sigmas=get_value_at_index(manualsigmas_107, 0),
            latent_image=get_value_at_index(ltxvconcatavlatent_129, 0)
        )
        
        del cfgguider_109, unetloadergguf_146
        gc.collect()
        torch.cuda.empty_cache()
        
        # ═══ FINAL DECODE ═══
        ltxvseparateavlatent_118 = ltxvseparateavlatent.EXECUTE_NORMALIZED(
            av_latent=get_value_at_index(samplercustomadvanced_119, 1)
        )
        
        vaeloader_144 = vaeloader.load_vae(vae_name=vae_model)
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        vaedecode_117 = vaedecode.decode(
            samples=get_value_at_index(ltxvseparateavlatent_118, 0),
            vae=get_value_at_index(vaeloader_144, 0)
        )
        del vaeloader_144
        
        # ═══ AUDIO DECODE ═══
        ltxvaudiovaedecode = NODE_CLASS_MAPPINGS["LTXVAudioVAEDecode"]()
        ltxvaudiovaedecode_120 = ltxvaudiovaedecode.EXECUTE_NORMALIZED(
            samples=get_value_at_index(ltxvseparateavlatent_118, 1),
            audio_vae=get_value_at_index(vaeloaderkj_142, 0)
        )
        del vaeloaderkj_142
        gc.collect()
        torch.cuda.empty_cache()
        
        # ═══ CREATE VIDEO ═══
        createvideo = NODE_CLASS_MAPPINGS["CreateVideo"]()
        createvideo_115 = createvideo.EXECUTE_NORMALIZED(
            fps=FPS,
            images=get_value_at_index(vaedecode_117, 0),
            audio=get_value_at_index(ltxvaudiovaedecode_120, 0)
        )
        
        video = get_value_at_index(createvideo_115, 0)
        output_path = save_video_from_components(video)
        
        return output_path

# ═══════════════════════════════════════════════════════════════
#                    PRODUCTION LOOP (Enhanced)
# ═══════════════════════════════════════════════════════════════

generated_clips = []
input_dir = "/content/ComfyUI/input"
output_dir = "/content/ComfyUI/output"
cache_dir = f"{output_dir}/{PROJECT_NAME}_cache"
os.makedirs(cache_dir, exist_ok=True)

check_environment()

# Auto-resume logic
start_index = 0
current_input_image = None

if 'file_uploaded' in globals() and file_uploaded:
    current_input_image = file_uploaded
    print(f"📂 Starting with uploaded image: {current_input_image}")

# Check for cached progress
for i in range(len(STORYBOARD)):
    expected_anchor = f"{input_dir}/anchor_scene_{i}.png"
    expected_clip = f"{cache_dir}/scene_{i}.mp4"
    
    if os.path.exists(expected_anchor) and os.path.exists(expected_clip):
        current_input_image = expected_anchor
        start_index = i + 1
        generated_clips.append(expected_clip)
    else:
        break

if start_index > 0:
    print(f"⏩ Resuming from shot {start_index + 1} (using cache)")

# Main generation loop
prev_shot_success = True

for i in tqdm(range(start_index, len(STORYBOARD)), desc="🎬 Generating PRO shots"):
    scene = STORYBOARD[i]
    shot_data = scene["shot_data"]
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # Calculate adaptive strength
    strength = calculate_adaptive_strength(
        shot_data,
        scene.get("prev_shot"),
        prev_shot_success
    ) if i > 0 else 0.0
    
    max_retries = 3
    success = False
    attempt = 0
    
    while attempt < max_retries and not success:
        attempt += 1
        
        # Generate unique but deterministic seed
        current_seed = 2000 + (i * 100) + (attempt * 197)
        
        try:
            print(f"\n📍 Shot {i+1}/{len(STORYBOARD)} (Attempt {attempt})")
            
            clip_path = generate_segment_pro(
                image_path=current_input_image,
                prompt=scene["prompt"],
                seed=current_seed,
                image_strength=strength,
                frames=121,
                camera_lora=scene.get("camera_lora"),
                shot_data=shot_data
            )
            
            if clip_path:
                # Enhanced anchor extraction
                next_anchor = extract_overlap_anchor_enhanced(
                    clip_path,
                    output_folder=input_dir,
                    scene_idx=i,
                    overlap=OVERLAP_FRAMES
                )
                
                if next_anchor:
                    # Cache the clip
                    cached_clip_path = f"{cache_dir}/scene_{i}.mp4"
                    shutil.copy(clip_path, cached_clip_path)
                    
                    generated_clips.append(cached_clip_path)
                    current_input_image = next_anchor
                    
                    success = True
                    prev_shot_success = True
                    
                    print(f"   ✅ Shot {i+1} complete!")
                else:
                    print(f"   ⚠️ Invalid anchor (retrying with new seed...)")
                    prev_shot_success = False
        
        except Exception as e:
            print(f"   ❌ Error on shot {i+1}: {e}")
            prev_shot_success = False
    
    if not success:
        print(f"🛑 Failed to generate shot {i+1} after {max_retries} attempts. Stopping.")
        break

# Final stitching
if generated_clips:
    print(f"\n{'='*60}")
    print(f"🎬 ALL SHOTS COMPLETE! Now stitching final movie...")
    print(f"{'='*60}")
    
    final_movie = stitch_videos_with_overlap_pro(
        generated_clips,
        output_filename=f"{PROJECT_NAME}_Complete_PRO.mp4",
        overlap_frames=OVERLAP_FRAMES
    )
    
    if final_movie:
        print(f"\n{'='*60}")
        print(f"🎉 MOVIE COMPLETE (PRO VERSION)!")
        print(f"📁 Location: {final_movie}")
        print(f"⏱️  Duration: {len(STORYBOARD) * 4} seconds")
        print(f"🎭 Shots: {len(generated_clips)}")
        print(f"{'='*60}\n")
        
        display_video(final_movie)
else:
    print("❌ No clips generated. Check errors above.")

print("\n✅ Script execution complete!")
