#!/usr/bin/env python3
"""
video_sub_translator_chutes.py
──────────────────────────────────────────────────────────
All-in-one tool that …

1. Extracts the first English subtitle track from a video
2. Translates it to Castilian Spanish with Chutes AI (DeepSeek)
3. Saves the translated Spanish SRT file alongside the video (deleting any pre-existing SRTs for that video).

Runs in batch from the command line **or** via a tiny Tk GUI.
"""

import argparse
import os
import sys
import tempfile
import subprocess
import shutil
import threading
import asyncio
import json
import re
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
try:
    import ttkbootstrap as ttk
    from ttkbootstrap import Style
    from ttkbootstrap.constants import *
    from ttkbootstrap.tooltip import ToolTip
    from ttkbootstrap.dialogs import Messagebox
    MODERN_UI_AVAILABLE = True
except ImportError:
    import tkinter.ttk as ttk
    MODERN_UI_AVAILABLE = False
from textwrap import dedent
from collections import deque
import random

import pysrt
import httpx  # For Chutes API
from tqdm.asyncio import tqdm

# ─────────────────────────────────────────────────────────
#  Chutes AI Configuration
# ─────────────────────────────────────────────────────────

def load_chutes_api_key() -> str:
    """Load the Chutes API key from environment or config file."""
    key = os.getenv("CHUTES_API_KEY")
    if key:
        return key

    config_path = Path(__file__).with_name("chutes_config.json")
    if config_path.exists():
        try:
            with open(config_path) as f:
                data = json.load(f)
            key = data.get("CHUTES_API_KEY")
            if key:
                return key
        except json.JSONDecodeError:
            pass

    raise SystemExit(
        "ERROR: CHUTES_API_KEY is missing. Set the CHUTES_API_KEY environment variable "
        "or provide it in chutes_config.json"
    )


CHUTES_API_KEY = load_chutes_api_key()
CHUTES_API_ENDPOINT = "https://llm.chutes.ai/v1/chat/completions"
CHUTES_MODEL_ID = "deepseek-ai/DeepSeek-V3-0324"

# General Configuration
TARGET_LANG            = "es-ES"
AVG_TOKENS_PER_LINE    = 30
IDEAL_TOKENS_PER_BATCH = 15_000
PROMPT_OVERHEAD_TOKENS = 200
FFMPEG_LOGLEVEL        = "error"
MAX_RPM                = 100
API_TIMEOUT_MAIN       = 180
API_TIMEOUT_RETRY_LINE = 60
MAX_OUTPUT_TOKENS_BATCH= 8000
MAX_OUTPUT_TOKENS_SINGLE=500
TRANSLATION_TEMPERATURE= 0.1
CONTEXT_WINDOW_SIZE     = 3

CHUTES_BATCH_SYSTEM_PROMPT = dedent(f"""
    You are an automated JSON translation service for subtitle content.
    Your SOLE function is to translate English text to {TARGET_LANG}.
    
    The user will provide:
    1. CONTEXT LINES (if any): Previous subtitle lines for narrative continuity - DO NOT translate these
    2. LINES TO TRANSLATE: The actual lines you must translate
    
    Your response MUST be ONLY a valid JSON array of strings, with no other text, thoughts, or markdown.
    The JSON array must contain ONLY the translations of the "LINES TO TRANSLATE" section.
    
    IMPORTANT: Use the context lines to maintain dialogue continuity, character consistency, and narrative flow.
    Consider:
    - Character names and how they're addressed
    - Ongoing conversations and references
    - Emotional tone and speaking style consistency
    - References to previous dialogue or events
    
    CRITICAL NAME PRESERVATION:
    - Text may contain placeholders like __NAME_0__, __NAME_1__, etc.
    - These represent proper names (people, places, organizations)
    - NEVER translate these placeholders - keep them exactly as they appear
    - The placeholders will be restored to original names after translation
    
    Example interaction:
    User: "CONTEXT: ['__NAME_0__ walked into the room.', '__NAME_1__ looked surprised.']
    TRANSLATE: ['Hello there, __NAME_0__!', 'I missed you so much.']"
    
    Your response: ["¡Hola, __NAME_0__!", "Te eché mucho de menos."]
    
    CRITICAL INSTRUCTIONS:
    1. ABSOLUTELY NO EXTRA TEXT. Your entire response content must be only the JSON array.
    2. DO NOT translate context lines - they are for reference only.
    3. Preserve all original HTML-like markup (e.g., <i>, </i>) and bracketed text (e.g., [noise], (speaker)).
    4. NEVER translate name placeholders (__NAME_X__) - keep them exactly as they appear.
    5. Ensure the output is a syntactically valid JSON array of strings.
    6. DO NOT THINK OUT LOUD. DO NOT OUTPUT <think> TAGS or any other non-JSON text.
    7. Adhere strictly to the {TARGET_LANG} (Castilian Spanish) localization.
    8. Use context to maintain consistency in character names, tone, and references.
""").strip()

CHUTES_SINGLE_LINE_SYSTEM_PROMPT = dedent(f"""
    You are a precise and direct translator.
    Translate the given English text to {TARGET_LANG} (Castilian Spanish).
    Return ONLY the translated string, without any additional text, JSON formatting, or explanations.
    Preserve HTML-like tags (e.g., <i>) and bracketed text (e.g., [noise]).
""").strip()

# ─────────────────────────────────────────────────────────
#  ffmpeg helpers
# ─────────────────────────────────────────────────────────
def run_ffmpeg(cmd_list):
    proc = subprocess.run(cmd_list, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, check=False)
    if proc.returncode:
        raise RuntimeError(f"ffmpeg error ({proc.returncode}):\n{proc.stdout}")
    return proc.stdout

def extract_srt(video: Path, tmpdir: Path) -> Path | None:
    srt_out = tmpdir / "eng.srt"
    cmd = ["ffmpeg", "-v", FFMPEG_LOGLEVEL, "-y",
           "-i", str(video),
           "-map", "0:s:m:language:eng",
           "-c:s", "srt",
           str(srt_out)]
    try:
        run_ffmpeg(cmd)
    except RuntimeError as e:
        if "Subtitle stream not found" in str(e) or "does not contain any stream" in str(e):
            return None
        raise
    
    if not srt_out.exists() or srt_out.stat().st_size == 0:
        return None
    return srt_out

def embed_spanish_srt(video_path: Path, spanish_srt_path: Path):
    """Embeds Spanish SRT into video file alongside existing subtitles"""
    output_video = video_path.parent / f"{video_path.stem}.with_spanish{video_path.suffix}"
    
    print(f"  Embedding Spanish subtitles into {output_video.name}...")
    
    cmd = [
        "ffmpeg", "-v", FFMPEG_LOGLEVEL, "-y",
        "-i", str(video_path),
        "-i", str(spanish_srt_path),
        "-map", "0",  # Copy all streams from input video
        "-map", "1:0",  # Add the Spanish SRT as new stream
        "-c", "copy",  # Copy without re-encoding
        "-c:s:1", "srt",  # Set subtitle codec for new stream
        "-metadata:s:s:1", f"language={TARGET_LANG}",  # Set language metadata
        "-metadata:s:s:1", "title=Spanish",  # Set title metadata
        str(output_video)
    ]
    
    try:
        run_ffmpeg(cmd)
        print(f"  ✅ Video with Spanish subtitles saved as: {output_video.name}")
        return output_video
    except RuntimeError as e:
        print(f"  ❌ Failed to embed Spanish subtitles: {e}")
        raise

# ─────────────────────────────────────────────────────────
#  Translation Cache
# ─────────────────────────────────────────────────────────
_translation_cache: dict[str, str] = {}
_cache_hits = 0
_cache_misses = 0

def get_cache_stats() -> tuple[int, int]:
    """Returns (hits, misses) for cache statistics"""
    return _cache_hits, _cache_misses

def clear_cache():
    """Clear the translation cache"""
    global _translation_cache, _cache_hits, _cache_misses
    _translation_cache.clear()
    _cache_hits = 0
    _cache_misses = 0

# ─────────────────────────────────────────────────────────
#  HTTP Client Management
# ─────────────────────────────────────────────────────────
_http_client: httpx.AsyncClient | None = None

async def get_http_client() -> httpx.AsyncClient:
    """Get or create the shared HTTP client with connection pooling"""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(API_TIMEOUT_MAIN),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
            headers={"Accept-Encoding": "gzip, deflate"}
        )
    return _http_client

async def close_http_client():
    """Close the shared HTTP client"""
    global _http_client
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()
        _http_client = None

# ─────────────────────────────────────────────────────────
#  Token Bucket Rate Limiter
# ─────────────────────────────────────────────────────────
class TokenBucket:
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.last_refill = time.monotonic()
        self.lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> None:
        async with self.lock:
            now = time.monotonic()
            # Refill tokens based on elapsed time
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
            self.last_refill = now
            
            # If not enough tokens, wait
            if self.tokens < tokens:
                wait_time = (tokens - self.tokens) / self.refill_rate
                await asyncio.sleep(wait_time)
                self.tokens = 0
            else:
                self.tokens -= tokens

_rate_limiter = TokenBucket(capacity=MAX_RPM, refill_rate=MAX_RPM / 60.0)

async def throttle_async():
    """Advanced rate limiting using token bucket algorithm"""
    await _rate_limiter.acquire(1)

# ─────────────────────────────────────────────────────────
#  Name Detection and Preservation
# ─────────────────────────────────────────────────────────
import string

# Detective Conan character names only - ultra conservative
KNOWN_NAMES = {
    # Detective Conan main characters
    "Gin", "Vodka", "Vermouth", "Chianti", "Korn", "Bourbon", "Rye", 
    "Conan", "Shinichi", "Ran", "Kogoro", "Mouri", "Agasa", "Haibara", "Ai",
    "Ayumi", "Genta", "Mitsuhiko", "Sonoko", "Yusaku", "Yukiko", "Heiji", "Kazuha",
    "Jodie", "Akai", "Shuichi", "Camel", "James", "Black", "Takagi", "Sato",
    "Megure", "Shiratori", "Chiba", "Yokomizo", "Yamato", "Uehara", "Morofushi",
    "Kudo", "Edogawa", "Sera", "Okiya", "Subaru", "Masumi", "Eisuke", "Hondou",
    "Wataru", "Miwako", "Juzo", "Ninzaburo", "Kansuke", "Yui", "Hiromitsu",
    "Scotch", "Rikumichi", "Fusae", "Tomoaki", "Araide"
}

# No pattern detection - Detective Conan names only
NAME_PATTERNS = []

def detect_names_in_text(text: str) -> set[str]:
    """Detect Detective Conan character names only - ultra conservative"""
    detected_names = set()
    
    # Only exact matches against Detective Conan character names
    words = re.findall(r'\b\w+\b', text)
    for word in words:
        if word in KNOWN_NAMES:
            detected_names.add(word)
    
    return detected_names

def is_common_word(word: str) -> bool:
    """Check if a word is a common English word that shouldn't be treated as a name"""
    common_words = {
        # Articles, pronouns, basic words
        "The", "This", "That", "These", "Those", "A", "An", "And", "But", "Or", "So", "For", "If", "When", "Where",
        "What", "Who", "Why", "How", "Yes", "No", "OK", "Well", "Now", "Then", "Still", "Just", "Only", "Even",
        "Here", "There", "Please", "Thank", "Sorry", "Hello", "Goodbye", "Good", "Bad", "Right", "Wrong", "True", "False",
        "Big", "Small", "New", "Old", "First", "Last", "Next", "Come", "Go", "Back", "Away", "Up", "Down", "Out", "In",
        "Get", "Take", "Give", "Make", "Look", "See", "Know", "Think", "Want", "Need", "Have", "Has", "Had", "Will", "Would",
        
        # Time and sequence words
        "After", "Before", "During", "While", "Since", "Until", "Always", "Never", "Sometimes", "Often", "Usually",
        "Today", "Tomorrow", "Yesterday", "Tonight", "Morning", "Afternoon", "Evening", "Night", "Early", "Late",
        
        # Quantity and degree
        "All", "Some", "Many", "Few", "Much", "Little", "More", "Most", "Less", "Least", "Every", "Each", "Any", "None",
        "Something", "Nothing", "Everything", "Anything", "Someone", "Nobody", "Everyone", "Anyone", "Somewhere",
        "Nowhere", "Everywhere", "Anywhere", "Very", "Too", "Quite", "Really", "Actually", "Probably", "Maybe",
        "Definitely", "Certainly", "Possibly", "Perhaps", "Especially", "Particularly", "Exactly", "Almost", "Nearly",
        
        # Common sentence starters and conjunctions
        "Because", "Although", "However", "Therefore", "Besides", "Moreover", "Furthermore", "Nevertheless", "Meanwhile",
        "Otherwise", "Instead", "Rather", "Either", "Neither", "Whether", "Unless", "Though", "Whereas", "Whereas",
        
        # Direction and position
        "Above", "Below", "Over", "Under", "Between", "Among", "Through", "Across", "Around", "Behind", "Beside",
        "Inside", "Outside", "Within", "Without", "Against", "Toward", "Forward", "Backward", "Left", "Right",
        
        # Common adjectives and descriptors
        "Important", "Serious", "Strange", "Weird", "Crazy", "Amazing", "Incredible", "Wonderful", "Terrible", "Awful",
        "Great", "Excellent", "Perfect", "Fine", "Nice", "Beautiful", "Ugly", "Smart", "Stupid", "Easy", "Hard",
        "Fast", "Slow", "Hot", "Cold", "Warm", "Cool", "Dry", "Wet", "Clean", "Dirty", "Fresh", "Old", "Young",
        
        # Common verbs and actions
        "Being", "Doing", "Having", "Getting", "Going", "Coming", "Looking", "Seeing", "Hearing", "Feeling", "Thinking",
        "Knowing", "Understanding", "Believing", "Hoping", "Trying", "Working", "Playing", "Running", "Walking",
        "Talking", "Speaking", "Saying", "Telling", "Asking", "Answering", "Helping", "Watching", "Waiting", "Moving",
        
        # Emotions and states
        "Happy", "Sad", "Angry", "Scared", "Worried", "Excited", "Surprised", "Confused", "Tired", "Hungry", "Thirsty",
        "Sick", "Healthy", "Strong", "Weak", "Busy", "Free", "Ready", "Finished", "Done", "Started", "Stopped",
        
        # Common exclamations and responses  
        "Oh", "Ah", "Wow", "Hey", "Hi", "Bye", "Thanks", "Welcome", "Excuse", "Pardon", "Sure", "Course", "Alright",
        "Okay", "Fine", "Whatever", "Anyway", "Actually", "Basically", "Obviously", "Clearly", "Apparently",
        
        # Technical and common abbreviations (3+ letters to avoid catching real acronyms)
        "DVD", "USB", "GPS", "CPU", "RAM", "ROM", "LCD", "LED", "PDF", "ZIP", "EXE", "APP", "WEB", "NET", "COM",
        
        # Common short words that might be capitalized
        "IT", "US", "WE", "MY", "HIS", "HER", "OUR", "ITS", "HIM", "SHE", "HE", "I", "YOU", "THEY", "THEM",
        "AM", "IS", "ARE", "WAS", "WERE", "BE", "BEEN", "DO", "DID", "DOES", "CAN", "COULD", "WILL", "WOULD", "SHOULD",
        "MAY", "MIGHT", "MUST", "SHALL", "TO", "OF", "AT", "ON", "BY", "FROM", "WITH", "INTO", "ONTO", "UPON"
    }
    return word.lower() in {w.lower() for w in common_words}

def create_name_placeholders(text: str, names: set[str]) -> tuple[str, dict[str, str]]:
    """Replace names with placeholders and return mapping"""
    if not names:
        return text, {}
    
    placeholder_map = {}
    modified_text = text
    
    # Sort names by length (longest first) to avoid partial replacements
    sorted_names = sorted(names, key=len, reverse=True)
    
    for i, name in enumerate(sorted_names):
        placeholder = f"__NAME_{i}__"
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(name) + r'\b'
        modified_text = re.sub(pattern, placeholder, modified_text, flags=re.IGNORECASE)
        placeholder_map[placeholder] = name
    
    return modified_text, placeholder_map

def restore_names_from_placeholders(text: str, placeholder_map: dict[str, str]) -> str:
    """Restore original names from placeholders"""
    restored_text = text
    for placeholder, original_name in placeholder_map.items():
        restored_text = restored_text.replace(placeholder, original_name)
    return restored_text

def collect_names_from_batch(indexed_lines: list[tuple[int, str]]) -> set[str]:
    """Collect all names from a batch of subtitle lines"""
    all_names = set()
    for _, text in indexed_lines:
        names = detect_names_in_text(text)
        all_names.update(names)
    return all_names

# ─────────────────────────────────────────────────────────
#  Translation Logic (Chutes AI)
# ─────────────────────────────────────────────────────────
_eng_word_re = re.compile(r"[A-Za-z]{3}")
def looks_english(text: str) -> bool:
    return bool(_eng_word_re.search(text)) and not re.search(r"[áéíóúñÁÉÍÓÚÑ¿¡]", text)

async def translate_batch_with_context_async(indexed_lines: list[tuple[int, str]], context_lines: list[str], max_retries=3) -> list[tuple[int, str]]:
    """Translate batch with context awareness for better dialogue continuity"""
    global _cache_hits, _cache_misses
    
    if not indexed_lines:
        return []

    # Check cache for each line first
    cached_results = []
    uncached_lines = []
    
    for idx, text in indexed_lines:
        text_key = text.strip().lower()
        if text_key in _translation_cache:
            cached_results.append((idx, _translation_cache[text_key]))
            _cache_hits += 1
        else:
            uncached_lines.append((idx, text))
            _cache_misses += 1
    
    # If all lines were cached, return immediately
    if not uncached_lines:
        return cached_results
    
    # Process uncached lines with context
    idx_list = [i for i, _ in uncached_lines]
    en_lines = [text for _, text in uncached_lines]
    
    # Detect and preserve names
    all_names = collect_names_from_batch(uncached_lines)
    if context_lines:
        # Also collect names from context
        for ctx_text in context_lines:
            ctx_names = detect_names_in_text(ctx_text)
            all_names.update(ctx_names)
    
    # Replace names with placeholders for translation
    placeholder_maps = []
    protected_en_lines = []
    protected_context_lines = []
    
    for line in en_lines:
        protected_line, placeholder_map = create_name_placeholders(line, all_names)
        protected_en_lines.append(protected_line)
        placeholder_maps.append(placeholder_map)
    
    if context_lines:
        for ctx_line in context_lines:
            protected_ctx_line, _ = create_name_placeholders(ctx_line, all_names)
            protected_context_lines.append(protected_ctx_line)
    
    # Build context-aware prompt with protected names
    en_lines_json_array_string = json.dumps(protected_en_lines, ensure_ascii=False)
    
    if protected_context_lines:
        context_json_string = json.dumps(protected_context_lines, ensure_ascii=False)
        user_prompt_content = f"CONTEXT: {context_json_string}\nTRANSLATE: {en_lines_json_array_string}"
    else:
        user_prompt_content = f"TRANSLATE: {en_lines_json_array_string}"

    headers = {
        "Authorization": f"Bearer {CHUTES_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": CHUTES_MODEL_ID,
        "messages": [
            {"role": "system", "content": CHUTES_BATCH_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt_content}
        ],
        "stream": False, 
        "max_tokens": MAX_OUTPUT_TOKENS_BATCH, 
        "temperature": TRANSLATION_TEMPERATURE
    }

    def parse_json_array_from_string(txt: str, expected_len: int) -> list[str] | None:
        try:
            think_start_tag = "<think>"
            think_end_tag = "</think>"
            if think_start_tag in txt:
                last_end_tag_idx = txt.rfind(think_end_tag)
                if last_end_tag_idx != -1:
                    txt = txt[last_end_tag_idx + len(think_end_tag):].strip()

            if txt.startswith("```json"):
                txt = txt.strip()[7:-3].strip()
            elif txt.startswith("```"):
                txt = txt.strip()[3:-3].strip()
            
            first_bracket = txt.find('[')
            last_bracket = txt.rfind(']')
            if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
                json_candidate = txt[first_bracket : last_bracket+1]
                out = json.loads(json_candidate)
                if isinstance(out, list) and all(isinstance(item, str) for item in out) and len(out) == expected_len:
                    return out
            else: 
                out = json.loads(txt)
                if isinstance(out, list) and all(isinstance(item, str) for item in out) and len(out) == expected_len:
                    return out
        except json.JSONDecodeError:
            pass
        return None

    for attempt in range(max_retries):
        try:
            await throttle_async()
            client = await get_http_client()
            response = await client.post(CHUTES_API_ENDPOINT, headers=headers, json=payload)
            response.raise_for_status() 
            
            response_data = response.json()
            if not response_data.get("choices") or not response_data["choices"][0].get("message"):
                raise ValueError(f"Unexpected Chutes API response structure: {str(response_data)[:300]}")
            
            assistant_response_content = response_data["choices"][0]["message"].get("content", "")
            
            spa_lines = parse_json_array_from_string(assistant_response_content, len(en_lines))

            if spa_lines:
                for k, original_line_text in enumerate(en_lines):
                    translated_text = spa_lines[k]
                    if looks_english(translated_text) or \
                       len(translated_text) > (len(original_line_text) * 3 + 30) or \
                       (len(original_line_text) > 5 and len(translated_text) < (len(original_line_text) * 0.3)):
                        
                        await throttle_async()
                        single_retry_payload = {
                            "model": CHUTES_MODEL_ID,
                            "messages": [
                                {"role": "system", "content": CHUTES_SINGLE_LINE_SYSTEM_PROMPT},
                                {"role": "user", "content": original_line_text}
                            ],
                            "stream": False,
                            "max_tokens": MAX_OUTPUT_TOKENS_SINGLE,
                            "temperature": 0.0 
                        }
                        try:
                            retry_client = await get_http_client()
                            single_retry_http_response = await retry_client.post(CHUTES_API_ENDPOINT, headers=headers, json=single_retry_payload)
                            single_retry_http_response.raise_for_status()
                            
                            single_retry_data = single_retry_http_response.json()
                            if not single_retry_data.get("choices") or not single_retry_data["choices"][0].get("message"):
                                print(f"Warning: Unexpected structure in single line retry response for '{original_line_text}'. Skipping retry.")
                                continue
                            cleaned_retry_text = single_retry_data["choices"][0]["message"].get("content", "").strip()
                            
                            if cleaned_retry_text or not original_line_text.strip(): 
                                spa_lines[k] = cleaned_retry_text
                        except Exception as e_single:
                            print(f"Warning: Single line retry via Chutes failed for '{original_line_text}': {type(e_single).__name__}. Keeping batch attempt.")
                
                # Restore names from placeholders in all translated lines
                for k in range(len(spa_lines)):
                    if k < len(placeholder_maps):
                        spa_lines[k] = restore_names_from_placeholders(spa_lines[k], placeholder_maps[k])
                
                # Cache the successful translations (with original text as key)
                for k, original_line_text in enumerate(en_lines):
                    text_key = original_line_text.strip().lower()
                    _translation_cache[text_key] = spa_lines[k]
                
                # Combine cached and newly translated results
                new_results = list(zip(idx_list, spa_lines))
                all_results = cached_results + new_results
                all_results.sort(key=lambda x: x[0])  # Sort by original index
                return all_results

            raise ValueError(f"Non-JSON array or malformed/unparseable response from Chutes: {assistant_response_content[:300]}...")

        except httpx.HTTPStatusError as e:
            error_text = e.response.text[:200] if hasattr(e.response, 'text') else "No response body"
            print(f"Warning: Chutes API HTTP Error attempt {attempt + 1}/{max_retries}. Status: {e.response.status_code}. Response: {error_text}")
            if e.response.status_code == 429: 
                print("Rate limit hit. Waiting longer...")
                await asyncio.sleep(min(60, 15 + (3 ** attempt) + random.random())) 
            elif e.response.status_code >= 500: 
                if attempt < max_retries - 1: await asyncio.sleep((2 ** attempt) + random.random())
            else: 
                 if attempt < max_retries - 1: await asyncio.sleep((2 ** attempt) + random.random())
                 else: break 
        except (httpx.RequestError, httpx.TimeoutException, ValueError) as e: 
            print(f"Warning: API call/parsing attempt {attempt + 1}/{max_retries} failed. Error: {type(e).__name__} - {str(e)[:200]}")
            if attempt < max_retries - 1:
                await asyncio.sleep((2 ** attempt) + random.random())
        
    if len(uncached_lines) > 1 and attempt == max_retries -1 : 
        print(f"Splitting batch of {len(uncached_lines)} lines and retrying after exhausting direct retries...")
        mid = len(uncached_lines) // 2
        res1_task = translate_batch_with_context_async(uncached_lines[:mid], context_lines, max_retries=max_retries)
        res2_task = translate_batch_with_context_async(uncached_lines[mid:], context_lines, max_retries=max_retries)
        res1, res2 = await asyncio.gather(res1_task, res2_task)
        all_results = cached_results + res1 + res2
        all_results.sort(key=lambda x: x[0])
        return all_results
    elif uncached_lines: 
        original_text = uncached_lines[0][1]
        print(f"Error: Chutes API failed to translate line/batch after all retries and splits: {original_text!r}")
        failed_results = [(idx_list[i], f"[UNTRANSLATED] {en_lines[i]}") for i in range(len(idx_list))]
        # Still combine with cached results
        all_results = cached_results + failed_results
        all_results.sort(key=lambda x: x[0])
        return all_results
    
    print(f"Error: Exhausted all options for batch starting with: {(en_lines[0] if en_lines else 'empty batch')!r}")
    failed_results = [(idx, f"[FAILED_TRANSLATION] {text}") for idx, text in uncached_lines]
    all_results = cached_results + failed_results
    all_results.sort(key=lambda x: x[0])
    return all_results

async def translate_batch_async(indexed_lines: list[tuple[int, str]], max_retries=3) -> list[tuple[int, str]]:
    global _cache_hits, _cache_misses
    
    if not indexed_lines:
        return []

    # Check cache for each line first
    cached_results = []
    uncached_lines = []
    
    for idx, text in indexed_lines:
        text_key = text.strip().lower()
        if text_key in _translation_cache:
            cached_results.append((idx, _translation_cache[text_key]))
            _cache_hits += 1
        else:
            uncached_lines.append((idx, text))
            _cache_misses += 1
    
    # If all lines were cached, return immediately
    if not uncached_lines:
        return cached_results
    
    # Process uncached lines
    idx_list = [i for i, _ in uncached_lines]
    en_lines = [text for _, text in uncached_lines]
    
    en_lines_json_array_string = json.dumps(en_lines, ensure_ascii=False)
    user_prompt_content = f"Translate the English subtitle lines in the following JSON array to {TARGET_LANG}:\n{en_lines_json_array_string}"

    headers = {
        "Authorization": f"Bearer {CHUTES_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": CHUTES_MODEL_ID,
        "messages": [
            {"role": "system", "content": CHUTES_BATCH_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt_content}
        ],
        "stream": False, 
        "max_tokens": MAX_OUTPUT_TOKENS_BATCH, 
        "temperature": TRANSLATION_TEMPERATURE
    }

    def parse_json_array_from_string(txt: str, expected_len: int) -> list[str] | None:
        try:
            think_start_tag = "<think>"
            think_end_tag = "</think>"
            if think_start_tag in txt:
                last_end_tag_idx = txt.rfind(think_end_tag)
                if last_end_tag_idx != -1:
                    txt = txt[last_end_tag_idx + len(think_end_tag):].strip()

            if txt.startswith("```json"):
                txt = txt.strip()[7:-3].strip()
            elif txt.startswith("```"):
                txt = txt.strip()[3:-3].strip()
            
            first_bracket = txt.find('[')
            last_bracket = txt.rfind(']')
            if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
                json_candidate = txt[first_bracket : last_bracket+1]
                out = json.loads(json_candidate)
                if isinstance(out, list) and all(isinstance(item, str) for item in out) and len(out) == expected_len:
                    return out
            else: 
                out = json.loads(txt)
                if isinstance(out, list) and all(isinstance(item, str) for item in out) and len(out) == expected_len:
                    return out
        except json.JSONDecodeError:
            pass
        return None

    for attempt in range(max_retries):
        try:
            await throttle_async()
            client = await get_http_client()
            response = await client.post(CHUTES_API_ENDPOINT, headers=headers, json=payload)
            response.raise_for_status() 
            
            response_data = response.json()
            if not response_data.get("choices") or not response_data["choices"][0].get("message"):
                raise ValueError(f"Unexpected Chutes API response structure: {str(response_data)[:300]}")
            
            assistant_response_content = response_data["choices"][0]["message"].get("content", "")
            
            spa_lines = parse_json_array_from_string(assistant_response_content, len(en_lines))

            if spa_lines:
                for k, original_line_text in enumerate(en_lines):
                    translated_text = spa_lines[k]
                    if looks_english(translated_text) or \
                       len(translated_text) > (len(original_line_text) * 3 + 30) or \
                       (len(original_line_text) > 5 and len(translated_text) < (len(original_line_text) * 0.3)):
                        
                        await throttle_async()
                        single_retry_payload = {
                            "model": CHUTES_MODEL_ID,
                            "messages": [
                                {"role": "system", "content": CHUTES_SINGLE_LINE_SYSTEM_PROMPT},
                                {"role": "user", "content": original_line_text}
                            ],
                            "stream": False,
                            "max_tokens": MAX_OUTPUT_TOKENS_SINGLE,
                            "temperature": 0.0 
                        }
                        try:
                            retry_client = await get_http_client()
                            single_retry_http_response = await retry_client.post(CHUTES_API_ENDPOINT, headers=headers, json=single_retry_payload)
                            single_retry_http_response.raise_for_status()
                            
                            single_retry_data = single_retry_http_response.json()
                            if not single_retry_data.get("choices") or not single_retry_data["choices"][0].get("message"):
                                print(f"Warning: Unexpected structure in single line retry response for '{original_line_text}'. Skipping retry.")
                                continue
                            cleaned_retry_text = single_retry_data["choices"][0]["message"].get("content", "").strip()
                            
                            if cleaned_retry_text or not original_line_text.strip(): 
                                spa_lines[k] = cleaned_retry_text
                        except Exception as e_single:
                            print(f"Warning: Single line retry via Chutes failed for '{original_line_text}': {type(e_single).__name__}. Keeping batch attempt.")
                
                # Cache the successful translations
                for k, original_line_text in enumerate(en_lines):
                    text_key = original_line_text.strip().lower()
                    _translation_cache[text_key] = spa_lines[k]
                
                # Combine cached and newly translated results
                new_results = list(zip(idx_list, spa_lines))
                all_results = cached_results + new_results
                all_results.sort(key=lambda x: x[0])  # Sort by original index
                return all_results

            raise ValueError(f"Non-JSON array or malformed/unparseable response from Chutes: {assistant_response_content[:300]}...")

        except httpx.HTTPStatusError as e:
            error_text = e.response.text[:200] if hasattr(e.response, 'text') else "No response body"
            print(f"Warning: Chutes API HTTP Error attempt {attempt + 1}/{max_retries}. Status: {e.response.status_code}. Response: {error_text}")
            if e.response.status_code == 429: 
                print("Rate limit hit. Waiting longer...")
                await asyncio.sleep(min(60, 15 + (3 ** attempt) + random.random())) 
            elif e.response.status_code >= 500: 
                if attempt < max_retries - 1: await asyncio.sleep((2 ** attempt) + random.random())
            else: 
                 if attempt < max_retries - 1: await asyncio.sleep((2 ** attempt) + random.random())
                 else: break 
        except (httpx.RequestError, httpx.TimeoutException, ValueError) as e: 
            print(f"Warning: API call/parsing attempt {attempt + 1}/{max_retries} failed. Error: {type(e).__name__} - {str(e)[:200]}")
            if attempt < max_retries - 1:
                await asyncio.sleep((2 ** attempt) + random.random())
        
    if len(indexed_lines) > 1 and attempt == max_retries -1 : 
        print(f"Splitting batch of {len(indexed_lines)} lines and retrying after exhausting direct retries...")
        mid = len(indexed_lines) // 2
        res1_task = translate_batch_async(indexed_lines[:mid], max_retries=max_retries)
        res2_task = translate_batch_async(indexed_lines[mid:], max_retries=max_retries)
        res1, res2 = await asyncio.gather(res1_task, res2_task)
        return res1 + res2
    elif uncached_lines: 
        original_text = uncached_lines[0][1]
        print(f"Error: Chutes API failed to translate line/batch after all retries and splits: {original_text!r}")
        failed_results = [(idx_list[i], f"[UNTRANSLATED] {en_lines[i]}") for i in range(len(idx_list))]
        # Still combine with cached results
        all_results = cached_results + failed_results
        all_results.sort(key=lambda x: x[0])
        return all_results
    
    print(f"Error: Exhausted all options for batch starting with: {(en_lines[0] if en_lines else 'empty batch')!r}")
    failed_results = [(idx, f"[FAILED_TRANSLATION] {text}") for idx, text in uncached_lines]
    all_results = cached_results + failed_results
    all_results.sort(key=lambda x: x[0])
    return all_results


def estimate_tokens_more_accurately(text: str) -> int:
    """More accurate token estimation considering punctuation, special chars, and text complexity"""
    if not text:
        return 5
    
    # Base word count
    words = len(text.split())
    
    # Add tokens for punctuation and special characters
    punct_tokens = len([c for c in text if c in ".,!?;:()[]{}\"'"])
    
    # Add tokens for HTML tags and bracketed content
    html_tags = len(re.findall(r'<[^>]+>', text))
    bracketed = len(re.findall(r'\[[^\]]+\]', text))
    
    # Longer words typically use more tokens
    long_word_bonus = sum(1 for word in text.split() if len(word) > 8)
    
    # Non-ASCII characters may use more tokens
    non_ascii_chars = sum(1 for c in text if ord(c) > 127)
    
    total_tokens = (words * 1.3) + (punct_tokens * 0.3) + (html_tags * 2) + (bracketed * 1.5) + long_word_bonus + (non_ascii_chars * 0.2) + 5
    
    return int(total_tokens)

def build_batches_with_context(subs: pysrt.SubRipFile, batch_lines_config: int):
    """Build batches with context-aware sliding window for better translation continuity"""
    if not subs:
        return

    batch: list[tuple[int, str]] = []
    all_lines = [(i, s_item.text) for i, s_item in enumerate(subs)]
    
    def get_context_lines(current_batch_start_idx: int) -> list[str]:
        """Get context lines from previous batches"""
        context_start = max(0, current_batch_start_idx - CONTEXT_WINDOW_SIZE)
        context_lines = []
        
        for idx in range(context_start, current_batch_start_idx):
            if idx < len(all_lines):
                context_lines.append(all_lines[idx][1])
        
        return context_lines
    
    if batch_lines_config == 0:  # Token-based batching
        current_tokens_for_subs = 0
        batch_start_idx = 0
        
        for i, s_item in enumerate(subs):
            line_tokens_estimate = estimate_tokens_more_accurately(s_item.text)
            
            # Add token cost for context (roughly estimated)
            context_lines = get_context_lines(batch_start_idx)
            context_tokens = sum(estimate_tokens_more_accurately(ctx) for ctx in context_lines) // 2  # Context tokens count less
            
            total_tokens = PROMPT_OVERHEAD_TOKENS + current_tokens_for_subs + line_tokens_estimate + context_tokens
            
            if total_tokens > IDEAL_TOKENS_PER_BATCH and batch:
                yield (batch, get_context_lines(batch_start_idx))
                batch = []
                current_tokens_for_subs = 0
                batch_start_idx = i
            
            batch.append((i, s_item.text)) 
            current_tokens_for_subs += line_tokens_estimate
            
        if batch:
            yield (batch, get_context_lines(batch_start_idx))
            
    else:  # Line-based batching
        batch_start_idx = 0
        
        for i, s_item in enumerate(subs):
            batch.append((i, s_item.text))
            
            if len(batch) >= batch_lines_config:
                yield (batch, get_context_lines(batch_start_idx))
                batch = []
                batch_start_idx = i + 1
                
        if batch:
            yield (batch, get_context_lines(batch_start_idx))

def build_batches(subs: pysrt.SubRipFile, batch_lines_config: int):
    """Legacy wrapper for backwards compatibility"""
    for batch_data, context_lines in build_batches_with_context(subs, batch_lines_config):
        yield batch_data

async def _translate_and_save_srt_async_job(subs: pysrt.SubRipFile, video_path: Path, batch_lines_config: int, show_review: bool = True): # RENAMED
    # Store original subtitles for review dialog
    original_subs = pysrt.SubRipFile()
    for sub in subs:
        original_subs.append(pysrt.SubRipItem(sub.index, sub.start, sub.end, sub.text))
    
    batches_with_context = list(build_batches_with_context(subs, batch_lines_config))
    if not batches_with_context:
        print("  No batches to translate.")
        return

    print(f"  📝 Processing {len(batches_with_context)} batches with context awareness...")
    
    while True:  # Loop to handle "Try Again" functionality
        # Reset translations for retry
        translated_subs = pysrt.SubRipFile()
        for sub in original_subs:
            translated_subs.append(pysrt.SubRipItem(sub.index, sub.start, sub.end, sub.text))
        
        tasks = [translate_batch_with_context_async(batch_data, context_lines) 
                 for batch_data, context_lines in batches_with_context]
        
        all_translated_indexed_lines_nested = await tqdm.gather(
            *tasks, desc=f"Translating {video_path.name}", unit="batch"
        )

        found_translations = False
        for translated_batch in all_translated_indexed_lines_nested:
            for original_idx, new_text in translated_batch:
                if original_idx < len(translated_subs): 
                    translated_subs[original_idx].text = new_text
                    found_translations = True
                else:
                    print(f"Warning: Invalid subtitle index {original_idx} returned from translation.")

        if not found_translations and translated_subs: 
            print("  Warning: No translations were applied.")
        
        # Show review dialog if requested (GUI mode)
        if show_review:
            print("  📋 Opening review dialog for user approval...")
            try:
                # This will be called from a thread, so we need to schedule UI work on main thread
                return await _handle_translation_review(original_subs, translated_subs, video_path)
            except Exception as e:
                print(f"  Error showing review dialog: {e}")
                # Fall back to auto-accept if review fails
                break
        else:
            # CLI mode - auto accept
            break
    
    # If we get here, proceed with embedding (either auto-accepted or no review)
    return await _finalize_translation(translated_subs, video_path)

async def _handle_translation_review(original_subs: pysrt.SubRipFile, translated_subs: pysrt.SubRipFile, video_path: Path):
    """Handle the translation review process"""
    # This function will be implemented to work with the UI thread
    # For now, return the translated subs directly
    return await _finalize_translation(translated_subs, video_path)

async def _finalize_translation(subs: pysrt.SubRipFile, video_path: Path):
    """Finalize the translation by embedding into video"""
    # --- Save Spanish SRT and embed into video ---
    video_dir = video_path.parent
    video_stem = video_path.stem

    # Save Spanish SRT temporarily
    temp_spanish_srt = video_dir / f"{video_stem}.temp.{TARGET_LANG}.srt"
    
    try:
        subs.save(str(temp_spanish_srt), encoding="utf-8")
        print(f"  ✅ Spanish SRT created: {temp_spanish_srt.name}")
        
        sample_text = "-- No text found --"
        for s_item in subs:
            if s_item.text and s_item.text.strip() and \
               not s_item.text.startswith("[UNTRANSLATED]") and \
               not s_item.text.startswith("[FAILED_TRANSLATION]"):
                sample_text = s_item.text
                break
        print(f"  ↳ Sample translation: {sample_text}")

        # Embed Spanish SRT into video alongside existing subtitles
        embed_spanish_srt(video_path, temp_spanish_srt)
        
        # Clean up temporary SRT file
        temp_spanish_srt.unlink()
        
        # Report cache statistics
        hits, misses = get_cache_stats()
        total_requests = hits + misses
        if total_requests > 0:
            cache_rate = (hits / total_requests) * 100
            print(f"  📊 Cache stats: {hits}/{total_requests} hits ({cache_rate:.1f}% hit rate)")

        return True  # Success

    except Exception as e:
        print(f"  Error saving/embedding Spanish SRT: {e}")
        if temp_spanish_srt.exists():
            temp_spanish_srt.unlink()
        return False  # Failure




# ─────────────────────────────────────────────────────────
#  Per-file pipeline
# ─────────────────────────────────────────────────────────
def process_video_file(video_file: Path, batch_size: int, preview_mode: bool = False):
    with tempfile.TemporaryDirectory(prefix="subtrans_") as td_name:
        tmpdir_path = Path(td_name)
        print(f"\n▶ Processing: {video_file.name}")

        eng_srt_path = extract_srt(video_file, tmpdir_path)
        if not eng_srt_path:
            print(f"  …No English subtitle track found or extracted for {video_file.name}, skipped.")
            return

        try:
            subs = pysrt.open(str(eng_srt_path), encoding="utf-8")
        except Exception as e:
            print(f"  Error opening SRT file {eng_srt_path.name}: {e}. Skipping.")
            return
            
        if not subs:
            print(f"  Extracted SRT {eng_srt_path.name} is empty or unreadable. Possibly a bitmap (PGS/VobSub) subtitle. Skipped.")
            return

        if preview_mode:
            print("  Previewing first ~5 lines (or up to 60 lines if less)...")
            preview_subs_count = min(len(subs), 60) 
            preview_indexed_lines = [(i, subs[i].text) for i in range(min(len(subs), preview_subs_count))]
            
            if not preview_indexed_lines:
                print("  No lines to preview.")
            else:
                async def run_preview():
                    translated_preview = await translate_batch_async(preview_indexed_lines)
                    print("  SAMPLE TRANSLATION OUTPUT:")
                    for _, line in translated_preview[:min(len(translated_preview), 5)]: 
                        print(f"    {line}")
                
                asyncio.run(run_preview())

            ans = input("\nLooks good? Continue with full translation? (y/N): ").strip().lower()
            if ans != "y":
                print(f"  Skipped full translation for {video_file.name} by user.")
                return
        
        try:
            # Call the renamed function (CLI mode - no review dialog)
            asyncio.run(_translate_and_save_srt_async_job(subs, video_file, batch_size, show_review=False))
        except Exception as e:
            print(f"  Error during translation/saving for {video_file.name}: {type(e).__name__} - {e}")
            import traceback
            traceback.print_exc()


# ─────────────────────────────────────────────────────────
#  Command-line interface
# ─────────────────────────────────────────────────────────
async def process_multiple_videos_async(video_paths: list[Path], batch_size: int, preview_mode: bool = False, max_concurrent: int = 3):
    """Process multiple videos concurrently with limited concurrency"""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_single_video(video_path: Path):
        async with semaphore:
            return await asyncio.to_thread(process_video_file, video_path, batch_size, preview_mode)
    
    tasks = [process_single_video(v_path) for v_path in video_paths if v_path.is_file()]
    invalid_files = [v_path for v_path in video_paths if not v_path.is_file()]
    
    for invalid_file in invalid_files:
        print(f"Warning: Video file not found: {invalid_file}. Skipping.")
    
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    
    # Final cleanup
    await close_http_client()

def cli_main():
    global MAX_RPM, _rate_limiter

    parser = argparse.ArgumentParser(description="Subtitle translator using Chutes AI, embeds Spanish SRT into video.")
    parser.add_argument("videos", nargs="+", help="Paths to video files (MKV, MP4, etc.)")
    parser.add_argument("--batch", type=int, default=60,
                        help="Lines per API request. Use 0 for auto-batching by token budget. Default: 60 lines.")
    parser.add_argument("--preview", action="store_true",
                        help="Translate only the first few lines and ask for confirmation.")
    parser.add_argument("--rpm", type=int, default=MAX_RPM, 
                        help=f"Max requests per minute to the API. Default: {MAX_RPM}.")
    parser.add_argument("--concurrent", type=int, default=3,
                        help="Max number of videos to process concurrently. Default: 3.")

    args = parser.parse_args()
    
    MAX_RPM = args.rpm
    # Update rate limiter with new RPM
    _rate_limiter = TokenBucket(capacity=MAX_RPM, refill_rate=MAX_RPM / 60.0) 

    video_paths = [Path(v) for v in args.videos]
    asyncio.run(process_multiple_videos_async(video_paths, args.batch, args.preview, args.concurrent))
    
    print("\nAll tasks completed.")

# ─────────────────────────────────────────────────────────
#  Modern UI Application
# ─────────────────────────────────────────────────────────
class ModernSubtitleTranslatorApp:
    def __init__(self):
        # Initialize the modern themed window
        if MODERN_UI_AVAILABLE:
            self.root = ttk.Window(themename="superhero")  # Dark modern theme
            self.style = self.root.style
        else:
            self.root = tk.Tk()
            self.style = ttk.Style()
        
        self.setup_window()
        self.files_to_process = []
        self.is_running = False
        self.current_file_count = 0
        self.total_files = 0
        
        self.build_modern_ui()
        self.setup_drag_drop()

    def setup_window(self):
        """Configure the main window"""
        self.root.title("AISRT - AI Subtitle Translator")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        
        # Center window on screen
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")
        
        # Set window icon (if available)
        try:
            self.root.iconbitmap('icon.ico')
        except:
            pass

    def build_modern_ui(self):
        """Build the modern, beautiful UI"""
        # Main container with padding
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=BOTH, expand=True, padx=20, pady=20)
        
        # Header Section
        self.create_header(main_container)
        
        # File Management Section  
        self.create_file_section(main_container)
        
        # Settings Section
        self.create_settings_section(main_container)
        
        # Action Buttons Section
        self.create_action_section(main_container)
        
        # Progress and Status Section
        self.create_progress_section(main_container)

    def create_header(self, parent):
        """Create the header with title and description"""
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill=X, pady=(0, 30))
        
        # Main title
        title_label = ttk.Label(
            header_frame, 
            text="🎬 AI Subtitle Translator", 
            font=("Segoe UI", 24, "bold")
        )
        title_label.pack(anchor=W)
        
        # Subtitle description
        desc_text = "Transform your videos with AI-powered Spanish subtitle translation"
        desc_label = ttk.Label(
            header_frame, 
            text=desc_text,
            font=("Segoe UI", 11),
            foreground="gray"
        )
        desc_label.pack(anchor=W, pady=(5, 0))
        
        # Separator
        ttk.Separator(header_frame, orient=HORIZONTAL).pack(fill=X, pady=(15, 0))

    def create_file_section(self, parent):
        """Create the file management section"""
        file_frame = ttk.LabelFrame(parent, text="📁 Video Files", padding=20)
        file_frame.pack(fill=BOTH, expand=True, pady=(0, 20))
        
        # File list with modern styling
        list_container = ttk.Frame(file_frame)
        list_container.pack(fill=BOTH, expand=True)
        
        # Treeview for better file display
        columns = ("name", "size", "status")
        self.file_tree = ttk.Treeview(
            list_container, 
            columns=columns, 
            show="tree headings",
            height=12
        )
        
        # Configure columns
        self.file_tree.heading("#0", text="📄", anchor=W)
        self.file_tree.heading("name", text="File Name", anchor=W)
        self.file_tree.heading("size", text="Size", anchor=CENTER)
        self.file_tree.heading("status", text="Status", anchor=CENTER)
        
        self.file_tree.column("#0", width=40, minwidth=40)
        self.file_tree.column("name", width=400, minwidth=200)
        self.file_tree.column("size", width=100, minwidth=80)
        self.file_tree.column("status", width=120, minwidth=100)
        
        # Scrollbar for file list
        file_scrollbar = ttk.Scrollbar(list_container, orient=VERTICAL, command=self.file_tree.yview)
        self.file_tree.configure(yscrollcommand=file_scrollbar.set)
        
        # Pack treeview and scrollbar
        self.file_tree.pack(side=LEFT, fill=BOTH, expand=True)
        file_scrollbar.pack(side=RIGHT, fill=Y)
        
        # File action buttons
        file_buttons_frame = ttk.Frame(file_frame)
        file_buttons_frame.pack(fill=X, pady=(15, 0))
        
        self.add_button = ttk.Button(
            file_buttons_frame, 
            text="➕ Add Videos",
            command=self.add_files,
            style="Accent.TButton",
            width=15
        )
        self.add_button.pack(side=LEFT, padx=(0, 10))
        
        self.remove_button = ttk.Button(
            file_buttons_frame,
            text="🗑️ Remove Selected", 
            command=self.remove_files,
            width=18
        )
        self.remove_button.pack(side=LEFT, padx=(0, 10))
        
        self.clear_button = ttk.Button(
            file_buttons_frame,
            text="🧹 Clear All",
            command=self.clear_all_files,
            width=12
        )
        self.clear_button.pack(side=LEFT)
        
        # Drop zone indicator
        drop_label = ttk.Label(
            file_frame, 
            text="💡 Tip: Drag and drop video files here!",
            font=("Segoe UI", 9),
            foreground="gray"
        )
        drop_label.pack(pady=(10, 0))

    def create_settings_section(self, parent):
        """Create the settings configuration section"""
        settings_frame = ttk.LabelFrame(parent, text="⚙️ Translation Settings", padding=20)
        settings_frame.pack(fill=X, pady=(0, 20))
        
        # Create two columns for settings
        left_settings = ttk.Frame(settings_frame)
        left_settings.pack(side=LEFT, fill=X, expand=True)
        
        right_settings = ttk.Frame(settings_frame)
        right_settings.pack(side=RIGHT, fill=X, expand=True, padx=(20, 0))
        
        # Batch Size Setting
        batch_frame = ttk.Frame(left_settings)
        batch_frame.pack(fill=X, pady=(0, 15))
        
        ttk.Label(batch_frame, text="Batch Size:", font=("Segoe UI", 10, "bold")).pack(anchor=W)
        
        batch_input_frame = ttk.Frame(batch_frame)
        batch_input_frame.pack(fill=X, pady=(5, 0))
        
        self.batch_var = tk.IntVar(value=60)
        batch_entry = ttk.Entry(batch_input_frame, textvariable=self.batch_var, width=8)
        batch_entry.pack(side=LEFT)
        
        batch_info = ttk.Label(
            batch_input_frame, 
            text="lines per request (0 = auto)",
            font=("Segoe UI", 9),
            foreground="gray"
        )
        batch_info.pack(side=LEFT, padx=(10, 0))
        
        if MODERN_UI_AVAILABLE:
            ToolTip(batch_entry, text="Number of subtitle lines to translate in each API request. Set to 0 for automatic token-based batching.")
        
        # Rate Limit Setting  
        rpm_frame = ttk.Frame(left_settings)
        rpm_frame.pack(fill=X)
        
        ttk.Label(rpm_frame, text="Rate Limit:", font=("Segoe UI", 10, "bold")).pack(anchor=W)
        
        rpm_input_frame = ttk.Frame(rpm_frame)
        rpm_input_frame.pack(fill=X, pady=(5, 0))
        
        self.rpm_var = tk.IntVar(value=MAX_RPM)
        rpm_entry = ttk.Entry(rpm_input_frame, textvariable=self.rpm_var, width=8)
        rpm_entry.pack(side=LEFT)
        
        rpm_info = ttk.Label(
            rpm_input_frame,
            text="requests per minute",
            font=("Segoe UI", 9), 
            foreground="gray"
        )
        rpm_info.pack(side=LEFT, padx=(10, 0))
        
        if MODERN_UI_AVAILABLE:
            ToolTip(rpm_entry, text="Maximum API requests per minute to avoid rate limiting.")
        
        # Context Window Setting
        context_frame = ttk.Frame(right_settings)
        context_frame.pack(fill=X, pady=(0, 15))
        
        ttk.Label(context_frame, text="Context Window:", font=("Segoe UI", 10, "bold")).pack(anchor=W)
        
        context_input_frame = ttk.Frame(context_frame)
        context_input_frame.pack(fill=X, pady=(5, 0))
        
        self.context_var = tk.IntVar(value=CONTEXT_WINDOW_SIZE)
        context_entry = ttk.Entry(context_input_frame, textvariable=self.context_var, width=8)
        context_entry.pack(side=LEFT)
        
        context_info = ttk.Label(
            context_input_frame,
            text="previous lines for context",
            font=("Segoe UI", 9),
            foreground="gray"
        )
        context_info.pack(side=LEFT, padx=(10, 0))
        
        if MODERN_UI_AVAILABLE:
            ToolTip(context_entry, text="Number of previous subtitle lines to include as context for better translation continuity.")
        
        # Advanced Options
        advanced_frame = ttk.Frame(right_settings)
        advanced_frame.pack(fill=X)
        
        ttk.Label(advanced_frame, text="Options:", font=("Segoe UI", 10, "bold")).pack(anchor=W)
        
        self.preview_var = tk.BooleanVar(value=False)
        preview_check = ttk.Checkbutton(
            advanced_frame,
            text="Preview mode (translate first few lines only)",
            variable=self.preview_var
        )
        preview_check.pack(anchor=W, pady=(5, 0))

    def create_action_section(self, parent):
        """Create the main action buttons"""
        action_frame = ttk.Frame(parent)
        action_frame.pack(fill=X, pady=(0, 20))
        
        # Status info on the left
        status_info_frame = ttk.Frame(action_frame)
        status_info_frame.pack(side=LEFT, fill=X, expand=True)
        
        self.file_count_label = ttk.Label(
            status_info_frame,
            text="No files selected",
            font=("Segoe UI", 10)
        )
        self.file_count_label.pack(anchor=W)
        
        # Main action button on the right
        self.start_button = ttk.Button(
            action_frame,
            text="🚀 Start Translation",
            command=self.start_processing_thread,
            style="Accent.TButton",
            width=20
        )
        self.start_button.pack(side=RIGHT, padx=(10, 0))

    def create_progress_section(self, parent):
        """Create the progress and status section"""
        progress_frame = ttk.LabelFrame(parent, text="📊 Progress", padding=20)
        progress_frame.pack(fill=X)
        
        # Current status
        self.status_var = tk.StringVar(value="Ready to translate videos")
        status_label = ttk.Label(
            progress_frame,
            textvariable=self.status_var,
            font=("Segoe UI", 10)
        )
        status_label.pack(anchor=W, pady=(0, 10))
        
        # Progress bar with modern styling
        self.progress_var = tk.DoubleVar()
        if MODERN_UI_AVAILABLE:
            self.progressbar = ttk.Progressbar(
                progress_frame,
                variable=self.progress_var,
                mode="determinate",
                style="info.Striped.Horizontal.TProgressbar"
            )
        else:
            self.progressbar = ttk.Progressbar(
                progress_frame,
                variable=self.progress_var,
                mode="determinate"
            )
        self.progressbar.pack(fill=X, pady=(0, 10))
        
        # Additional progress info
        progress_info_frame = ttk.Frame(progress_frame)
        progress_info_frame.pack(fill=X)
        
        self.progress_text = ttk.Label(
            progress_info_frame,
            text="",
            font=("Segoe UI", 9),
            foreground="gray"
        )
        self.progress_text.pack(side=LEFT)
        
        self.eta_label = ttk.Label(
            progress_info_frame,
            text="",
            font=("Segoe UI", 9),
            foreground="gray"
        )
        self.eta_label.pack(side=RIGHT)

    def setup_drag_drop(self):
        """Setup drag and drop functionality"""
        # This would require tkinterdnd2 library for full drag-drop support
        # For now, we'll add visual feedback for the drop zone
        def on_file_tree_drag_enter(event):
            self.file_tree.configure(style="info.Treeview")
        
        def on_file_tree_drag_leave(event):
            self.file_tree.configure(style="Treeview")
        
        # Bind events for visual feedback
        self.file_tree.bind("<Enter>", on_file_tree_drag_enter)
        self.file_tree.bind("<Leave>", on_file_tree_drag_leave)

    def add_files(self):
        """Add video files to the processing list"""
        selected_files = filedialog.askopenfilenames(
            title="Select Video Files",
            filetypes=(
                ("Video files", "*.mkv *.mp4 *.avi *.mov *.wmv *.flv *.webm"),
                ("All files", "*.*")
            )
        )
        
        added_count = 0
        for file_path_str in selected_files:
            file_path = Path(file_path_str)
            if file_path not in self.files_to_process:
                self.files_to_process.append(file_path)
                
                # Get file size
                try:
                    size_bytes = file_path.stat().st_size
                    size_str = self.format_file_size(size_bytes)
                except:
                    size_str = "Unknown"
                
                # Add to treeview
                self.file_tree.insert(
                    "", 
                    "end",
                    text="🎬",
                    values=(file_path.name, size_str, "Ready")
                )
                added_count += 1
        
        self.update_file_count()
        if added_count > 0:
            self.status_var.set(f"Added {added_count} video file(s)")

    def remove_files(self):
        """Remove selected files from the processing list"""
        selected_items = self.file_tree.selection()
        if not selected_items:
            if MODERN_UI_AVAILABLE:
                Messagebox.show_info("No Selection", "Please select files to remove.")
            else:
                messagebox.showinfo("No Selection", "Please select files to remove.")
            return
        
        # Get indices to remove (in reverse order)
        indices_to_remove = []
        for item in selected_items:
            index = self.file_tree.index(item)
            indices_to_remove.append(index)
        
        # Remove from list (in reverse order to maintain indices)
        for index in sorted(indices_to_remove, reverse=True):
            del self.files_to_process[index]
        
        # Remove from treeview
        for item in selected_items:
            self.file_tree.delete(item)
        
        self.update_file_count()
        self.status_var.set(f"Removed {len(selected_items)} file(s)")

    def clear_all_files(self):
        """Clear all files from the processing list"""
        if not self.files_to_process:
            return
        
        # Confirm clear all
        if MODERN_UI_AVAILABLE:
            result = Messagebox.yesno("Clear All Files", "Are you sure you want to remove all files?")
        else:
            result = messagebox.askyesno("Clear All Files", "Are you sure you want to remove all files?")
        
        if result:
            self.files_to_process.clear()
            for item in self.file_tree.get_children():
                self.file_tree.delete(item)
            self.update_file_count()
            self.status_var.set("All files cleared")

    def update_file_count(self):
        """Update the file count display"""
        count = len(self.files_to_process)
        if count == 0:
            self.file_count_label.config(text="No files selected")
            self.start_button.config(state="disabled")
        elif count == 1:
            self.file_count_label.config(text="1 file ready for translation")
            self.start_button.config(state="normal")
        else:
            self.file_count_label.config(text=f"{count} files ready for translation")
            self.start_button.config(state="normal")

    def format_file_size(self, size_bytes):
        """Format file size in human readable format"""
        if size_bytes == 0:
            return "0 B"
        size_names = ["B", "KB", "MB", "GB"]
        import math
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {size_names[i]}"

    def start_processing_thread(self):
        """Start the translation process in a separate thread"""
        global MAX_RPM, _rate_limiter, CONTEXT_WINDOW_SIZE
        
        if self.is_running:
            if MODERN_UI_AVAILABLE:
                Messagebox.show_warning("Processing", "Translation is already in progress.")
            else:
                messagebox.showwarning("Processing", "Translation is already in progress.")
            return
        
        if not self.files_to_process:
            if MODERN_UI_AVAILABLE:
                Messagebox.show_info("No Files", "Please add video files to translate.")
            else:
                messagebox.showinfo("No Files", "Please add video files to translate.")
            return
        
        # Validate and update settings
        try:
            rpm_value = self.rpm_var.get()
            if rpm_value <= 0:
                raise ValueError("RPM must be greater than 0")
            MAX_RPM = rpm_value
            _rate_limiter = TokenBucket(capacity=MAX_RPM, refill_rate=MAX_RPM / 60.0)
            
            context_value = self.context_var.get()
            if context_value < 0:
                raise ValueError("Context window must be 0 or greater")
            CONTEXT_WINDOW_SIZE = context_value
            
        except (ValueError, tk.TclError) as e:
            if MODERN_UI_AVAILABLE:
                Messagebox.show_error("Invalid Settings", f"Please check your settings: {e}")
            else:
                messagebox.showerror("Invalid Settings", f"Please check your settings: {e}")
            return
        
        # Start processing
        self.is_running = True
        self.total_files = len(self.files_to_process)
        self.current_file_count = 0
        
        # Update UI for processing state
        self.set_controls_state(False)
        self.start_button.config(text="⏸️ Processing...", state="disabled")
        self.progress_var.set(0)
        
        # Start processing thread
        thread = threading.Thread(target=self.processing_worker, daemon=True)
        thread.start()

    def set_controls_state(self, enabled: bool):
        """Enable or disable UI controls"""
        state = "normal" if enabled else "disabled"
        
        # File management buttons
        self.add_button.config(state=state)
        self.remove_button.config(state=state)
        self.clear_button.config(state=state)
        
        # Settings controls  
        for widget in [self.batch_var, self.rpm_var, self.context_var, self.preview_var]:
            try:
                # Find the widget associated with these variables
                for child in self.root.winfo_children():
                    self.set_widget_state_recursive(child, state)
            except:
                pass

    def set_widget_state_recursive(self, widget, state):
        """Recursively set widget states"""
        try:
            if hasattr(widget, 'config'):
                if isinstance(widget, (ttk.Entry, ttk.Checkbutton)):
                    widget.config(state=state)
            for child in widget.winfo_children():
                self.set_widget_state_recursive(child, state)
        except:
            pass

    def processing_worker(self):
        """Worker method for processing files"""
        batch_val = self.batch_var.get()
        preview_mode = self.preview_var.get()
        
        async def process_gui_videos():
            completed = 0
            semaphore = asyncio.Semaphore(2)  # Limit concurrent processing
            
            async def process_single_gui_video(video_path: Path, index: int):
                nonlocal completed
                async with semaphore:
                    # Update file status in treeview
                    self.root.after(0, self.update_file_status, index, "Processing...")
                    self.root.after(0, self.status_var.set, f"Processing: {video_path.name}")
                    
                    try:
                        # Use GUI-specific processing with review dialog
                        success = await self.process_video_with_review(video_path, batch_val, preview_mode, index)
                        if success:
                            self.root.after(0, self.update_file_status, index, "✅ Completed")
                        else:
                            self.root.after(0, self.update_file_status, index, "❌ Cancelled")
                    except Exception as e:
                        error_msg = f"Error processing {video_path.name}: {type(e).__name__} - {str(e)[:100]}"
                        print(error_msg)
                        self.root.after(0, self.update_file_status, index, "❌ Failed")
                        self.root.after(0, lambda: (
                            Messagebox.show_error("Processing Error", error_msg) if MODERN_UI_AVAILABLE 
                            else messagebox.showerror("Processing Error", error_msg)
                        ))
                    finally:
                        completed += 1
                        progress = (completed / self.total_files) * 100
                        self.root.after(0, self.progress_var.set, progress)
                        self.root.after(0, self.progress_text.config, text=f"Completed {completed}/{self.total_files} files")
            
            # Process all files
            tasks = [
                process_single_gui_video(video_path, index) 
                for index, video_path in enumerate(self.files_to_process)
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
            await close_http_client()
        
        # Run async processing
        asyncio.run(process_gui_videos())
        
        # Update UI when done
        self.root.after(0, self.processing_complete)

    def update_file_status(self, index: int, status: str):
        """Update the status of a file in the treeview"""
        try:
            items = self.file_tree.get_children()
            if index < len(items):
                item = items[index]
                current_values = list(self.file_tree.item(item, "values"))
                current_values[2] = status  # Status is the 3rd column
                self.file_tree.item(item, values=current_values)
        except:
            pass

    async def process_video_with_review(self, video_path: Path, batch_size: int, preview_mode: bool, file_index: int) -> bool:
        """Process a video file with GUI review dialog integration"""
        with tempfile.TemporaryDirectory(prefix="subtrans_") as td_name:
            tmpdir_path = Path(td_name)
            
            # Extract English subtitles
            eng_srt_path = extract_srt(video_path, tmpdir_path)
            if not eng_srt_path:
                print(f"  No English subtitle track found for {video_path.name}")
                return False

            try:
                subs = pysrt.open(str(eng_srt_path), encoding="utf-8")
            except Exception as e:
                print(f"  Error opening SRT file: {e}")
                return False
                
            if not subs:
                print(f"  Extracted SRT is empty or unreadable")
                return False

            # Handle preview mode
            if preview_mode:
                # Simplified preview for GUI - just show first few lines
                preview_subs_count = min(len(subs), 10)
                preview_indexed_lines = [(i, subs[i].text) for i in range(preview_subs_count)]
                
                if preview_indexed_lines:
                    translated_preview = await translate_batch_async(preview_indexed_lines)
                    preview_result = "\n".join([f"{i+1}. {line}" for i, (_, line) in enumerate(translated_preview[:3])])
                    
                    # Show preview dialog on main thread
                    continue_translation = await self.show_preview_dialog_async(video_path.name, preview_result)
                    
                    if not continue_translation:
                        return False

            # Perform full translation with potential review
            return await self.translate_with_review_dialog(subs, video_path, batch_size)

    async def show_preview_dialog_async(self, video_name: str, preview_text: str) -> bool:
        """Show preview dialog asynchronously on main thread"""
        result_container = [None]
        
        def show_dialog():
            try:
                if MODERN_UI_AVAILABLE:
                    from ttkbootstrap.dialogs import Messagebox
                    result = Messagebox.yesno(
                        title="Preview Translation",
                        message=f"Preview for {video_name}:\n\n{preview_text}\n\nContinue with full translation?",
                        parent=self.root
                    )
                else:
                    result = messagebox.askyesno(
                        "Preview Translation",
                        f"Preview for {video_name}:\n\n{preview_text}\n\nContinue with full translation?"
                    )
                result_container[0] = result
            except Exception as e:
                print(f"Error showing preview dialog: {e}")
                result_container[0] = True  # Default to continue
        
        # Schedule on main thread and wait for completion
        self.root.after(0, show_dialog)
        
        # Wait for dialog to complete
        while result_container[0] is None:
            await asyncio.sleep(0.1)
        
        return result_container[0]

    async def translate_with_review_dialog(self, subs: pysrt.SubRipFile, video_path: Path, batch_size: int) -> bool:
        """Translate subtitles and show review dialog"""
        # Store original subtitles
        original_subs = pysrt.SubRipFile()
        for sub in subs:
            original_subs.append(pysrt.SubRipItem(sub.index, sub.start, sub.end, sub.text))
        
        while True:  # Loop for "Try Again" functionality
            # Reset translations for retry
            translated_subs = pysrt.SubRipFile()
            for sub in original_subs:
                translated_subs.append(pysrt.SubRipItem(sub.index, sub.start, sub.end, sub.text))
            
            # Perform translation
            batches_with_context = list(build_batches_with_context(translated_subs, batch_size))
            tasks = [translate_batch_with_context_async(batch_data, context_lines) 
                     for batch_data, context_lines in batches_with_context]
            
            all_translated_lines = await asyncio.gather(*tasks, return_exceptions=True)

            # Apply translations
            for translated_batch in all_translated_lines:
                if isinstance(translated_batch, list):  # Not an exception
                    for original_idx, new_text in translated_batch:
                        if original_idx < len(translated_subs):
                            translated_subs[original_idx].text = new_text

            # Show review dialog on main thread
            result, final_subs = await self.show_review_dialog_async(
                original_subs,
                translated_subs,
                video_path.name
            )
            
            if result == 'accept':
                # Finalize and embed
                success = await _finalize_translation(final_subs, video_path)
                return success
            elif result == 'reject':
                return False  # User cancelled
            elif result == 'retry':
                # Continue loop for retry
                self.root.after(0, self.status_var.set, f"Retrying translation: {video_path.name}")
                continue
            else:
                return False  # Dialog was closed or error occurred

    async def show_review_dialog_async(self, original_subs: pysrt.SubRipFile, translated_subs: pysrt.SubRipFile, video_name: str):
        """Show the review dialog asynchronously on main thread"""
        result_container = [None]
        
        def show_dialog():
            try:
                dialog = SRTReviewDialog(self.root, original_subs, translated_subs, video_name)
                result = dialog.show_modal()
                result_container[0] = result
            except Exception as e:
                print(f"Error showing review dialog: {e}")
                result_container[0] = ('accept', translated_subs)  # Fallback to auto-accept
        
        # Schedule on main thread and wait for completion
        self.root.after(0, show_dialog)
        
        # Wait for dialog to complete
        while result_container[0] is None:
            await asyncio.sleep(0.1)
        
        return result_container[0]

    def processing_complete(self):
        """Called when all processing is complete"""
        self.is_running = False
        self.set_controls_state(True)
        self.start_button.config(text="🚀 Start Translation", state="normal")
        self.status_var.set(f"Translation complete! Processed {self.total_files} file(s)")
        self.progress_text.config(text="All translations completed successfully")
        
        if MODERN_UI_AVAILABLE:
            Messagebox.show_info(
                "Translation Complete", 
                f"Successfully processed {self.total_files} video file(s) with embedded Spanish subtitles!"
            )
        else:
            messagebox.showinfo(
                "Translation Complete",
                f"Successfully processed {self.total_files} video file(s) with embedded Spanish subtitles!"
            )

    def run(self):
        """Start the application"""
        self.root.mainloop()

# ─────────────────────────────────────────────────────────
#  SRT Review Dialog
# ─────────────────────────────────────────────────────────
class SRTReviewDialog:
    def __init__(self, parent, original_subs: pysrt.SubRipFile, translated_subs: pysrt.SubRipFile, video_name: str):
        self.parent = parent
        self.original_subs = original_subs
        self.translated_subs = translated_subs
        self.video_name = video_name
        self.result = None  # Will be 'accept', 'reject', or 'retry'
        self.edited_translations = {}  # Track edited lines
        
        self.create_dialog()
    
    def create_dialog(self):
        """Create the review dialog window"""
        if MODERN_UI_AVAILABLE:
            self.dialog = ttk.Toplevel(self.parent)
            self.dialog.style = self.parent.style
        else:
            self.dialog = tk.Toplevel(self.parent)
        
        self.dialog.title(f"Review Translation - {self.video_name}")
        self.dialog.geometry("1200x700")
        self.dialog.resizable(True, True)
        
        # Center dialog on parent
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        
        # Center on parent window
        self.center_dialog()
        
        self.build_dialog_ui()
    
    def center_dialog(self):
        """Center dialog on parent window"""
        self.dialog.update_idletasks()
        parent_x = self.parent.winfo_x()
        parent_y = self.parent.winfo_y()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        
        dialog_width = self.dialog.winfo_width()
        dialog_height = self.dialog.winfo_height()
        
        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2
        
        self.dialog.geometry(f"{dialog_width}x{dialog_height}+{x}+{y}")
    
    def build_dialog_ui(self):
        """Build the dialog UI components"""
        main_frame = ttk.Frame(self.dialog, padding=20)
        main_frame.pack(fill=BOTH, expand=True)
        
        # Header
        self.create_header(main_frame)
        
        # Search/Filter section
        self.create_search_section(main_frame)
        
        # Main content area with side-by-side comparison
        self.create_comparison_view(main_frame)
        
        # Action buttons
        self.create_action_buttons(main_frame)
    
    def create_header(self, parent):
        """Create header with title and stats"""
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill=X, pady=(0, 20))
        
        # Title
        title_label = ttk.Label(
            header_frame,
            text="📝 Review Translation Results",
            font=("Segoe UI", 18, "bold")
        )
        title_label.pack(anchor=W)
        
        # Stats
        total_lines = len(self.translated_subs)
        stats_text = f"Video: {self.video_name} • {total_lines} subtitle lines translated"
        stats_label = ttk.Label(
            header_frame,
            text=stats_text,
            font=("Segoe UI", 10),
            foreground="gray"
        )
        stats_label.pack(anchor=W, pady=(5, 0))
        
        # Instructions
        instructions = "Review the translations below. Click on Spanish text to edit. Choose your action when ready."
        inst_label = ttk.Label(
            header_frame,
            text=instructions,
            font=("Segoe UI", 9),
            foreground="gray"
        )
        inst_label.pack(anchor=W, pady=(5, 0))
        
        ttk.Separator(header_frame, orient=HORIZONTAL).pack(fill=X, pady=(10, 0))
    
    def create_search_section(self, parent):
        """Create search/filter section"""
        search_frame = ttk.Frame(parent)
        search_frame.pack(fill=X, pady=(0, 15))
        
        ttk.Label(search_frame, text="🔍 Search:", font=("Segoe UI", 10)).pack(side=LEFT, padx=(0, 10))
        
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=30)
        search_entry.pack(side=LEFT, padx=(0, 10))
        search_entry.bind('<KeyRelease>', self.on_search_changed)
        
        search_btn = ttk.Button(search_frame, text="Clear", command=self.clear_search, width=8)
        search_btn.pack(side=LEFT)
        
        # Filter options
        filter_frame = ttk.Frame(search_frame)
        filter_frame.pack(side=RIGHT)
        
        self.show_context_var = tk.BooleanVar(value=True)
        context_check = ttk.Checkbutton(
            filter_frame,
            text="Show context indicators",
            variable=self.show_context_var,
            command=self.refresh_view
        )
        context_check.pack(side=LEFT, padx=(20, 0))
    
    def create_comparison_view(self, parent):
        """Create the main side-by-side comparison view"""
        # Container for the comparison
        comparison_frame = ttk.LabelFrame(parent, text="Translation Comparison", padding=15)
        comparison_frame.pack(fill=BOTH, expand=True, pady=(0, 20))
        
        # Headers
        headers_frame = ttk.Frame(comparison_frame)
        headers_frame.pack(fill=X, pady=(0, 10))
        
        ttk.Label(
            headers_frame,
            text="⏱️ Time",
            font=("Segoe UI", 10, "bold"),
            width=12
        ).pack(side=LEFT, padx=(0, 10))
        
        ttk.Label(
            headers_frame,
            text="🇺🇸 Original English",
            font=("Segoe UI", 10, "bold")
        ).pack(side=LEFT, fill=X, expand=True, padx=(0, 10))
        
        ttk.Label(
            headers_frame,
            text="🇪🇸 Spanish Translation",
            font=("Segoe UI", 10, "bold")
        ).pack(side=RIGHT, fill=X, expand=True)
        
        # Scrollable content area
        content_container = ttk.Frame(comparison_frame)
        content_container.pack(fill=BOTH, expand=True)
        
        # Create canvas with scrollbar for subtitle lines
        self.canvas = tk.Canvas(content_container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(content_container, orient=VERTICAL, command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        self.canvas.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)
        
        # Bind mouse wheel to canvas
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # Populate with subtitle lines
        self.populate_subtitle_lines()
    
    def populate_subtitle_lines(self):
        """Populate the scrollable frame with subtitle line entries"""
        self.line_widgets = []
        
        for i, (orig_sub, trans_sub) in enumerate(zip(self.original_subs, self.translated_subs)):
            line_frame = self.create_subtitle_line(i, orig_sub, trans_sub)
            self.line_widgets.append(line_frame)
    
    def create_subtitle_line(self, index: int, orig_sub, trans_sub):
        """Create a single subtitle line entry"""
        # Main container for this line
        line_frame = ttk.Frame(self.scrollable_frame, padding=5)
        line_frame.pack(fill=X, pady=2)
        
        # Alternate background colors
        if index % 2 == 0:
            if MODERN_UI_AVAILABLE:
                line_frame.configure(style="Card.TFrame")
        
        # Time column
        time_text = f"{orig_sub.start} → {orig_sub.end}"
        time_label = ttk.Label(
            line_frame,
            text=time_text,
            font=("Consolas", 8),
            width=20,
            foreground="gray"
        )
        time_label.pack(side=LEFT, padx=(0, 10), anchor=N)
        
        # Original English text
        orig_frame = ttk.Frame(line_frame)
        orig_frame.pack(side=LEFT, fill=X, expand=True, padx=(0, 10))
        
        orig_text = ttk.Label(
            orig_frame,
            text=orig_sub.text,
            font=("Segoe UI", 9),
            wraplength=350,
            justify=LEFT
        )
        orig_text.pack(anchor=W)
        
        # Spanish translation (editable)
        trans_frame = ttk.Frame(line_frame)
        trans_frame.pack(side=RIGHT, fill=X, expand=True)
        
        # Check if this translation has issues
        is_problematic = self.is_translation_problematic(orig_sub.text, trans_sub.text)
        
        # Create text widget for editing
        text_widget = tk.Text(
            trans_frame,
            font=("Segoe UI", 9),
            height=2,
            width=40,
            wrap=tk.WORD,
            relief=FLAT,
            borderwidth=1,
            background="white" if not is_problematic else "#fff5f5"
        )
        text_widget.insert('1.0', trans_sub.text)
        text_widget.pack(fill=X)
        
        # Bind editing events
        text_widget.bind('<KeyRelease>', lambda e, idx=index: self.on_translation_edited(idx, e))
        text_widget.bind('<FocusIn>', lambda e, tw=text_widget: tw.configure(relief=SOLID, borderwidth=2))
        text_widget.bind('<FocusOut>', lambda e, tw=text_widget: tw.configure(relief=FLAT, borderwidth=1))
        
        # Add warning indicator if problematic
        if is_problematic:
            warning_label = ttk.Label(
                trans_frame,
                text="⚠️ May need review",
                font=("Segoe UI", 8),
                foreground="orange"
            )
            warning_label.pack(anchor=W, pady=(2, 0))
        
        # Store widget reference for later access
        setattr(line_frame, 'text_widget', text_widget)
        setattr(line_frame, 'index', index)
        
        return line_frame
    
    def is_translation_problematic(self, original: str, translation: str) -> bool:
        """Check if a translation might be problematic"""
        # Check for untranslated markers
        if "[UNTRANSLATED]" in translation or "[FAILED_TRANSLATION]" in translation:
            return True
        
        # Check if it still looks English
        if looks_english(translation):
            return True
        
        # Check for extreme length differences
        if len(translation) > len(original) * 3 + 50:
            return True
        
        if len(original) > 10 and len(translation) < len(original) * 0.2:
            return True
        
        return False
    
    def on_translation_edited(self, index: int, event):
        """Handle when a translation is edited"""
        widget = event.widget
        new_text = widget.get('1.0', 'end-1c')
        self.edited_translations[index] = new_text
    
    def on_search_changed(self, event):
        """Handle search text changes"""
        # This would filter visible lines based on search term
        # For now, just placeholder
        pass
    
    def clear_search(self):
        """Clear search and show all lines"""
        self.search_var.set("")
        self.refresh_view()
    
    def refresh_view(self):
        """Refresh the view based on current filters"""
        # Placeholder for filtering logic
        pass
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def create_action_buttons(self, parent):
        """Create the action buttons at the bottom"""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=X, pady=(10, 0))
        
        # Statistics on the left
        stats_frame = ttk.Frame(button_frame)
        stats_frame.pack(side=LEFT, fill=X, expand=True)
        
        total_lines = len(self.translated_subs)
        problematic_count = sum(
            1 for i, (orig, trans) in enumerate(zip(self.original_subs, self.translated_subs))
            if self.is_translation_problematic(orig.text, trans.text)
        )
        
        stats_text = f"📊 {total_lines} total lines"
        if problematic_count > 0:
            stats_text += f" • ⚠️ {problematic_count} may need review"
        
        stats_label = ttk.Label(stats_frame, text=stats_text, font=("Segoe UI", 9))
        stats_label.pack(anchor=W)
        
        # Action buttons on the right
        actions_frame = ttk.Frame(button_frame)
        actions_frame.pack(side=RIGHT)
        
        # Try Again button
        retry_btn = ttk.Button(
            actions_frame,
            text="🔄 Try Again",
            command=self.try_again,
            width=15
        )
        retry_btn.pack(side=LEFT, padx=(0, 10))
        
        # Reject button
        reject_btn = ttk.Button(
            actions_frame,
            text="❌ Reject",
            command=self.reject_translation,
            width=15
        )
        reject_btn.pack(side=LEFT, padx=(0, 10))
        
        # Accept All button (primary action)
        accept_btn = ttk.Button(
            actions_frame,
            text="✅ Accept All",
            command=self.accept_translation,
            style="Accent.TButton" if MODERN_UI_AVAILABLE else "TButton",
            width=15
        )
        accept_btn.pack(side=LEFT)
    
    def accept_translation(self):
        """Accept the translation with any edits"""
        # Apply any edits made by the user
        for index, edited_text in self.edited_translations.items():
            if index < len(self.translated_subs):
                self.translated_subs[index].text = edited_text
        
        self.result = 'accept'
        self.dialog.destroy()
    
    def reject_translation(self):
        """Reject the translation entirely"""
        self.result = 'reject'
        self.dialog.destroy()
    
    def try_again(self):
        """Try translation again (with possibly different settings)"""
        self.result = 'retry'
        self.dialog.destroy()
    
    def show_modal(self):
        """Show the dialog modally and return the result"""
        self.dialog.wait_window()
        return self.result, self.translated_subs

# Legacy App class for backward compatibility
class App:
    def __init__(self):
        self.modern_app = ModernSubtitleTranslatorApp()
    
    def mainloop(self):
        self.modern_app.run()

# ─────────────────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    if len(sys.argv) > 1:
        cli_main()
    else:
        app_instance = App()
        app_instance.mainloop()