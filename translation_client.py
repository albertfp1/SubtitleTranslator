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

import asyncio
import json
import os
import re
import time
from pathlib import Path
from textwrap import dedent
from collections import deque, OrderedDict
import random

import pysrt
import httpx  # For Chutes API
from tqdm.asyncio import tqdm
from ffmpeg_utils import embed_spanish_srt

# ─────────────────────────────────────────────────────────
#  Chutes AI Configuration
# ─────────────────────────────────────────────────────────
CHUTES_API_KEY = os.getenv("CHUTES_API_KEY")
CHUTES_API_ENDPOINT = "https://llm.chutes.ai/v1/chat/completions"
CHUTES_MODEL_ID = "deepseek-ai/DeepSeek-V3-0324"

# General Configuration
TARGET_LANG            = "es-ES"
AVG_TOKENS_PER_LINE    = 30
IDEAL_TOKENS_PER_BATCH = 15_000
PROMPT_OVERHEAD_TOKENS = 200
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


def _require_api_key() -> str:
    """Return the API key or raise a helpful error if missing"""
    key = CHUTES_API_KEY or os.getenv("CHUTES_API_KEY")
    if not key:
        raise RuntimeError(
            "CHUTES_API_KEY environment variable is not set; cannot call translation API."
        )
    return key
# ─────────────────────────────────────────────────────────
#  Translation Cache (LRU with optional persistence)
# ─────────────────────────────────────────────────────────
_CACHE_MAX_ENTRIES = int(os.getenv("TRANSLATION_CACHE_MAX", "1000"))
_CACHE_FILE_ENV = os.getenv("TRANSLATION_CACHE_FILE")
_CACHE_FILE_PATH: Path | None = Path(_CACHE_FILE_ENV) if _CACHE_FILE_ENV else None

_translation_cache: "OrderedDict[str, str]" = OrderedDict()
_cache_hits = 0
_cache_misses = 0


def _load_cache_from_file():
    global _translation_cache
    if _CACHE_FILE_PATH and _CACHE_FILE_PATH.exists():
        try:
            with open(_CACHE_FILE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                # Limit entries when loading
                for k, v in list(data.items())[:_CACHE_MAX_ENTRIES]:
                    _translation_cache[k] = v
        except Exception:
            pass


def _save_cache_to_file():
    if _CACHE_FILE_PATH:
        try:
            with open(_CACHE_FILE_PATH, "w", encoding="utf-8") as f:
                json.dump(_translation_cache, f, ensure_ascii=False)
        except Exception:
            pass


def configure_translation_cache(max_entries: int | None = None, cache_file: str | None = None) -> None:
    """Configure cache size and optional persistence path"""
    global _CACHE_MAX_ENTRIES, _CACHE_FILE_PATH
    if max_entries is not None:
        _CACHE_MAX_ENTRIES = max_entries
    if cache_file is not None:
        _CACHE_FILE_PATH = Path(cache_file)
    _translation_cache.clear()
    _load_cache_from_file()


def _cache_get(key: str) -> str | None:
    global _cache_hits, _cache_misses
    key = key.strip().lower()
    if key in _translation_cache:
        value = _translation_cache.pop(key)
        _translation_cache[key] = value  # move to end (most recently used)
        _cache_hits += 1
        return value
    _cache_misses += 1
    return None


def _cache_set(key: str, value: str) -> None:
    key = key.strip().lower()
    if key in _translation_cache:
        _translation_cache.pop(key)
    _translation_cache[key] = value
    if len(_translation_cache) > _CACHE_MAX_ENTRIES:
        _translation_cache.popitem(last=False)  # remove least recently used
    _save_cache_to_file()


def get_cache_stats() -> tuple[int, int]:
    """Returns (hits, misses) for cache statistics"""
    return _cache_hits, _cache_misses


def clear_cache():
    """Clear the translation cache"""
    global _translation_cache, _cache_hits, _cache_misses
    _translation_cache.clear()
    _cache_hits = 0
    _cache_misses = 0
    _save_cache_to_file()


_load_cache_from_file()

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
            # First, refill based on elapsed time since last acquisition
            now = time.monotonic()
            elapsed = now - self.last_refill
            if elapsed > 0:
                self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)

            # If not enough tokens, wait for the required amount
            if self.tokens < tokens:
                wait_time = (tokens - self.tokens) / self.refill_rate
                await asyncio.sleep(wait_time)
                now = time.monotonic()
                elapsed = now - self.last_refill
                if elapsed > 0:
                    self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)

            # Update timestamp and deduct tokens
            self.last_refill = now
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

COMMON_WORDS = {
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

_COMMON_WORDS_LOWER = {w.lower() for w in COMMON_WORDS}

def is_common_word(word: str) -> bool:
    """Check if a word is a common English word that shouldn't be treated as a name"""
    return word.lower() in _COMMON_WORDS_LOWER

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
        cached = _cache_get(text)
        if cached is not None:
            cached_results.append((idx, cached))
        else:
            uncached_lines.append((idx, text))
    
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

    api_key = _require_api_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
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
                    _cache_set(original_line_text, spa_lines[k])
                
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
        cached = _cache_get(text)
        if cached is not None:
            cached_results.append((idx, cached))
        else:
            uncached_lines.append((idx, text))
    
    # If all lines were cached, return immediately
    if not uncached_lines:
        return cached_results
    
    # Process uncached lines
    idx_list = [i for i, _ in uncached_lines]
    en_lines = [text for _, text in uncached_lines]
    
    en_lines_json_array_string = json.dumps(en_lines, ensure_ascii=False)
    user_prompt_content = f"Translate the English subtitle lines in the following JSON array to {TARGET_LANG}:\n{en_lines_json_array_string}"

    api_key = _require_api_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
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
                    _cache_set(original_line_text, spa_lines[k])
                
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
        embed_spanish_srt(video_path, temp_spanish_srt, TARGET_LANG)
        
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



