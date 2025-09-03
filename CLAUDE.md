# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AISRT (AI Subtitle Translator) is a Python tool that extracts English subtitles from video files and translates them to Castilian Spanish using the Chutes AI API (DeepSeek model). The tool can run via command line or GUI interface and saves translated SRT files externally alongside the original video files.

## Key Components

- **translate_srt_gemini.py**: Main application file containing both CLI and GUI interfaces
- **requirements.txt**: Python dependencies including google-generativeai, pysrt, tqdm, ffmpeg-python, and ttkbootstrap
- **start.bat**: Simple batch file with example usage command
- **Episodes/**: Directory for video files (currently empty)

## Dependencies and Setup

```bash
pip install -r requirements.txt
```

Required external dependency: `ffmpeg` must be installed and available in PATH for video subtitle extraction.

## Running the Application

### GUI Mode (default)
```bash
python translate_srt_gemini.py
```

### CLI Mode
```bash
python translate_srt_gemini.py video1.mkv video2.mp4 [options]
```

### CLI Options
- `--batch N`: Lines per API request (default: 60, use 0 for auto token-based batching)
- `--preview`: Translate first few lines and ask for confirmation
- `--rpm N`: Max requests per minute to API (default: 100)

## Configuration

Key configuration constants in translate_srt_gemini.py:
- `CHUTES_API_KEY`: API key for Chutes AI service
- `CHUTES_MODEL_ID`: DeepSeek-V3-0324 model
- `TARGET_LANG`: "es-ES" (Castilian Spanish)
- `MAX_RPM`: Rate limiting for API calls
- `IDEAL_TOKENS_PER_BATCH`: Token budget for batching (15,000)

## Architecture

### Translation Pipeline
1. **Subtitle Extraction**: Uses ffmpeg to extract English subtitle tracks from video files
2. **Batching**: Groups subtitle lines for efficient API usage (token-based or line-based)
3. **Translation**: Calls Chutes AI API with structured prompts for JSON array responses
4. **Quality Control**: Validates translations and retries individual lines if needed
5. **File Management**: Deletes existing SRT files and saves new translated versions

### Rate Limiting
Implements async rate limiting with `throttle_async()` to respect API rate limits and handle retries with exponential backoff.

### Error Handling
- Automatic retry logic for failed API calls
- Batch splitting for large requests that fail
- Single-line retry for suspicious translations
- Graceful degradation with fallback messages

## File Naming Convention

Output files follow the pattern: `{video_stem}.translated.{TARGET_LANG}.srt`

Example: `movie.mkv` → `movie.translated.es-ES.srt`