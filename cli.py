import argparse
import asyncio
import tempfile
from pathlib import Path

import pysrt

from ffmpeg_utils import extract_srt
from translation_client import (
    translate_batch_async,
    _translate_and_save_srt_async_job,
    close_http_client,
    TokenBucket,
    MAX_RPM,
    _rate_limiter,
)


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
            print(
                f"  Extracted SRT {eng_srt_path.name} is empty or unreadable. Possibly a bitmap (PGS/VobSub) subtitle. Skipped."
            )
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
                    for _, line in translated_preview[: min(len(translated_preview), 5)]:
                        print(f"    {line}")

                asyncio.run(run_preview())

            ans = input("\nLooks good? Continue with full translation? (y/N): ").strip().lower()
            if ans != "y":
                print(f"  Skipped full translation for {video_file.name} by user.")
                return

        try:
            asyncio.run(_translate_and_save_srt_async_job(subs, video_file, batch_size, show_review=False))
        except Exception as e:
            print(f"  Error during translation/saving for {video_file.name}: {type(e).__name__} - {e}")
            import traceback

            traceback.print_exc()


def process_multiple_videos_async(video_paths: list[Path], batch_size: int, preview_mode: bool = False, max_concurrent: int = 3):
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
        asyncio.run(asyncio.gather(*tasks, return_exceptions=True))

    asyncio.run(close_http_client())


def cli_main():
    global MAX_RPM, _rate_limiter

    parser = argparse.ArgumentParser(
        description="Subtitle translator using Chutes AI, embeds Spanish SRT into video."
    )
    parser.add_argument("videos", nargs="+", help="Paths to video files (MKV, MP4, etc.)")
    parser.add_argument(
        "--batch",
        type=int,
        default=60,
        help="Lines per API request. Use 0 for auto-batching by token budget. Default: 60 lines.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Translate only the first few lines and ask for confirmation.",
    )
    parser.add_argument(
        "--rpm", type=int, default=MAX_RPM, help=f"Max requests per minute to the API. Default: {MAX_RPM}."
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=3,
        help="Max number of videos to process concurrently. Default: 3.",
    )

    args = parser.parse_args()

    MAX_RPM = args.rpm
    _rate_limiter = TokenBucket(capacity=MAX_RPM, refill_rate=MAX_RPM / 60.0)

    video_paths = [Path(v) for v in args.videos]
    asyncio.run(process_multiple_videos_async(video_paths, args.batch, args.preview, args.concurrent))

    print("\nAll tasks completed.")
