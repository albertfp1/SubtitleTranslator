import subprocess
from pathlib import Path

FFMPEG_LOGLEVEL = "error"


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


def embed_spanish_srt(video_path: Path, spanish_srt_path: Path, target_lang: str):
    """Embeds Spanish SRT into video file alongside existing subtitles"""
    output_video = video_path.parent / f"{video_path.stem}.with_spanish{video_path.suffix}"

    print(f"  Embedding Spanish subtitles into {output_video.name}...")

    cmd = [
        "ffmpeg", "-v", FFMPEG_LOGLEVEL, "-y",
        "-i", str(video_path),
        "-i", str(spanish_srt_path),
        "-map", "0",
        "-map", "1:0",
        "-c", "copy",
        "-c:s:1", "srt",
        "-metadata:s:s:1", f"language={target_lang}",
        "-metadata:s:s:1", "title=Spanish",
        str(output_video),
    ]

    try:
        run_ffmpeg(cmd)
        print(f"  ✅ Video with Spanish subtitles saved as: {output_video.name}")
        return output_video
    except RuntimeError as e:
        print(f"  ❌ Failed to embed Spanish subtitles: {e}")
        raise
