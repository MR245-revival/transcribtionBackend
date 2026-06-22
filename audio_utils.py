import os
import subprocess
import uuid
import json
import shutil

FFMPEG_EXE = os.environ.get("FFMPEG_EXE")
FFPROBE_EXE = os.environ.get("FFPROBE_EXE")

def _resolve(tool: str, override: str | None) -> str:
    if override and os.path.exists(override):
        return override
    found = shutil.which(tool)
    if found:
        return found
    return tool

def _run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout)[-2000:])
    return p.stdout

def probe_duration_seconds(path: str) -> float:
    ffprobe = _resolve("ffprobe", FFPROBE_EXE)
    out = _run([
        ffprobe,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        path
    ])
    data = json.loads(out)
    return float(data["format"]["duration"])

def convert_to_wav_16k_mono(input_path: str, out_dir: str) -> str:
    ffmpeg = _resolve("ffmpeg", FFMPEG_EXE)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{uuid.uuid4()}.wav")

    ext = os.path.splitext(input_path)[1].lower()

    if ext in [".dss", ".ds2"]:
        af = ",".join([
            "dcshift=0",
            "highpass=f=80",
            "lowpass=f=3800",
            "aresample=16000:resampler=soxr:precision=28"
        ])
        cmd = [
            ffmpeg, "-y",
            "-i", input_path,
            "-vn",
            "-af", af,
            "-ac", "1",
            "-c:a", "pcm_s16le",
            out_path
        ]
    else:
        cmd = [
            ffmpeg, "-y",
            "-i", input_path,
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-c:a", "pcm_s16le",
            out_path
        ]

    _run(cmd)
    return out_path