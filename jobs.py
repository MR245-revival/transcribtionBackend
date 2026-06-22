import os
import time
import uuid
import json
from dataclasses import dataclass, asdict
from threading import Thread, Lock
from typing import Dict, Optional, List, Tuple, Any

from audio_utils import convert_to_wav_16k_mono, probe_duration_seconds

# faster-whisper
from faster_whisper import WhisperModel

# pyannote (optional speaker diarization)
from pyannote.audio import Pipeline


UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
EXPORT_DIR = os.path.join(UPLOAD_DIR, "exports")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

# ---------------- Whisper Model Cache ----------------
_model_lock = Lock()
_model: Optional[WhisperModel] = None


def get_whisper_model() -> WhisperModel:
    """Cached WhisperModel. GPU: device="cuda", compute_type="float16" ist ideal."""
    global _model
    with _model_lock:
        if _model is None:
            _model = WhisperModel(
                "large-v3",
                device="cuda",
                compute_type="float16",
            )
        return _model


# ---------------- Pyannote Pipeline Cache ----------------
_diar_lock = Lock()
_diar: Optional[Pipeline] = None


def get_diarization_pipeline() -> Pipeline:
    """
    Cached pyannote diarization pipeline.
    Requires HF_TOKEN or PYANNOTE_TOKEN environment variable.
    """
    global _diar
    with _diar_lock:
        if _diar is None:
            token = os.environ.get("HF_TOKEN") or os.environ.get("PYANNOTE_TOKEN")
            print(
                "HF_TOKEN vorhanden:", bool(os.environ.get("HF_TOKEN")),
                "PYANNOTE_TOKEN vorhanden:", bool(os.environ.get("PYANNOTE_TOKEN"))
            )
            if not token:
                raise RuntimeError("pyannote/Token fehlt (HF_TOKEN oder PYANNOTE_TOKEN).")

            # Kompatibilität über pyannote-Versionen hinweg:
            # je nach Version heißt das Argument "use_auth_token" oder wird indirekt via HF env genutzt.
            try:
                _diar = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization",
                    use_auth_token=token
                )
            except TypeError:
                # fallback: manche Versionen erwarten anderes/kein kwarg
                _diar = Pipeline.from_pretrained("pyannote/speaker-diarization")

            # wenn CUDA verfügbar, Pipeline auf GPU
            try:
                import torch
                if torch.cuda.is_available():
                    _diar.to(torch.device("cuda"))
            except Exception:
                pass

        return _diar


@dataclass
class Job:
    id: str
    kind: str  # "transcribe"
    status: str  # "queued" | "running" | "done" | "error"
    progress: int  # 0..100
    eta_seconds: Optional[int]
    message: str
    created_at: float
    started_at: Optional[float]
    finished_at: Optional[float]
    input_path: Optional[str]

    # UI output
    result_text: Optional[str]

    # Export files
    result_txt_path: Optional[str]
    result_json_path: Optional[str]

    error: Optional[str]


_jobs: Dict[str, Job] = {}
_lock = Lock()


def create_job(kind: str, input_path: str) -> Job:
    job_id = str(uuid.uuid4())
    job = Job(
        id=job_id,
        kind=kind,
        status="queued",
        progress=0,
        eta_seconds=None,
        message="Wartet…",
        created_at=time.time(),
        started_at=None,
        finished_at=None,
        input_path=input_path,
        result_text=None,
        result_txt_path=None,
        result_json_path=None,
        error=None,
    )
    with _lock:
        _jobs[job_id] = job
    return job


def get_job(job_id: str) -> Optional[Job]:
    with _lock:
        return _jobs.get(job_id)


def update_job(job_id: str, **kwargs) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return
        for k, v in kwargs.items():
            setattr(job, k, v)


def job_to_dict(job: Job) -> dict:
    d = asdict(job)

    # Für Frontend praktische Download-URLs (keine absoluten Pfade rausgeben)
    d["result_txt_url"] = f"/jobs/{job.id}/result.txt" if job.result_txt_path else None
    d["result_json_url"] = f"/jobs/{job.id}/result.json" if job.result_json_path else None
    return d


def start_transcribe_job(job_id: str, with_timestamps: bool = False, diarize_speakers: bool = False) -> None:
    t = Thread(
        target=_run_transcribe,
        args=(job_id, with_timestamps, diarize_speakers),
        daemon=True
    )
    t.start()


def _estimate_eta_from_duration(duration_s: float, stage: str) -> int:
    overhead = 8
    if stage == "convert":
        return int(min(20, duration_s * 0.05 + 3))
    if stage == "diarize":
        return int(duration_s * 0.7 + overhead)
    if stage == "transcribe":
        return int(duration_s * 0.5 + overhead)
    return int(overhead)


def _fmt_ts_cs(seconds: float) -> str:
    """MM:SS.xx (Hundertstel)"""
    if seconds < 0:
        seconds = 0
    m = int(seconds // 60)
    s = int(seconds % 60)
    cs = int((seconds - int(seconds)) * 100)
    return f"{m:02d}:{s:02d}.{cs:02d}"


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _diarize_to_segments(wav_path: str) -> List[Tuple[float, float, str]]:
    pipeline = get_diarization_pipeline()
    diar = pipeline(wav_path)

    segs: List[Tuple[float, float, str]] = []
    for segment, _, speaker in diar.itertracks(yield_label=True):
        segs.append((float(segment.start), float(segment.end), str(speaker)))
    return segs


def _assign_speaker_to_whisper(
    whisper: List[Tuple[float, float, str]],
    diar: List[Tuple[float, float, str]],
) -> List[Tuple[float, float, str, str]]:
    """For each whisper segment pick the speaker with max overlap. Return (start,end,speaker,text)."""
    out: List[Tuple[float, float, str, str]] = []
    for ws, we, text in whisper:
        best_spk = "SPEAKER_0"
        best_ov = 0.0
        for ds, de, spk in diar:
            ov = _overlap(ws, we, ds, de)
            if ov > best_ov:
                best_ov = ov
                best_spk = spk
        out.append((ws, we, best_spk, text))
    return out


def _write_exports(
    job_id: str,
    txt: str,
    payload: dict,
) -> Tuple[str, str]:
    """
    Schreibt TXT + JSON nach uploads/exports/
    Return: (txt_path, json_path)
    """
    os.makedirs(EXPORT_DIR, exist_ok=True)
    txt_path = os.path.join(EXPORT_DIR, f"{job_id}.txt")
    json_path = os.path.join(EXPORT_DIR, f"{job_id}.json")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(txt)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return txt_path, json_path


def _run_transcribe(job_id: str, with_timestamps: bool, diarize_speakers: bool) -> None:
    job = get_job(job_id)
    if not job:
        return

    update_job(job_id, status="running", started_at=time.time(), message="Starte…", progress=1, eta_seconds=60)

    try:
        if not job.input_path:
            raise RuntimeError("No input file")
        input_path = job.input_path

        # 1) Duration for ETA
        update_job(job_id, message="Analysiere Audio…", progress=5)
        try:
            duration_s = probe_duration_seconds(input_path)
        except Exception:
            duration_s = 0.0

        # 2) Convert to WAV 16k mono
        update_job(
            job_id,
            message="Konvertiere in WAV (16kHz mono)…",
            progress=15,
            eta_seconds=_estimate_eta_from_duration(duration_s, "convert")
        )
        wav_path = convert_to_wav_16k_mono(input_path, UPLOAD_DIR)

        if duration_s <= 0:
            try:
                duration_s = probe_duration_seconds(wav_path)
            except Exception:
                duration_s = 0.0

        warning_lines: List[str] = []
        diar_segments: Optional[List[Tuple[float, float, str]]] = None

        # 3) (Optional) diarize speakers — Fallback statt Error
        if diarize_speakers:
            update_job(
                job_id,
                message="Erkenne Sprecher (Diarization)…",
                progress=25,
                eta_seconds=_estimate_eta_from_duration(duration_s, "diarize")
            )
            try:
                diar_segments = _diarize_to_segments(wav_path)
            except Exception as e:
                diar_segments = None
                diarize_speakers = False  # fallback
                warning_lines.append(
                    "[Hinweis] Sprechertrennung war aktiviert, aber Diarization ist nicht verfügbar "
                    f"({str(e)}). Ausgabe ohne Speaker-Labels."
                )

        # 4) Whisper model (cached)
        update_job(
            job_id,
            message="Lade Whisper Modell…",
            progress=35 if not diarize_speakers else 45,
            eta_seconds=_estimate_eta_from_duration(duration_s, "transcribe")
        )
        model = get_whisper_model()

        # 5) Transcribe (LIVE Progress)
        base_progress = 35 if not diarize_speakers else 45
        update_job(
            job_id,
            message="Transkribiere…",
            progress=base_progress,
            eta_seconds=_estimate_eta_from_duration(duration_s, "transcribe")
        )

        start_t = time.time()
        segments_iter, info = model.transcribe(
            wav_path,
            language=None,
            vad_filter=True,
            beam_size=5,
        )

        whisper_segments: List[Tuple[float, float, str]] = []
        last_progress = base_progress
        denom = duration_s if duration_s > 0 else None

        for seg in segments_iter:
            ws, we = float(seg.start), float(seg.end)
            text = (seg.text or "").strip()
            whisper_segments.append((ws, we, text))

            # progress anhand "we / duration"
            if denom:
                frac = min(1.0, max(0.0, float(we) / denom))
                p = base_progress + int(frac * (95 - base_progress))
            else:
                p = min(95, last_progress + 1)

            if p > last_progress:
                elapsed = time.time() - start_t
                if p > base_progress + 5:
                    done_frac = (p - base_progress) / max(1, (95 - base_progress))
                    remaining = max(0.0, 1.0 - done_frac)
                    eta = int(elapsed * (remaining / max(0.01, done_frac)))
                else:
                    eta = _estimate_eta_from_duration(duration_s, "transcribe")

                update_job(job_id, progress=p, eta_seconds=max(0, eta))
                last_progress = p

        # 6) Compose result (TXT + JSON)
        update_job(job_id, message="Erstelle Ausgabe…", progress=96, eta_seconds=3)

        json_segments: List[dict] = []

        if diarize_speakers and diar_segments is not None:
            assigned = _assign_speaker_to_whisper(whisper_segments, diar_segments)
            lines = []
            for ws, we, spk, text in assigned:
                if not text:
                    continue

                seg_obj = {
                    "start": ws,
                    "end": we,
                    "start_ts": _fmt_ts_cs(ws),
                    "end_ts": _fmt_ts_cs(we),
                    "speaker": spk,
                    "text": text,
                }
                json_segments.append(seg_obj)

                if with_timestamps:
                    lines.append(f"[{_fmt_ts_cs(ws)}-{_fmt_ts_cs(we)}] {spk}: {text}")
                else:
                    lines.append(f"{spk}: {text}")

            final_text = "\n".join(lines).strip()

        else:
            # no diarization
            lines = []
            for ws, we, text in whisper_segments:
                if not text:
                    continue

                seg_obj = {
                    "start": ws,
                    "end": we,
                    "start_ts": _fmt_ts_cs(ws),
                    "end_ts": _fmt_ts_cs(we),
                    "speaker": None,
                    "text": text,
                }
                json_segments.append(seg_obj)

                if with_timestamps:
                    lines.append(f"[{_fmt_ts_cs(ws)}-{_fmt_ts_cs(we)}] {text}")
                else:
                    lines.append(text)

            final_text = ("\n".join(lines) if with_timestamps else " ".join(lines)).strip()

        if not final_text:
            final_text = "(leer) — Whisper hat keinen Text erkannt."

        if warning_lines:
            final_text = final_text.strip() + "\n\n" + "\n".join(warning_lines)

        payload: dict[str, Any] = {
            "job_id": job_id,
            "kind": "transcribe",
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": time.time(),
            "options": {
                "with_timestamps": with_timestamps,
                "diarize_speakers": diarize_speakers,
            },
            "input": {
                "original_path": input_path,
                "wav_path": wav_path,
                "duration_seconds": duration_s,
            },
            "segments": json_segments,
            "text": final_text,
            "warnings": warning_lines,
        }

        txt_path, json_path = _write_exports(job_id, final_text, payload)

        update_job(
            job_id,
            result_text=final_text,
            result_txt_path=txt_path,
            result_json_path=json_path,
            message="Fertig",
            progress=100,
            status="done",
            finished_at=time.time(),
            eta_seconds=0,
        )

    except Exception as e:
        update_job(job_id, status="error", error=str(e), message="Fehler", finished_at=time.time())