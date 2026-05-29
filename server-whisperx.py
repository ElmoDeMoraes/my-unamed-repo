"""
WhisperX FastAPI server — transcrição de áudio com diarização de falantes.

Endpoints:
  GET  /health          → 200 {"status":"ok","model":"large-v3-turbo"} quando pronto
  POST /v1/transcribe   → transcreve áudio e retorna segmentos por falante
"""
import gc
import os
import logging
import subprocess
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

import torch
import whisperx
from whisperx.diarize import DiarizationPipeline
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = os.environ.get(
    "COMPUTE_TYPE", "int8_float16" if DEVICE == "cuda" else "int8"
)
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
WHISPERX_DIARIZATION_MODEL = os.environ.get(
    "WHISPERX_DIARIZATION_MODEL",
    "pyannote/speaker-diarization-community-1",
)
HF_TOKEN = os.environ.get("HF_TOKEN", "")
WHISPERX_BATCH_SIZE = int(os.environ.get("WHISPERX_BATCH_SIZE", "0"))
WHISPERX_DEFAULT_LANGUAGE = os.environ.get("WHISPERX_DEFAULT_LANGUAGE", "pt")
WHISPERX_DIARIZATION_MAX_AUDIO_SECONDS = float(
    os.environ.get("WHISPERX_DIARIZATION_MAX_AUDIO_SECONDS", "0")
)
WHISPERX_CHUNK_DURATION_S = int(
    os.environ.get("WHISPERX_CHUNK_DURATION_S", "1800")
)
WHISPERX_ENABLE_TF32 = os.environ.get("WHISPERX_ENABLE_TF32", "false").strip().lower() in {
    "1",
    "true",
    "yes",
}
LOGGER = logging.getLogger("whisperx.server")
logging.basicConfig(level=logging.INFO)

_model = None
_diarize_model = None
_align_models: dict = {}
_cancel_events: dict[str, threading.Event] = {}
_cancel_lock = threading.Lock()


def _free_gpu_cache():
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        gc.collect()


def _get_vram_gb() -> float:
    if DEVICE != "cuda":
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)


def _auto_batch_size() -> int:
    vram = _get_vram_gb()
    if vram >= 20:
        return 24
    if vram >= 14:
        return 16
    if vram >= 10:
        return 8
    return 4


def _get_align_model(language_code: str):
    normalized = (language_code or WHISPERX_DEFAULT_LANGUAGE or "pt").strip().lower()
    if normalized not in _align_models:
        LOGGER.info("Loading WhisperX align model | language=%s", normalized)
        _align_models[normalized] = whisperx.load_align_model(
            language_code=normalized,
            device=DEVICE,
        )
    return _align_models[normalized]


def _preprocess_audio(input_path: str) -> str:
    """Converte áudio para 16kHz mono WAV (formato nativo do Whisper)."""
    out_path = input_path + ".16k.wav"
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", input_path,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            out_path,
        ],
        capture_output=True,
        check=True,
    )
    return out_path


def _get_audio_duration_from_file(path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except (ValueError, AttributeError):
        return 0.0


def _split_audio(path: str, chunk_seconds: int) -> list[str]:
    """Divide áudio longo em chunks para evitar O(n²) na diarização."""
    duration = _get_audio_duration_from_file(path)
    if duration <= chunk_seconds:
        return [path]

    LOGGER.info(
        "Splitting audio into chunks | duration_min=%.1f | chunk_min=%.1f",
        duration / 60, chunk_seconds / 60,
    )
    chunks = []
    start = 0.0
    idx = 0
    while start < duration:
        out = f"{path}.chunk{idx}.wav"
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", path,
                "-ss", str(start),
                "-t", str(chunk_seconds),
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
                out,
            ],
            capture_output=True,
            check=True,
        )
        chunks.append(out)
        start += chunk_seconds
        idx += 1
    LOGGER.info("Audio split into %d chunks", len(chunks))
    return chunks


class TranscriptionCanceled(Exception):
    pass


def _check_canceled(cancel_event: Optional[threading.Event], label: str = ""):
    if cancel_event is not None and cancel_event.is_set():
        LOGGER.info("Transcription canceled at stage: %s", label)
        raise TranscriptionCanceled()


def _process_chunk(
    audio_path: str,
    language: Optional[str],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
    time_offset: float,
    filename: str,
    started_at: float,
    cancel_event: Optional[threading.Event] = None,
) -> tuple[list[dict], str]:
    """Processa um chunk: transcrição → alinhamento → diarização."""
    _check_canceled(cancel_event, "chunk_start")
    audio_data = whisperx.load_audio(audio_path)

    transcribe_kwargs = {}
    effective_language = language or WHISPERX_DEFAULT_LANGUAGE
    if effective_language:
        transcribe_kwargs["language"] = effective_language

    _check_canceled(cancel_event, "before_transcribe")
    with torch.inference_mode():
        result = _model.transcribe(
            audio_data,
            batch_size=WHISPERX_BATCH_SIZE,
            **transcribe_kwargs,
        )
    transcription_language = result.get("language") or effective_language
    LOGGER.info(
        "WhisperX chunk transcription done | file=%s | offset=%.0fs | segments=%s | elapsed_s=%.1f",
        filename, time_offset, len(result.get("segments", [])),
        time.monotonic() - started_at,
    )

    _check_canceled(cancel_event, "before_align")
    align_model, metadata = _get_align_model(transcription_language)
    with torch.inference_mode():
        result = whisperx.align(
            result["segments"], align_model, metadata, audio_data, DEVICE,
            return_char_alignments=False,
        )

    _free_gpu_cache()

    _check_canceled(cancel_event, "before_diarize")
    diarize_kwargs = {}
    if min_speakers is not None:
        diarize_kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        diarize_kwargs["max_speakers"] = max_speakers

    with torch.inference_mode():
        diarize_segments = _diarize_model(audio_data, **diarize_kwargs)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    LOGGER.info(
        "WhisperX chunk diarization done | file=%s | offset=%.0fs | elapsed_s=%.1f",
        filename, time_offset, time.monotonic() - started_at,
    )

    _free_gpu_cache()

    segments = result.get("segments", [])
    if time_offset > 0:
        for seg in segments:
            if "start" in seg:
                seg["start"] += time_offset
            if "end" in seg:
                seg["end"] += time_offset

    return segments, transcription_language


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _diarize_model, WHISPERX_BATCH_SIZE

    if DEVICE == "cuda" and WHISPERX_ENABLE_TF32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        LOGGER.info("TF32 enabled for WhisperX CUDA workloads.")

    if WHISPERX_BATCH_SIZE <= 0:
        WHISPERX_BATCH_SIZE = _auto_batch_size()
        LOGGER.info(
            "Auto batch size: %d (VRAM=%.1f GB)", WHISPERX_BATCH_SIZE, _get_vram_gb()
        )

    _model = whisperx.load_model(WHISPER_MODEL, DEVICE, compute_type=COMPUTE_TYPE)
    _diarize_model = DiarizationPipeline(
        model_name=WHISPERX_DIARIZATION_MODEL,
        use_auth_token=HF_TOKEN,
        device=DEVICE,
    )
    if WHISPERX_DEFAULT_LANGUAGE:
        _get_align_model(WHISPERX_DEFAULT_LANGUAGE)

    LOGGER.info(
        "WhisperX ready | model=%s | compute=%s | batch=%d | device=%s | vram=%.1f GB",
        WHISPER_MODEL, COMPUTE_TYPE, WHISPERX_BATCH_SIZE, DEVICE, _get_vram_gb(),
    )
    yield

    _align_models.clear()
    _free_gpu_cache()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    if _model is None or _diarize_model is None:
        raise HTTPException(status_code=503, detail="Model loading")
    info = {
        "status": "ok",
        "model": WHISPER_MODEL,
        "diarization_model": WHISPERX_DIARIZATION_MODEL,
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "batch_size": WHISPERX_BATCH_SIZE,
    }
    if DEVICE == "cuda":
        info["vram_gb"] = round(_get_vram_gb(), 1)
    return info


@app.post("/v1/cancel")
async def cancel_transcription(request_id: str = Form(...)):
    """Sinaliza cancelamento de uma transcrição em andamento pelo request_id."""
    with _cancel_lock:
        event = _cancel_events.get(request_id)
    if event is None:
        return JSONResponse({"canceled": False, "reason": "request_id not found or already finished"})

    event.set()
    LOGGER.info("Cancel requested | request_id=%s", request_id)
    return JSONResponse({"canceled": True, "request_id": request_id})


@app.post("/v1/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    language: Optional[str] = Form(None),
    diarization_enabled: bool = Form(True),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
    request_id: Optional[str] = Form(None),
):
    if _model is None or _diarize_model is None:
        raise HTTPException(status_code=503, detail="Model loading")

    cancel_event: Optional[threading.Event] = None
    if request_id:
        cancel_event = threading.Event()
        with _cancel_lock:
            _cancel_events[request_id] = cancel_event

    ext = os.path.splitext(audio.filename or "")[1] or ".wav"
    started_at = time.monotonic()
    filename = audio.filename or "audio"
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
        payload = await audio.read()
        f.write(payload)
        tmp_path = f.name

    temp_files = [tmp_path]
    try:
        LOGGER.info(
            "WhisperX request started | file=%s | size_mb=%.2f | language=%s | diarization_enabled=%s | request_id=%s",
            filename,
            len(payload) / (1024 * 1024),
            language or WHISPERX_DEFAULT_LANGUAGE or "auto",
            diarization_enabled,
            request_id or "(none)",
        )

        wav_path = _preprocess_audio(tmp_path)
        temp_files.append(wav_path)

        _check_canceled(cancel_event, "after_preprocess")

        audio_data = whisperx.load_audio(wav_path)
        audio_duration = len(audio_data) / 16000 if hasattr(audio_data, "__len__") else 0.0
        LOGGER.info(
            "WhisperX audio loaded | file=%s | duration_min=%.2f | elapsed_s=%.1f",
            filename, audio_duration / 60, time.monotonic() - started_at,
        )

        _check_canceled(cancel_event, "after_load")

        if not diarization_enabled:
            transcribe_kwargs = {}
            effective_language = language or WHISPERX_DEFAULT_LANGUAGE
            if effective_language:
                transcribe_kwargs["language"] = effective_language
            with torch.inference_mode():
                result = _model.transcribe(
                    audio_data, batch_size=WHISPERX_BATCH_SIZE, **transcribe_kwargs,
                )
            _check_canceled(cancel_event, "after_transcribe_no_diarize")
            transcription_language = result.get("language") or effective_language
            for segment in result.get("segments", []):
                segment["speaker"] = "SPEAKER_00"
            LOGGER.info(
                "WhisperX transcription done (no diarization) | file=%s | segments=%s | elapsed_s=%.1f",
                filename, len(result.get("segments", [])), time.monotonic() - started_at,
            )
            return JSONResponse({
                "segments": result["segments"],
                "language": transcription_language,
                "diarization_skipped": True,
            })

        if (
            WHISPERX_DIARIZATION_MAX_AUDIO_SECONDS > 0
            and audio_duration > WHISPERX_DIARIZATION_MAX_AUDIO_SECONDS
        ):
            transcribe_kwargs = {}
            effective_language = language or WHISPERX_DEFAULT_LANGUAGE
            if effective_language:
                transcribe_kwargs["language"] = effective_language
            with torch.inference_mode():
                result = _model.transcribe(
                    audio_data, batch_size=WHISPERX_BATCH_SIZE, **transcribe_kwargs,
                )
            _check_canceled(cancel_event, "after_transcribe_max_audio")
            transcription_language = result.get("language") or effective_language
            for segment in result.get("segments", []):
                segment["speaker"] = "SPEAKER_00"
            reason = (
                f"Audio com {audio_duration:.1f}s excede o limite de "
                f"{WHISPERX_DIARIZATION_MAX_AUDIO_SECONDS:.1f}s para separação de falantes."
            )
            LOGGER.warning(
                "WhisperX diarization skipped | file=%s | reason=%s | elapsed_s=%.1f",
                filename, reason, time.monotonic() - started_at,
            )
            return JSONResponse({
                "segments": result["segments"],
                "language": transcription_language,
                "diarization_skipped": True,
                "diarization_skip_reason": reason,
            })

        chunks = _split_audio(wav_path, WHISPERX_CHUNK_DURATION_S)
        temp_files.extend(c for c in chunks if c != wav_path)

        all_segments = []
        detected_language = None
        offset = 0.0

        for i, chunk_path in enumerate(chunks):
            _check_canceled(cancel_event, f"before_chunk_{i}")
            chunk_duration = _get_audio_duration_from_file(chunk_path)
            LOGGER.info(
                "Processing chunk %d/%d | file=%s | offset=%.0fs | chunk_duration=%.0fs",
                i + 1, len(chunks), filename, offset, chunk_duration,
            )

            segments, lang = _process_chunk(
                chunk_path, language, min_speakers, max_speakers,
                offset, filename, started_at, cancel_event,
            )
            if not detected_language:
                detected_language = lang

            all_segments.extend(segments)
            offset += chunk_duration

        final_lang = detected_language or language or WHISPERX_DEFAULT_LANGUAGE or "unknown"

        LOGGER.info(
            "WhisperX request finished | file=%s | total_segments=%d | total_elapsed_s=%.1f",
            filename, len(all_segments), time.monotonic() - started_at,
        )

        return JSONResponse({
            "segments": all_segments,
            "language": final_lang,
            "diarization_skipped": False,
        })

    except TranscriptionCanceled:
        LOGGER.info(
            "WhisperX request canceled | file=%s | request_id=%s | elapsed_s=%.1f",
            filename, request_id or "(none)", time.monotonic() - started_at,
        )
        _free_gpu_cache()
        return JSONResponse(
            {"segments": [], "language": "", "canceled": True},
            status_code=499,
        )

    except Exception as exc:
        LOGGER.exception("WhisperX transcribe failed for %s", audio.filename or tmp_path)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    finally:
        if request_id:
            with _cancel_lock:
                _cancel_events.pop(request_id, None)
        for f_path in temp_files:
            try:
                os.unlink(f_path)
            except OSError:
                pass
