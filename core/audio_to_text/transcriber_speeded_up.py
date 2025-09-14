from pathlib import Path
from typing import List, Dict
import multiprocessing as mp
import numpy as np
import torchaudio
import mlx_whisper
from pydub import AudioSegment  # noqa: F401
from pydub.silence import split_on_silence  # noqa: F401

from log_utils import setup_logger
from yaml_reader import ConfigLoader

logger = setup_logger(__name__)
config = ConfigLoader("../configs/config.yaml")

# ===== per-worker cache =====
_AUDIO = None            # numpy float32 mono @16k (весь файл)
_SR = 16000
_REPO = None
_LANG = "ru"

_whisper_model = None  # оставлено для совместимости с твоим API (как строка-репо)
_model_name_map = {
    'tiny': 'mlx-community/whisper-tiny',
    'base': 'mlx-community/whisper-base',
    'small': 'mlx-community/whisper-small',
    'medium': 'mlx-community/whisper-medium',
    'large': 'mlx-community/whisper-large-v3',
    'large-v2': 'mlx-community/whisper-large-v2',
    'large-v3': 'mlx-community/whisper-large-v3',
}


def _init_worker(audio_path_str: str, repo: str, lang: str):
    """
    Выполняется один раз при старте процесса-воркера:
      - читаем и ресемплим аудио в 16k mono в RAM
      - сохраняем параметры для вызова transcribe
    """
    global _AUDIO, _SR, _REPO, _LANG
    _REPO = repo or "mlx-community/whisper-large-v3-turbo"
    _LANG = lang or "ru"

    if _AUDIO is None:
        logger.info(f"[{mp.current_process().name}] Loading audio once: {audio_path_str}")
        wav, sr = torchaudio.load(audio_path_str)  # [C, T], float32 в [-1, 1]
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)    # mono
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)
            sr = 16000
        _SR = sr
        _AUDIO = wav.squeeze(0).contiguous().numpy().astype("float32")


def load_whisper_model():
    """
    Совместимость с твоей логикой — возвращаем строку-идентификатор модели (репо).
    Фактическая загрузка кэширется внутри mlx, поэтому дополнительной работы не требуется.
    """
    global _whisper_model
    if _whisper_model is None:
        model_name = config.get('transcribe.model_name', 'medium')
        hf_repo = config.get('transcribe.hf_repo', 'mlx-community/whisper-large-v3-turbo')
        logger.info(f"Loading MLX Whisper model (identifier only): {hf_repo} ({model_name=})")
        _whisper_model = hf_repo  # просто идентификатор, как и было
    return _whisper_model


def extend_speech_segments(segments: List[Dict[str, float]]) -> List[Dict[str, float]]:
    # оставляю как было (если хочешь не менять сегменты — просто верни segments)
    if not segments:
        return []
    result = []
    prev_end = 0.0
    for seg in segments[0:]:
        updated_seg = seg.copy()
        updated_seg['start'] = prev_end
        prev_end = updated_seg['end']
        result.append(updated_seg)
    return result


def _slice_from_global(start: float, end: float) -> np.ndarray:
    """Вырезает [start, end) из буфера _AUDIO (секунды)."""
    global _AUDIO, _SR
    if _AUDIO is None:
        return np.zeros(0, dtype="float32")
    i = max(0, int(start * _SR))
    j = min(len(_AUDIO), int(end * _SR))
    return _AUDIO[i:j].copy() if j > i else np.zeros(0, dtype=_AUDIO.dtype)


def transcribe_chunk(args) -> Dict:
    """
    Работает внутри воркера. Получает (start, end).
    Аудио и «модель» (репо) уже инициализированы в _init_worker.
    """
    try:
        start, end = args
        min_duration_sec = 0.29

        if end <= start:
            return {"result": [], "start": start, "end": end}

        chunk = _slice_from_global(start, end)
        duration = end - start
        if duration < min_duration_sec or chunk.size == 0:
            return {"result": [], "start": start, "end": end}

        model_name = _REPO or load_whisper_model()
        result = mlx_whisper.transcribe(
            chunk,
            path_or_hf_repo=model_name,   # важно: в твоей сборке нет аргумента model=
            language=_LANG,
        )

        # Возвращаем локальные времена, агрегатор их сместит на res["start"]
        segments = [{
            "text": result.get("text", "") or "",
            "start": 0.0,
            "end": duration,
        }]
        return {"result": segments, "start": start, "end": end}

    except Exception as e:
        logger.error(f"Transcription error [{start}-{end}]: {e}")
        return {"result": [], "start": start, "end": end}


def transcribe(audio_path: Path, valid_segments: List[Dict[str, float]]) -> dict:
    logger.debug(f"{audio_path.name} - TRANSCRIBING...")
    speech_timestamps = extend_speech_segments(valid_segments)  # оставлено как было
    args_list = [(ts["start"], ts["end"]) for ts in speech_timestamps]
    if not args_list:
        return {"text": "", "segments": []}

    # параметры пула
    workers_cfg = config.get("transcribe.threads")
    processes = min(int(workers_cfg) if workers_cfg else mp.cpu_count(), mp.cpu_count())
    repo = config.get('transcribe.hf_repo', 'mlx-community/whisper-large-v3-turbo')
    lang = config.get("transcribe.lang", "ru")

    # Инициализатор прогревает аудио/репо один раз на процесс
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=processes,
        initializer=_init_worker,
        initargs=(str(audio_path), repo, lang),
    ) as pool:
        chunksize = max(1, len(args_list) // (processes * 4) or 1)
        results = pool.map(transcribe_chunk, args_list, chunksize=chunksize)

    all_segments = []
    full_text = ""

    for res in results:
        segs = res.get("result", [])
        if not segs:
            continue

        segment_text = ""
        first = segs[0]
        for seg in segs:
            # локальные времена -> абсолютные
            seg["start"] += res["start"]
            seg["end"] += res["start"]

            txt = (seg.get("text") or "").strip()
            if txt:
                full_text += txt + " "
                segment_text += txt + " "

        first["text"] = segment_text
        first["start"] = res["start"]
        first["end"] = res["end"]
        all_segments.append(first)

    return {
        "text": full_text.strip(),
        "segments": sorted(all_segments, key=lambda x: x["start"]),
    }
