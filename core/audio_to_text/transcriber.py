from pathlib import Path
from typing import List

from log_utils import setup_logger
from yaml_reader import ConfigLoader
import whisper
from pathlib import Path
from typing import List, Dict
import torchaudio
import numpy as np
import whisper
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

logger = setup_logger(__name__)
config = ConfigLoader("../configs/config.yaml")

_whisper_model = None


def load_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        model_name = config.get('transcribe.model_name', 'medium')
        logger.info(f"Loading Whisper model: {model_name}")
        _whisper_model = whisper.load_model(model_name, device="cpu")
    return _whisper_model


def merge_speech_segments(segments: List[Dict[str, float]], max_silence: float = 1.0) -> List[Dict[str, float]]:
    if not segments:
        return []

    merged = []
    current = segments[0].copy()

    for seg in segments[1:]:
        if seg['end'] - current['start'] <= max_silence:
            current['end'] = seg['end']
        else:
            merged.append(current)
            current = seg.copy()

    merged.append(current)
    return merged


def extend_speech_segments(segments: List[Dict[str, float]]) -> List[Dict[str, float]]:
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


def transcribe_chunk(args) -> Dict:
    audio_path, start, end = args
    min_duration_sec = 0.5

    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        start_frame = int(start * sample_rate)
        end_frame = int(end * sample_rate)

        if end_frame <= start_frame:
            raise ValueError("Segment duration is zero or negative.")

        chunk = waveform[:, start_frame:end_frame]

        # Проверка: достаточно ли данных
        duration = (end_frame - start_frame) / sample_rate
        if duration < min_duration_sec:
            raise ValueError(f"Chunk too short ({duration:.2f} sec), skipping.")

        if sample_rate != 16000:
            chunk = torchaudio.functional.resample(chunk, sample_rate, 16000)

        audio_np = chunk.squeeze().numpy()

        if audio_np.size == 0:
            raise ValueError("Audio chunk is empty after processing.")

        model = load_whisper_model()
        result = model.transcribe(
            audio_np,
            language="ru",
            beam_size=4,
            temperature=0.4,
            compression_ratio_threshold=2.2
        )

        return {"result": result["segments"], "start": start, "end": end}

    except Exception as e:
        logger.error(f"Transcription error [{start}-{end}]: {e}")
        return {"start": start, "end": end, "text": ""}


def transcribe(audio_path: Path, valid_segments: List[Dict[str, float]]) -> dict:
    logger.debug(f"{audio_path.name} - TRANSCRIBING...")
    # speech_timestamps = merge_speech_segments(speech_timestamps)
    speech_timestamps = extend_speech_segments(valid_segments)
    args_list = [(audio_path, ts["start"], ts["end"]) for ts in speech_timestamps]

    with mp.Pool(processes=min(config.get("transcribe.threads"), mp.cpu_count())) as pool:
        results = pool.map(transcribe_chunk, args_list)

    all_segments = []
    full_text = ""

    # for res in results:
    #     for seg in res["result"]:
    #         seg["start"] += res["start"]
    #         seg["end"] += res["start"]
    #         all_segments.append(seg)
    #         full_text += seg["text"].strip() + " "
    for res in results:
        segment_text = ""
        if len(res["result"]) == 0:
            continue
        first = res["result"][0]
        for seg in res["result"]:
            seg["start"] += res["start"]
            seg["end"] += res["start"]

            full_text += seg["text"].strip() + " "
            segment_text += seg["text"].strip() + " "
        first["text"] = segment_text
        first["start"] = res["start"]
        first["end"] = res["end"]
        all_segments.append(first)

    return {
        "text": full_text.strip(),
        "segments": sorted(all_segments, key=lambda x: x["start"])
    }
