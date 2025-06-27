from pathlib import Path
import numpy as np
import torch
from pyannote.audio import Pipeline
from core.dto.diarisation_result import DiarizedResult
from log_utils import setup_logger
from yaml_reader import ConfigLoader

# Setup
logger = setup_logger(__name__)
config = ConfigLoader("../configs/config.yaml")

# Максимальное число говорящих из конфига
MAX_SPEAKERS = config.get('diarize.max_speakers', 5)


def diarize(file_path: Path) -> DiarizedResult:
    logger.info(f"Diarizing {file_path.name} with pyannote")

    try:
        # Загрузка пайплайна pyannote (предварительно необходимо указать токен huggingface в окружении)
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token="")
    except Exception as e:
        logger.error(f"Failed to load pyannote pipeline: {e}")
        return DiarizedResult([], np.array([]))

    try:
        # Диаризация: возвращается annotation с сегментами и метками говорящих
        diarization = pipeline(str(file_path))
    except Exception as e:
        logger.error(f"Failed to diarize audio: {e}")
        return DiarizedResult([], np.array([]))

    segments = []
    labels = []

    # Получаем все сегменты из аннотации
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start_time = turn.start
        end_time = turn.end
        segments.append({'start': start_time, 'end': end_time})

        labels.append(speaker)

    if not segments:
        logger.warning("No speech segments found by pyannote")
        return DiarizedResult([], np.array([]))

    # Если нужно ограничить число спикеров
    unique_speakers = list(sorted(set(labels)))
    if len(unique_speakers) > MAX_SPEAKERS:
        unique_speakers = unique_speakers[:MAX_SPEAKERS]
        labels = [spk if spk in unique_speakers else "unknown" for spk in labels]

    # Кодируем метки в числа
    label_to_id = {label: idx for idx, label in enumerate(unique_speakers)}
    labels_numeric = np.array([label_to_id.get(label, -1) for label in labels])

    logger.info(
        f"Diarization complete. Segments: {len(segments)}, "
        f"Speakers: {len(unique_speakers)}"
    )

    return DiarizedResult(segments, labels_numeric)
