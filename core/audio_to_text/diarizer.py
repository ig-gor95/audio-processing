from pathlib import Path

import numpy as np
from sklearn.metrics import silhouette_score
import torch
from sklearn.cluster import KMeans
from speechbrain.inference import EncoderClassifier

from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

from core.entity.diarisation_result import DiarizedResult
from log_utils import setup_logger
from yaml_reader import ConfigLoader

# Setup
logger = setup_logger(__name__)
config = ConfigLoader("../configs/config.yaml")

# Constants
SAMPLE_RATE = config.get('audio.sample_rate')
MIN_SEGMENT_LENGTH = config.get('diarize.min_segment_length', 0.75)
TARGET_SEGMENT_LENGTH = config.get('diarize.target_segment_length', 1.5)


def estimate_optimal_clusters(embeddings, max_speakers=5):
    best_score = -1
    best_n = 2  # minimum 2 speakers

    for n in range(2, max_speakers + 1):
        kmeans = KMeans(n_clusters=n, random_state=42).fit(embeddings)
        score = silhouette_score(embeddings, kmeans.labels_)

        if score > best_score:
            best_score = score
            best_n = n

    return best_n

def diarize(file_path: Path) -> DiarizedResult:
    logger.debug(f"{file_path.name} - DIARIZING..")

    # Load models
    embedding_model = EncoderClassifier.from_hparams(
        source=config.get("diarize.model_name"),
        savedir="../../pretrained_models"
    )
    vad_model = load_silero_vad()

    # Read and analyze audio
    audio = read_audio(file_path)
    speech_timestamps = get_speech_timestamps(audio, vad_model, return_seconds=True)

    valid_segments = []
    embeddings = []

    for segment in speech_timestamps:
        start_time, end_time = segment['start'], segment['end']
        duration = end_time - start_time

        if duration < MIN_SEGMENT_LENGTH:
            continue

        start = int(start_time * SAMPLE_RATE)
        end = int(end_time * SAMPLE_RATE)
        segment_wav = audio[start:end]

        # Pad if needed
        if duration < TARGET_SEGMENT_LENGTH:
            padding = int((TARGET_SEGMENT_LENGTH - duration) * SAMPLE_RATE)
            segment_wav = torch.nn.functional.pad(segment_wav, (0, padding))

        segment_wav = segment_wav.unsqueeze(0)

        try:
            embedding = embedding_model.encode_batch(segment_wav)
            embeddings.append(embedding.squeeze().cpu().numpy())
            valid_segments.append(segment)
        except Exception as e:
            logger.error(f"Error processing segment ({start_time}-{end_time}s): {e}")

    if not embeddings:
        logger.warning("No valid speech segments found.")
        return DiarizedResult([], np.array([]))

    embeddings_np = np.array(embeddings)
    optimal_n = estimate_optimal_clusters(embeddings_np)

    kmeans = KMeans(n_clusters=optimal_n, random_state=42)
    labels = kmeans.fit_predict(np.array(embeddings))

    return DiarizedResult(valid_segments, labels)
