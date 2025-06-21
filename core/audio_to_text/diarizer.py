from pathlib import Path
import numpy as np
from sklearn.metrics import silhouette_score
import torch
from sklearn.cluster import KMeans
from sklearn.mixture import BayesianGaussianMixture
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
MAX_SPEAKERS = config.get('diarize.max_speakers', 5)


def pad_segment(wav: torch.Tensor, target_samples: int) -> torch.Tensor:
    """Center-pad audio segment to target length"""
    current_length = wav.shape[-1]
    if current_length >= target_samples:
        return wav

    pad_total = target_samples - current_length
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return torch.nn.functional.pad(wav, (pad_left, pad_right))


def estimate_optimal_clusters(embeddings: np.ndarray) -> int:
    """Determine optimal number of speakers using silhouette score"""
    if len(embeddings) <= 2:
        return min(len(embeddings), 1)

    best_n = 1
    best_score = -1

    max_possible = min(MAX_SPEAKERS, len(embeddings) - 1)

    for n in range(2, max_possible + 1):
        try:
            kmeans = KMeans(n_clusters=n, random_state=42).fit(embeddings)
            score = silhouette_score(embeddings, kmeans.labels_)

            if score > best_score:
                best_score = score
                best_n = n
        except Exception as e:
            logger.warning(f"Clustering failed for {n} speakers: {e}")
            continue

    logger.debug(f"Optimal clusters: {best_n} (score: {best_score:.2f})")
    return best_n


def diarize(file_path: Path) -> DiarizedResult:
    """Main diarization pipeline"""
    logger.info(f"Diarizing {file_path.name}")

    # Load models
    try:
        embedding_model = EncoderClassifier.from_hparams(
            source=config.get("diarize.model_name"),
            savedir="../../pretrained_models"
        )
        vad_model = load_silero_vad()
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        return DiarizedResult([], np.array([]))

    # Read and analyze audio
    try:
        audio = read_audio(file_path)
        speech_timestamps = get_speech_timestamps(
            audio,
            vad_model,
            return_seconds=True,
            threshold=config.get('diarize.vad_threshold', 0.5)
        )
    except Exception as e:
        logger.error(f"Audio processing failed: {e}")
        return DiarizedResult([], np.array([]))

    valid_segments = []
    embeddings = []
    target_samples = int(TARGET_SEGMENT_LENGTH * SAMPLE_RATE)

    for segment in speech_timestamps:
        start_time, end_time = segment['start'], segment['end']
        duration = end_time - start_time

        if duration < MIN_SEGMENT_LENGTH:
            continue

        start = int(start_time * SAMPLE_RATE)
        end = int(end_time * SAMPLE_RATE)
        segment_wav = audio[start:end]

        # Pad/trim to consistent length
        segment_wav = pad_segment(segment_wav, target_samples)
        segment_wav = segment_wav.unsqueeze(0)

        try:
            with torch.no_grad():
                embedding = embedding_model.encode_batch(segment_wav)
                embedding = embedding.squeeze().cpu().numpy()

                # Validate embedding
                if np.isnan(embedding).any() or np.linalg.norm(embedding) < 0.1:
                    raise ValueError("Invalid embedding")

                embeddings.append(embedding)
                valid_segments.append(segment)
        except Exception as e:
            logger.warning(f"Segment {start_time:.2f}-{end_time:.2f}s failed: {e}")

    if not embeddings:
        logger.warning("No valid speech segments found")
        return DiarizedResult([], np.array([]))

    # Normalize embeddings
    embeddings_np = np.vstack(embeddings)
    embeddings_np = embeddings_np / np.linalg.norm(embeddings_np, axis=1, keepdims=True)

    # Cluster selection

    n_clusters = estimate_optimal_clusters(embeddings_np)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings_np)

    logger.info(
        f"Diarization complete. Segments: {len(valid_segments)}, "
        f"Speakers: {len(np.unique(labels))}"
    )

    return DiarizedResult(valid_segments, labels)
