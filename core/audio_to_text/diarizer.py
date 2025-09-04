from pathlib import Path
import numpy as np
from pyannote.audio import Pipeline
from pydub import AudioSegment
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.metrics import silhouette_score
import torch
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import BayesianGaussianMixture
from speechbrain.inference import EncoderClassifier
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
from core.dto.diarisation_result import DiarizedResult
from log_utils import setup_logger
from yaml_reader import ConfigLoader

# Setup
logger = setup_logger(__name__)
config = ConfigLoader("../configs/config.yaml")

# Constants
SAMPLE_RATE = int(config.get('audio.sample_rate', 16000))
MIN_SEGMENT_LENGTH = float(config.get('diarize.min_segment_length', 0.75))       # сек — до паддинга не режем
TARGET_SEGMENT_LENGTH = float(config.get('diarize.target_segment_length', 1.5)) # сек
MAX_SPEAKERS = int(config.get('diarize.max_speakers', 2))
FORCE_TWO = bool(config.get('diarize.force_two', True))  # <— по умолчанию строго 2
VAD_THR = float(config.get('diarize.vad_threshold', 0.30))
BATCH = int(config.get('diarize.batch_size', 64))
MIN_SPK_DUR = float(config.get('diarize.min_speaker_run', 0.8))  # сглаживание коротких вставок

# === Утилиты ===

def read_audio_with_sr(file_path: Path):
    """Read audio and return mono FloatTensor [T] at SAMPLE_RATE."""
    try:
        import torchaudio
        wav, sr = torchaudio.load(str(file_path))
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        wav = wav.mean(0)  # mono [T]
        return wav, SAMPLE_RATE
    except Exception as e:
        logger.error(f"Audio loading failed: {e}")
        raise

def merge_close(ts, gap=0.50):
    """Склейка близких VAD-сегментов (сек)."""
    out = []
    for seg in ts:
        if not out or seg['start'] - out[-1]['end'] > gap:
            out.append({'start': seg['start'], 'end': seg['end']})
        else:
            out[-1]['end'] = max(out[-1]['end'], seg['end'])
    return out

def center_crop_or_repeat(x: np.ndarray, target_len: int) -> np.ndarray:
    """Ровно target_len: центр-кроп или repeat-pad (без нулей)."""
    L = len(x)
    if L == target_len: return x
    if L > target_len:
        o = (L - target_len) // 2
        return x[o:o+target_len]
    reps = int(np.ceil(target_len / max(1, L)))
    return np.tile(x, reps)[:target_len]

def cmvn(x: np.ndarray) -> np.ndarray:
    """Channel Mean-Variance Norm (устойчиво к нулевой дисперсии)."""
    m, s = float(x.mean()), float(x.std())
    if not np.isfinite(s) or s < 1e-6: s = 1.0
    return (x - m) / s

def l2norm_rows(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / n

def extract_embeddings(wav_np: np.ndarray, segments, model: EncoderClassifier, target_len_samples: int) -> np.ndarray:
    """Батчевое извлечение эмбеддингов. Вход в модель: [B, T] float32."""
    buf = []
    keep_segments = []
    for seg in segments:
        s = int(seg['start'] * SAMPLE_RATE)
        e = int(seg['end'] * SAMPLE_RATE)
        if e <= s: continue
        x = wav_np[s:e].astype(np.float32)
        if len(x) < 8:  # защита от пустых
            continue
        # НЕ отбрасываем по MIN_SEGMENT_LENGTH до паддинга
        x = center_crop_or_repeat(x, target_len_samples)
        x = cmvn(x)
        buf.append(x)
        keep_segments.append(seg)

    if not buf:
        return np.empty((0,)), []

    X_list = []
    with torch.no_grad():
        for i in range(0, len(buf), BATCH):
            chunk = np.stack(buf[i:i+BATCH], axis=0)      # [b, T]
            x_t = torch.from_numpy(chunk)                 # [b, T] float32
            emb_t = model.encode_batch(x_t)               # [b, d] или [b,1,d]
            emb = emb_t.detach().cpu().numpy()
            if emb.ndim == 3 and emb.shape[1] == 1:
                emb = emb.squeeze(1)                      # [b, d]
            if emb.ndim == 1:
                emb = emb[None, :]                        # [1, d]
            X_list.append(emb)

    X = np.vstack(X_list).astype(np.float32)
    X = l2norm_rows(X)
    return X, keep_segments

def smooth_short_runs(labels: np.ndarray, segments, min_run_sec: float) -> np.ndarray:
    """Склеиваем вставки короче min_run_sec к соседям того же спикера."""
    labs = labels.copy()
    runs = []
    cur = labs[0]; s = segments[0]['start']; e = segments[0]['end']; i0 = 0
    for i in range(1, len(labs)):
        if labs[i] == cur:
            e = segments[i]['end']
        else:
            runs.append((cur, s, e, i0, i-1))
            cur = labs[i]; s = segments[i]['start']; e = segments[i]['end']; i0 = i
    runs.append((cur, s, e, i0, len(labs)-1))

    for j in range(1, len(runs)-1):
        lab, rs, re, a, b = runs[j]
        if (re - rs) < min_run_sec and runs[j-1][0] == runs[j+1][0]:
            # переливаем маленькую вставку к соседям
            labs[a:b+1] = runs[j-1][0]
    return labs

def rebalance_if_collapsed(labels: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Если один класс <15% → принудительно балансируем по проекции на (c1-c0)."""
    y = labels.copy()
    p0 = (y == 0).mean()
    if p0 < 0.15 or p0 > 0.85:
        # центры
        if (y == 0).any() and (y == 1).any():
            c0 = X[y == 0].mean(0)
            c1 = X[y == 1].mean(0)
        else:
            # инициализация по самым непохожим
            i0 = 0
            sims = X @ X[i0] / ((np.linalg.norm(X, axis=1) * np.linalg.norm(X[i0])) + 1e-12)
            i1 = int(np.argmin(sims))
            c0, c1 = X[i0], X[i1]
        v = c1 - c0
        v /= (np.linalg.norm(v) + 1e-12)
        score = X @ v
        order = np.argsort(score)
        m = len(X) // 2
        y = np.zeros_like(y)
        y[order[m:]] = 1
    return y

# === Основной пайплайн ===

def diarize(file_path: Path) -> DiarizedResult:
    logger.info(f"Diarizing {file_path.name}")

    # 1) Модели
    try:
        embedding_model = EncoderClassifier.from_hparams(
            source=config.get("diarize.model_name", "speechbrain/spkrec-ecapa-voxceleb"),
            savedir="../../pretrained_models"
        )
        vad_model = load_silero_vad()
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        return DiarizedResult([], np.array([]))

    # 2) Аудио + VAD
    try:
        audio, sr = read_audio_with_sr(file_path)   # torch.Tensor [T], SAMPLE_RATE
        wav = audio.unsqueeze(0)                    # [1, T] для silero_vad
        speech_ts = get_speech_timestamps(wav, vad_model, return_seconds=True, threshold=VAD_THR)
        if not speech_ts:
            logger.warning("VAD returned 0 speech segments")
            return DiarizedResult([], np.array([]))
        speech_ts = merge_close(speech_ts, gap=0.5)
    except Exception as e:
        logger.error(f"Audio processing failed: {e}")
        return DiarizedResult([], np.array([]))

    # 3) Сегменты -> эмбеддинги
    target_samples = int(TARGET_SEGMENT_LENGTH * SAMPLE_RATE)
    wav_np = audio.cpu().numpy().astype(np.float32)
    X, valid_segments = extract_embeddings(wav_np, speech_ts, embedding_model, target_samples)
    if X.size == 0 or len(valid_segments) == 0:
        logger.warning("No valid speech segments after embedding")
        return DiarizedResult([], np.array([]))

    # 4) Кластеризация (cosine, по умолчанию строго 2)
    try:
        if FORCE_TWO or MAX_SPEAKERS <= 2:
            n_clusters = 2
        else:
            n_clusters = min(MAX_SPEAKERS, max(2, len(X) // 3))  # мягкая верхняя оценка
        cl = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average")
        labels = cl.fit_predict(X)
        if n_clusters == 2:
            labels = rebalance_if_collapsed(labels, X)
    except Exception as e:
        logger.error(f"Clustering failed: {e}")
        return DiarizedResult([], np.array([]))

    # 5) Сглаживание коротких вставок
    if len(labels) > 1:
        labels = smooth_short_runs(labels, valid_segments, MIN_SPK_DUR)

    logger.info(
        f"Diarization complete. Segments: {len(valid_segments)}, "
        f"Speakers(forced): {len(np.unique(labels))}"
    )
    return DiarizedResult(valid_segments, labels)