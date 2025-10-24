import numpy as np
import librosa
from numba import jit
import multiprocessing as mp

from typing import List, Dict

from core.repository.dialog_rows_repository import DialogRowRepository


@jit(nopython=True)
def calculate_metrics_numba(rms_values: np.ndarray) -> tuple:
    """Numba-accelerated metric calculation"""
    if len(rms_values) == 0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

    mean_val = np.mean(rms_values)
    max_val = np.max(rms_values)
    min_val = np.min(rms_values)
    std_val = np.std(rms_values)
    dynamic_range = max_val - min_val
    p95 = np.percentile(rms_values, 95)
    p75 = np.percentile(rms_values, 75)
    p25 = np.percentile(rms_values, 25)

    return (mean_val, max_val, min_val, std_val, dynamic_range, p95, p75, p25, len(rms_values))


class LoudnessAnalyzer:
    def __init__(self, sample_rate=16000, hop_length=256):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.dialog_row_repository = DialogRowRepository()

    def process_file(self, audio_path: str, phrases: List[Dict]) -> List[Dict]:
        try:
            y, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)

            rms_series = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
            frames_per_second = self.sample_rate / self.hop_length

            results = []
            for phrase in phrases:
                start_frame = int(phrase['start_time'] * frames_per_second)
                end_frame = int(phrase['end_time'] * frames_per_second)
                row_id = phrase['row_id']

                start_frame = max(0, start_frame)
                end_frame = min(len(rms_series), end_frame)

                if start_frame < end_frame:
                    phrase_rms = rms_series[start_frame:end_frame]
                    metrics = calculate_metrics_numba(phrase_rms)
                else:
                    metrics = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

                result = phrase.copy()
                result.update({
                    'mean_loudness': metrics[0],
                    'max_loudness': metrics[1],
                    'min_loudness': metrics[2],
                    'std_loudness': metrics[3],
                    'dynamic_range': metrics[4],
                    'loudness_95th': metrics[5],
                    'loudness_75th': metrics[6],
                    'loudness_25th': metrics[7],
                    'num_frames': metrics[8]
                })
                self.dialog_row_repository.update_loudness(row_id, metrics[0])
                results.append(result)

            return results

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return []