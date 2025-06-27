import os
from typing import List, Tuple
import numpy as np
import librosa
from pydub import AudioSegment
from pydub.silence import split_on_silence
import noisereduce as nr

from log_utils import setup_logger

logger = setup_logger(__name__)

class AudioLoader:
    def __init__(self, sample_rate: int = 16000, mono: bool = True):
        self.sample_rate = sample_rate
        self.mono = mono
        self.supported_formats = ['.wav', '.mp3', '.ogg', '.flac', '.aac', '.m4a']

    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file and return normalized numpy array and sample rate."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.supported_formats:
            raise ValueError(f"Unsupported audio format {ext}. Supported: {self.supported_formats}")

        try:
            audio_data, sr = librosa.load(file_path, sr=self.sample_rate, mono=self.mono)
            logger.info(f"Loaded audio {file_path} with shape {audio_data.shape} and sample rate {sr}")
            return audio_data, sr
        except Exception as e:
            logger.error(f"Failed to load audio {file_path}: {e}")
            raise

    def remove_silence(self, audio_data: np.ndarray, sample_rate: int,
                       silence_thresh: float = -40, min_silence_len: int = 500,
                       keep_silence: int = 100) -> np.ndarray:
        """Remove silent parts from audio using pydub."""
        try:
            audio_seg = self._numpy_to_audiosegment(audio_data, sample_rate)
            chunks = split_on_silence(
                audio_seg,
                min_silence_len=min_silence_len,
                silence_thresh=silence_thresh,
                keep_silence=keep_silence
            )
            if not chunks:
                logger.warning("No non-silent chunks found; returning original audio")
                return audio_data

            combined = sum(chunks)
            logger.info(f"Removed silence: {len(chunks)} chunks combined")
            return self._audiosegment_to_numpy(combined)
        except Exception as e:
            logger.error(f"Error during silence removal: {e}")
            return audio_data  # fallback to original audio

    def normalize_audio(self, audio_data: np.ndarray, sample_rate: int, target_dBFS: float = -20.0) -> np.ndarray:
        """Normalize audio to target dBFS."""
        try:
            audio_seg = self._numpy_to_audiosegment(audio_data, sample_rate)
            change_in_dBFS = target_dBFS - audio_seg.dBFS
            logger.debug(f"Applying gain of {change_in_dBFS:.2f} dB for normalization")
            normalized_seg = audio_seg.apply_gain(change_in_dBFS)
            return self._audiosegment_to_numpy(normalized_seg)
        except Exception as e:
            logger.error(f"Normalization failed: {e}")
            return audio_data

    def reduce_noise(self, audio_data: np.ndarray, sample_rate: int,
                     prop_decrease: float = 0.7) -> np.ndarray:
        try:
            # Estimate noise from the first 0.5 seconds (adjust if needed)
            noise_clip = audio_data[:int(sample_rate * 1)]
            reduced_noise_audio = nr.reduce_noise(y=audio_data, y_noise=noise_clip,
                                                  sr=sample_rate, prop_decrease=prop_decrease)
            logger.info("Noise reduction applied successfully")
            return reduced_noise_audio
        except Exception as e:
            logger.error(f"Noise reduction failed: {e}")
            return audio_data  # fallback to original audio

    def get_audio_duration(self, audio_data: np.ndarray, sample_rate: int) -> float:
        """Calculate duration of audio in seconds."""
        if audio_data.ndim == 1:
            length = len(audio_data)
        else:
            # For multi-channel, take length of first dimension (samples)
            length = audio_data.shape[0]
        duration_sec = length / sample_rate
        logger.debug(f"Calculated audio duration: {duration_sec:.2f} seconds")
        return duration_sec

    def _numpy_to_audiosegment(self, audio_data: np.ndarray, sample_rate: int) -> AudioSegment:
        """Convert numpy float32 array in [-1, 1] to pydub AudioSegment."""
        # Clip to [-1, 1] to avoid overflow
        audio_data = np.clip(audio_data, -1, 1)
        # Convert to 16-bit PCM
        int_data = (audio_data * 32767).astype(np.int16)

        channels = 1 if self.mono else 2
        return AudioSegment(
            int_data.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,  # 16-bit
            channels=channels
        )

    def _audiosegment_to_numpy(self, audio_segment: AudioSegment) -> np.ndarray:
        """Convert pydub AudioSegment to numpy float32 array in [-1, 1]."""
        samples = np.array(audio_segment.get_array_of_samples())
        if audio_segment.sample_width == 2:
            samples = samples.astype(np.float32) / 32768.0
        else:
            raise NotImplementedError("Only 16-bit sample width supported")

        if audio_segment.channels > 1:
            samples = samples.reshape((-1, audio_segment.channels))
            if self.mono:
                # Convert to mono by averaging channels
                samples = samples.mean(axis=1)

        return samples

    def process(self, file_path: str,
                silence_thresh: float = -40,
                min_silence_len: int = 500,
                keep_silence: int = 100,
                target_dBFS: float = -20.0) -> Tuple[np.ndarray, int]:
        """Full pipeline: load -> remove silence -> normalize"""
        audio, sr = self.load_audio(file_path)
        audio = self.remove_silence(audio, sr, silence_thresh, min_silence_len, keep_silence)
        audio = self.normalize_audio(audio, sr, target_dBFS)
        return audio, sr
