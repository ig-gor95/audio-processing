#!/usr/bin/env python
# coding: utf-8

from sklearn.cluster import KMeans
import numpy as np
import warnings
import logging
import whisper
import torch
import time
import os
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
from log_utils import setup_logger, configure_root_logger
from speechbrain.pretrained import EncoderClassifier
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, date, time
from transformers import pipeline
from dataclasses import dataclass
from tqdm import tqdm_notebook
from tqdm.notebook import tqdm
from typing import Optional
from enum import Enum, auto


TRANSCRIBE_MODEL_NAME = 'medium'

logger = setup_logger(__name__, level='DEBUG')
configure_root_logger(level='INFO')

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.custom_fwd.*")
logging.getLogger("speechbrain").setLevel(logging.WARNING)

from huggingface_hub import login
login()

@dataclass
class DiarizedResult:
	valid_segments: List[Dict[str, float]]
	labels: np.ndarray

def diarize(file_path, min_segment_length=1.5):
	logger.debug(f"{os.path.basename(file_path)} - DIARIZING..")

	embedding_model = EncoderClassifier.from_hparams(
		source="speechbrain/spkrec-ecapa-voxceleb",
		savedir="pretrained_models"
	)
	diarization_model = load_silero_vad()

	audio = read_audio(file_path)
	speech_timestamps = get_speech_timestamps(audio, diarization_model, return_seconds=True)

	valid_segments = []
	embeddings = []

	for segment in speech_timestamps:
		duration = segment['end'] - segment['start']
		if duration < min_segment_length:
			continue

		start = int(segment['start'] * 16000)
		end = int(segment['end'] * 16000)
		segment_wav = audio[start:end]

		if duration < 1.5:
			padding = int((1.5 - duration) * 16000)
			segment_wav = torch.nn.functional.pad(segment_wav, (0, padding))

		segment_wav = segment_wav.unsqueeze(0)

		try:
			embedding = embedding_model.encode_batch(segment_wav)
			embeddings.append(embedding.squeeze().numpy())
			valid_segments.append(segment)
		except Exception as e:
			logger.error(f"Error processing segment: {e}")
			continue
	kmeans = KMeans(n_clusters=2, random_state=42)
	labels = kmeans.fit_predict(np.array(embeddings))

	return DiarizedResult(valid_segments, labels)

@dataclass
class AudioToTextResult:
	speaker_id: str
	start_time: float
	end_time: float
	text: str

def print_united_result():
	for view_res in silero_vad_speakers:
		logger.debug(f'Speaker: {view_res[0]} - {view_res[1]}')

def unite_results(transcribed_result, diarized_result, labels):
	diarization_result = []
	for i, segment in enumerate(diarized_result):
		speaker = f"Speaker_{labels[i] + 1}"
		diarization_result.append({
			"start": segment['start'],
			"end": segment['end'],
			"speaker": speaker
		})
	silero_vad_speakers = []
	for segment in transcribed_result["segments"]:
			start = segment["start"]
			end = segment["end"]
			text = segment["text"]
			max_overlap = 0
			best_speaker = None

			for diarization_segment in diarization_result:
				overlap_start = max(start, diarization_segment["start"])
				overlap_end = min(end, diarization_segment["end"])
				overlap = max(0, overlap_end - overlap_start)

				if overlap > max_overlap:
					max_overlap = overlap
					best_speaker = diarization_segment["speaker"]

			if best_speaker is None:
				for diarization_segment in diarization_result:
					if diarization_segment["end"] >= start or diarization_segment["start"] <= end:
						best_speaker = diarization_segment["speaker"]
						break

			speaker = best_speaker if best_speaker else "Unknown"
			silero_vad_speakers.append(AudioToTextResult(speaker, start, end, text))

	return silero_vad_speakers


class TranscribingStatus(Enum):
	INITIALIZED = (0.1, 'Processing not started')
	TRANSCRIBING = (0.2, 'Transcribing')
	TRANSCRIBED = (0.5, 'Transcribed')
	DIARIZING = (0.6, 'Diarizing')
	DIARIZED = (0.7, 'Diarizing')
	BUILDING_AUDIO_TO_TEXT = (0.9, 'Building audio to text')
	FAILED = (1.0, 'Failed')
	DONE = (1.0, 'Done')

	def __init__(self, progress, description):
		self.progress = progress
		self.description = description

	@classmethod
	def get_progress(cls, status):
		return status.progress

	@classmethod
	def get_description(cls, status):
		return status.description

class AudioTranscription:
	def __init__(self, audio_path: str):
		logger.debug(f'Создан Audio Analisys для {audio_path}')
		self.audio_path = os.path.join(os.path.expanduser("~"), audio_path)
		self.progress = 0.0
		self.status = TranscribingStatus.INITIALIZED
		self.results = {
			"transcription": None,
			"diarization": None,
			"audio_to_text": None
		}
		self.pbar = tqdm(total=1.0, initial=0, mininterval=0.5, leave=False)

	def full_transcribe(self):
		try:
			# 2. Transcribation
			self.results["transcription"] = self._transcribe()

			# 2. Diarization
			self.results["diarization"] = self._diarize()

			#Unite
			self.results["audio_to_text"] = self._unite_transcribed_and_diarized()

			self._update_status(TranscribingStatus.DONE)
		except Exception as e:
			logger.error(f"Error in {e}")
			self._update_status(TranscribingStatus.FAILED)

	def just_transcribe(self):
		try:
			self.results["transcription"] = self._transcribe()
		except Exception as e:
			logger.error(f"Error in {e}")
			self._update_status(TranscribingStatus.FAILED)

	def just_unite_transcribed_and_diarized(self):
		try:
			# 2. Diarization
			self.results["audio_to_text"] = self._unite_transcribed_and_diarized()
		except Exception as e:
			logger.error(f"Error in {e}")
			self._update_status(TranscribingStatus.FAILED)

	def just_diarize(self):
		try:
			# 2. Diarization
			self.results["diarization"] = self._diarize()
		except Exception as e:
			logger.error(f"Error in {e}")
			self._update_status(TranscribingStatus.FAILED)

	def _diarize(self) -> DiarizedResult:
		self._update_status(TranscribingStatus.DIARIZING)
		diarize_result = diarize(self.audio_path)
		self._update_status(TranscribingStatus.DIARIZED)
		return diarize_result

	def _unite_transcribed_and_diarized(self) -> AudioToTextResult:
		self._update_status(TranscribingStatus.BUILDING_AUDIO_TO_TEXT)
		united_result = unite_results(
				self.results["transcription"],
				self.results["diarization"].valid_segments,
				self.results["diarization"].labels
			)
		self._update_status(TranscribingStatus.DONE)
		return united_result

	def _transcribe(self) -> str:
		self._update_status(TranscribingStatus.TRANSCRIBING)
		logger.debug(f"{os.path.basename(self.audio_path)} - TRANSCRIBING..")
		whisper_model = whisper.load_model(TRANSCRIBE_MODEL_NAME)
		transcribe_result = whisper_model.transcribe(self.audio_path)
		self._update_status(TranscribingStatus.TRANSCRIBED)
		return transcribe_result

	def _update_status(self, status):
		self.status = status
		self.pbar.set_description(f'{TranscribingStatus.get_description(self.status)} - {self.audio_path}')
		self.pbar.n = TranscribingStatus.get_progress(self.status)
		self.pbar.refresh()

