from pathlib import Path

from core.audio_to_text.diarizer import diarize
from core.audio_to_text.text_to_speaker_resolver import unite_results
from core.audio_to_text.transcriber import transcribe
from core.dto.audio_to_text_result import ProcessingResults


def audio_to_text_processor(audio_path: Path) -> ProcessingResults:
    diarize_result = diarize(audio_path)
    transcribed_result = transcribe(audio_path, diarize_result.valid_segments)
    return unite_results(
        transcribed_result,
        diarize_result
    )
