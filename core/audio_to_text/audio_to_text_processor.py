from pathlib import Path

from core.audio_to_text.diarizer import diarize
from core.audio_to_text.text_to_speaker_resolver import unite_results
from core.audio_to_text.transcriber import transcribe
from core.entity.audio_to_text_result import ProcessingResults


def audio_to_text_processor(audio_path: Path) -> ProcessingResults:
    diarize_result = diarize(audio_path)

    return unite_results(
        transcribe(audio_path),
        diarize_result
    )
