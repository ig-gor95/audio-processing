from pathlib import Path
from log_utils import setup_logger
from yaml_reader import ConfigLoader
import whisper

logger = setup_logger(__name__)
config = ConfigLoader("../configs/config.yaml")

_whisper_model = None

def load_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        model_name = config.get('transcribe.model_name')
        logger.info(f"Loading Whisper model: {model_name}")
        _whisper_model = whisper.load_model(model_name, device="cpu")
    return _whisper_model

def transcribe(audio_path: Path) -> str:
    logger.debug(f"{audio_path.name} - TRANSCRIBING...")

    model = load_whisper_model()
    result = model.transcribe(
        str(audio_path),
        beam_size=4,
        temperature=0.5,
        compression_ratio_threshold=2.2
    )
    return result
