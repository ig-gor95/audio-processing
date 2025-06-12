from pathlib import Path

from core.audio_to_text.audio_to_text_processor import audio_to_text_processor
from core.post_processors.client_detector import detect_manager

if __name__ == "__main__":
    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    audio_file = "../data/input/temp/processed_f44a4dab-6f6e-4ce6-a549-a49f670decf9.wav"
    result = audio_to_text_processor(Path(audio_file))
    detect_manager(result)
    for r in result:
        if r.is_copy_line:
            continue
        print(f"[{r.start_time:.2f} - {r.end_time:.2f}]: {r.speaker_id} : {r.text}")
