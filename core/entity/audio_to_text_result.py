from dataclasses import dataclass


@dataclass
class AudioToTextResult:
    speaker_id: str
    start_time: float
    end_time: float
    text: str
    is_copy_line: bool