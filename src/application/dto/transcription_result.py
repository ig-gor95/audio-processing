from dataclasses import dataclass, field
from typing import List


@dataclass
class ObjectionProp:
    objection_str: int
    manager_offer_str: int
    was_resolved: bool


@dataclass
class ProcessingResult:

    def __init__(self,
                 phrase_id: int,
                 speaker: str,
                 text: str,
                 start_time: float,
                 end_time: float,
                 is_copy_line: bool):
        self.phrase_id = phrase_id
        self.speaker_id = speaker
        self.text = text
        self.start_time = start_time
        self.end_time = end_time
        self.is_copy_line = is_copy_line

    phrase_id: int
    speaker_id: str
    start_time: float
    end_time: float
    text: str
    is_copy_line: bool = False

    def to_string(self) -> str:
        return f"{self.speaker_id}) {self.speaker_id}: {self.text}"


@dataclass
class ProcessingResults:
    items: List[ProcessingResult] = field(default_factory=list)
    objection_prop: List[ObjectionProp] = field(default_factory=list)

    def to_string(self) -> str:
        string_res = ""
        for result in self.items:
            string_res += "\n" + result.to_string()
        return string_res

    def add_phrase(self,
                   speaker: str,
                   text: str,
                   start_time: float,
                   end_time: float,
                   is_copy_line: bool):
        self.items.append(ProcessingResult(
            phrase_id=len(self.items) + 1,
            speaker=speaker,
            text=text.replace("Продолжение следует...", '').replace('Субтитры сделал DimaTorzok', ''),
            start_time=start_time,
            end_time=end_time,
            is_copy_line=is_copy_line)
        )

    def get_last_phrase(self):
        return self.items[-1] if self.items else None