from dataclasses import dataclass, field
from typing import List


@dataclass
class ObjectionProp:
    objection_str: str
    manager_offer_str: str
    was_resolved: bool


@dataclass
class ProcessingResult:

    def __init__(self,
                 speaker: str,
                 text: str,
                 start_time: float,
                 end_time: float,
                 is_copy_line: bool):
        self.speaker_id = speaker
        self.text = text
        self.start_time = start_time
        self.end_time = end_time
        self.is_copy_line = is_copy_line

    speaker_id: str
    start_time: float
    end_time: float
    text: str
    is_copy_line: bool = False

    def to_string(self) -> str:
        return f"{self.speaker_id}: {self.text}"


@dataclass
class ProcessingResults:

    items: List[ProcessingResult] = field(default_factory=list)
    objection_prop: List[ObjectionProp] = field(default_factory=list)

    def to_string(self) -> str:
        string_res = ""
        for result in self.items:
            if string_res:
                string_res += "\n"
            string_res += result.to_string()
        return string_res
