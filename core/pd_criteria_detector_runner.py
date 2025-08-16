import uuid
from itertools import count
from typing import List, Optional

import pymorphy2
from natasha import MorphVocab, NamesExtractor

from core.post_processors.text_processing.DialogueAnalyzer import DialogueAnalyzer
from core.post_processors.text_processing.DialogueAnalyzerPandas import DialogueAnalyzerPandas
from core.post_processors.text_processing.detector.criteria_detector import process_rows_parallel
from core.repository.audio_dialog_repository import AudioDialogRepository
from core.repository.dialog_criteria_repository import DialogCriteriaRepository
from core.repository.dialog_rows_repository import DialogRowRepository
from core.repository.entity.dialog_rows import DialogRow
from core.service.dialog_row_util_service import print_dialog, print_dialog_with_row_text


def chunk_list(rows: List[DialogRow], chunk_size: int = 1000) -> List[List[DialogRow]]:
    return [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]


if __name__ == "__main__":
    analyzer = DialogueAnalyzerPandas()
    analyzer.analyze_dialogue()