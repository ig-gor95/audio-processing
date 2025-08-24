import uuid
from itertools import count
from typing import List, Optional

import pymorphy2
from natasha import MorphVocab, NamesExtractor

from core.post_processors.text_processing.DialogueAnalyzer import DialogueAnalyzer
from core.post_processors.text_processing.detector.criteria_detector import process_rows_parallel
from core.repository.audio_dialog_repository import AudioDialogRepository
from core.repository.dialog_criteria_repository import DialogCriteriaRepository
from core.repository.dialog_rows_repository import DialogRowRepository
from core.repository.entity.dialog_rows import DialogRow
from core.service.dialog_row_util_service import print_dialog, print_dialog_with_row_text


def chunk_list(rows: List[DialogRow], chunk_size: int = 1000) -> List[List[DialogRow]]:
    return [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]


def extract_valid_names(text: str) -> Optional[str]:
    """Extract names considering introduction context."""
    morph_vocab = MorphVocab()
    extractor = NamesExtractor(morph_vocab)
    matches = extractor(text)
    # More sophisticated filtering
    morph_analyzer = pymorphy2.MorphAnalyzer()

    valid_names = []
    for match in matches:
        if match.fact.first is None:
            continue
        name = match.fact.first
        parsed = morph_analyzer.parse(name)
        if not parsed:
            continue

        is_proper_noun = any('Name' in p.tag or 'Surn' in p.tag for p in parsed)
        if (is_proper_noun and name[0].isupper()):  # should be capitalized
            valid_names.append(name)

    return ', '.join(valid_names) if valid_names else None


if __name__ == "__main__":
    # res = extract_valid_names("Вот такая тема возникла. А клава придет")
    # print(res)
    # dialog_rows_repository = DialogRowRepository()
    # audio_dialog_repository = AudioDialogRepository()
    # dialog_criteria_repository = DialogCriteriaRepository()

    # rows = dialog_rows_repository.find_by_dialog_id(uuid.UUID('b6444029-efbd-4994-9f7b-3e07022fa386'))
    # rows = dialog_rows_repository.find_all()

    # rows = dialog_rows_repository.find_rows_without_criteria()
    # total = len(rows)
    # count = 0
    # for chunk in chunk_list(rows):
    #     result = process_rows_parallel(chunk)
    #     count += 1000
    #     print(f"saved {len(result)} rows. Processed {count} of {total} rows.")
    #     dialog_criteria_repository.update_all_criteria(result)
    # dialogs = audio_dialog_repository.find_all()
    # for dialog in dialogs:
    #     print_dialog_with_row_text(dialog.id, 'за ожидание')

    print_dialog(uuid.UUID("50de1958-185c-461a-96e5-ddc5c6246f58"))
