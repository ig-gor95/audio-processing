import uuid
from typing import List

from core.post_processors.text_processing.detector.criteria_detector import process_rows_parallel
from core.repository.audio_dialog_repository import AudioDialogRepository
from core.repository.dialog_criteria_repository import DialogCriteriaRepository
from core.repository.dialog_rows_repository import DialogRowRepository
from core.repository.entity.dialog_rows import DialogRow
from core.service.dialog_row_util_service import print_dialog, print_dialog_with_row_text


def chunk_list(rows: List[DialogRow], chunk_size: int = 500) -> List[List[DialogRow]]:
    return [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]

if __name__ == "__main__":
    dialog_rows_repository = DialogRowRepository()
    audio_dialog_repository = AudioDialogRepository()
    dialog_criteria_repository = DialogCriteriaRepository()

    # rows = dialog_rows_repository.find_by_dialog_id(uuid.UUID('b6444029-efbd-4994-9f7b-3e07022fa386'))
    rows = dialog_rows_repository.find_all()
    for chunk in chunk_list(rows):
        result = process_rows_parallel(chunk)
        dialog_criteria_repository.update_all_criteria(result)
    # dialogs = audio_dialog_repository.find_all()
    # for dialog in dialogs:
    #     print_dialog_with_row_text(dialog.id, 'за ожидание')
    #
    # print_dialog(uuid.UUID("1c6769b3-397b-44ea-b203-f6740a3677f4"))
