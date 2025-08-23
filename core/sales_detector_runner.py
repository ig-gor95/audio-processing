from typing import List

from core.post_processors.text_processing.detector.sales_detector import SalesDetector
from core.repository.audio_dialog_repository import AudioDialogRepository
from core.repository.dialog_criteria_repository import DialogCriteriaRepository
from core.repository.dialog_rows_repository import DialogRowRepository
from core.repository.entity.dialog_rows import DialogRow


def chunk_list(rows: List[DialogRow], chunk_size: int = 200) -> List[List[DialogRow]]:
    return [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]

if __name__ == "__main__":
    audio_dialog_repository = AudioDialogRepository()
    dialog_rows_repository = DialogRowRepository()
    dialog_criteria_repository = DialogCriteriaRepository()

    dialogs = audio_dialog_repository.find_all()
    sales_detector = SalesDetector()
    counter = 0
    total = len(dialogs)
    for dialog in dialogs:
        counter += 1
        print(f'{counter} of {total}')
        dialog_rows = dialog_rows_repository.find_by_dialog_id(dialog.id)
        dialog_rows = sorted(dialog_rows, key=lambda x: x.row_num)
        sales_detector(dialog_rows)
        for row in dialog_rows:
            dialog_rows_repository.update_speaker_id_by_id(row.id, row.speaker_id)
