from typing import List
from uuid import UUID

from core.repository.audio_dialog_repository import AudioDialogRepository
from core.repository.dialog_rows_repository import DialogRowRepository
from core.repository.entity.dialog_rows import DialogRow


def get_all_rows_in_order(dialog_id: UUID) -> List[DialogRow]:
    audio_dialog_repository = AudioDialogRepository()
    dialog_rows_repository = DialogRowRepository()
    audio_dialog = audio_dialog_repository.find_by_id(dialog_id)
    if audio_dialog is None:
        raise RuntimeError(f"Audio dialog with id {dialog_id} not found")
    rows = dialog_rows_repository.find_by_dialog_id(audio_dialog.id)
    rows = sorted(rows, key=lambda x: x.row_num)
    return rows


def print_dialog(dialog_id: UUID):
    rows = get_all_rows_in_order(dialog_id)
    for row in rows:
        row.print()

def print_dialog_to_text(dialog_id: UUID):
    rows = get_all_rows_in_order(dialog_id)
    parts = []
    for row in rows:
        parts.append(row.to_str())
    return "\n".join(parts)

def print_dialog_with_row_text(dialog_id: UUID, should_be_text: str):
    rows = get_all_rows_in_order(dialog_id)
    for row in rows:
        if (should_be_text in row.row_text.lower()):
            row.print()
            print(dialog_id)
