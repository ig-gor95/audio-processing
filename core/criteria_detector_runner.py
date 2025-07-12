from typing import List

from core.post_processors.text_processing.criteria_detector import process_rows_parallel
from core.repository.dialog_criteria_repository import DialogCriteriaRepository
from core.repository.dialog_rows_repository import DialogRowRepository
from core.repository.entity.dialog_rows import DialogRow


def chunk_list(rows: List[DialogRow], chunk_size: int = 200) -> List[List[DialogRow]]:
    return [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]

if __name__ == "__main__":
    dialog_rows_repository = DialogRowRepository()
    dialog_criteria_repository = DialogCriteriaRepository()
    rows = dialog_rows_repository.find_all()[:100]
    result = process_rows_parallel(rows)
    print(result)
