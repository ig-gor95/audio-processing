import uuid
from typing import List

from core.repository.entity.dialog_rows import DialogRow
from core.service.dialog_row_util_service import print_dialog


def chunk_list(rows: List[DialogRow], chunk_size: int = 200) -> List[List[DialogRow]]:
    return [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]

if __name__ == "__main__":
    # dialog_rows_repository = DialogRowRepository()
    # dialog_criteria_repository = DialogCriteriaRepository()
    # rows = dialog_rows_repository.find_all()[:100]
    # result = process_rows_parallel(rows)
    # print(result)
    print_dialog(uuid.UUID("3e523cf2-05b3-426f-ad5e-5cdd860c86c7"))
