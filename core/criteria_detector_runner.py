from typing import List

import pandas as pd

from core.config.datasource_config import DatabaseManager
from core.post_processors.criteria_detector import process_rows_parallel
from core.repository.dialog_criteria_repository import DialogCriteriaRepository
from core.repository.dialog_rows_repository import DialogRowRepository
from core.repository.entity.dialog_rows import DialogRow
from yaml_reader import ConfigLoader


def chunk_list(rows: List[DialogRow], chunk_size: int = 500) -> List[List[DialogRow]]:
    return [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]

if __name__ == "__main__":
    dialog_rows_repository = DialogRowRepository()
    dialog_criteria_repository = DialogCriteriaRepository()

    rows = dialog_rows_repository.find_all()
    count = 0

    print(len(rows))

    engine = DatabaseManager.get_engine()
    for chunk in chunk_list(rows):
        data = process_rows_parallel(chunk)
        dialog_criteria_repository.save_bulk(chunk)
