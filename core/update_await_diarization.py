import uuid
from itertools import count
from multiprocessing import Pool
from typing import List, Optional

from tqdm import tqdm

from core.post_processors.text_processing.criteria_utils import find_phrase
from core.repository.audio_dialog_repository import AudioDialogRepository
from core.repository.dialog_criteria_repository import DialogCriteriaRepository
from core.repository.dialog_rows_repository import DialogRowRepository
from core.repository.entity.audio_dialog import AudioDialog
from core.repository.entity.dialog_rows import DialogRow
from yaml_reader import ConfigLoader


def chunk_list(rows: List[DialogRow], chunk_size: int = 8000) -> List[List[DialogRow]]:
    return [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]
greeting_phrases = ConfigLoader("post_processors/config/phrase_patterns.yaml").get("patterns")['greetings']

def process_row_wrapper(args):
    dialog_id = args['dialog_id']
    dialog_rows_repository = DialogRowRepository()
    dialog_criteria_repository = DialogCriteriaRepository()
    rows = dialog_rows_repository.find_by_dialog_id(dialog_id)
    rows = sorted(rows, key=lambda x: x.row_num)
    criterias = dialog_criteria_repository.find_by_row_fk_id_in([row.id for row in rows])
    result_dict = {item.dialog_row_fk_id: item for item in criterias}
    should_be_client_rows = []
    should_be_sales_rows = []
    start_waiting = False
    for row in rows:
        row_criteria = result_dict.get(row.id)
        if start_waiting and row_criteria is None:
            greeting = find_phrase(row.row_text, greeting_phrases)
            if greeting is not None:
                should_be_sales_rows.append(row)
                start_waiting = False
                continue
        if row_criteria is None:
            continue
        if row_criteria.await_requests is not None and row_criteria.await_requests != '""':
            start_waiting = True
            continue
        if row.row_num > 5 and row_criteria.greeting_phrase is not None and start_waiting:
            start_waiting = False
        elif start_waiting:
            should_be_client_rows.append(row)
    if start_waiting:
        return
    else:
        if len(should_be_client_rows) > 0:
            for should_be_client_row in should_be_client_rows:
                dialog_rows_repository.update_speaker_id_by_id(should_be_client_row.id, "CLIENT")
        if len(should_be_sales_rows) > 0:
            for should_be_client_row in should_be_client_rows:
                dialog_rows_repository.update_speaker_id_by_id(should_be_client_row.id, "SALES")

def process_rows_parallel(dialogs: List[AudioDialog], processes=1):
    data = []
    count = 0

    args = [{"dialog_id": dialog.id} for dialog in dialogs]

    with Pool(processes=processes) as pool:
        for result in tqdm(pool.imap(process_row_wrapper, args), total=len(dialogs)):
            if result is not None:
                data.append(result)
            count += 1

    return data

if __name__ == "__main__":
    audio_dialog_repository = AudioDialogRepository()
    dialog_criteria_repository = DialogCriteriaRepository()

    dialogs = audio_dialog_repository.find_all()[3900:]
    for chunk in chunk_list(dialogs):
        result = process_rows_parallel(chunk)

