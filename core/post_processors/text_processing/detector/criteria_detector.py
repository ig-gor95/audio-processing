from typing import List
from core.post_processors.text_processing.DialogueAnalyzer import DialogueAnalyzer
from core.repository.dialog_criteria_repository import DialogCriteriaRepository
from core.repository.entity.dialog_rows import DialogRow
from multiprocessing import Pool
from tqdm import tqdm

dialog_criteria_repository = DialogCriteriaRepository()


def analyze_dialogue_enhanced(text, row_id):
    dialog_analyzer = DialogueAnalyzer()
    # dialog_criteria_repository.delete_by_dialog_row_fk_id(row_id)
    return dialog_analyzer.analyze_dialogue(text, row_id)


def process_row_wrapper(args):
    if args['speaker_id'] == 'CLIENT':
        return None
    return analyze_dialogue_enhanced(args['row_text'], args['row_id'])


def process_rows_parallel(rows: List[DialogRow], processes=4):
    data = []
    count = 0

    args = [{"row_text": row.row_text, "row_id": row.id, "speaker_id": row.speaker_id} for row in rows]

    with Pool(processes=processes) as pool:
        for result in tqdm(pool.imap(process_row_wrapper, args), total=len(rows)):
            if result is not None:
                data.append(result)
            count += 1

    return data