# from fontTools.misc.cython import returns  # <- remove this line

from concurrent.futures import ThreadPoolExecutor
from core.post_processors.llm_processing.objections_resolver import resolve_theme
from core.repository.audio_dialog_repository import AudioDialogRepository
from core.service.dialog_row_util_service import print_dialog_to_text
from log_utils import setup_logger

logger = setup_logger(__name__)
audio_dialog_repository = AudioDialogRepository()

if __name__ == "__main__":
    dialogs = audio_dialog_repository.find_all()
    to_process = [d for d in dialogs if d.theme is None]
    total = len(to_process)
    logger.info(f"to process: {total}")


    def worker(dialog):
        try:
            dialog_text = print_dialog_to_text(dialog.id)
            theme = resolve_theme(dialog_text)
            theme = (theme or "").strip()
            return dialog.id, (theme if theme else None)
        except Exception as e:
            logger.exception(f"failed on {dialog.id}: {e}")
            return dialog.id, None


    with ThreadPoolExecutor(max_workers=1) as ex:
        for i, (dialog_id, theme) in enumerate(ex.map(worker, to_process), 1):
            logger.info(f"{i} of {total}")
            if theme:
                try:
                    audio_dialog_repository.update_theme(dialog_id,
                                                         theme.replace('[', '').replace(']', '').lower().replace(
                                                             'Тема диалога - ', ' ').replace('Тема диалога: ', ''))
                except Exception as e:
                    logger.exception(f"failed on {dialog_id}: {e}")
                    continue
