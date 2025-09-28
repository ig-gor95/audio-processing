from concurrent.futures import ThreadPoolExecutor
from core.post_processors.llm_processing.objections_resolver import resolve_theme
from core.repository.audio_dialog_repository import AudioDialogRepository
from core.service.dialog_row_util_service import print_dialog_to_text
from log_utils import setup_logger

logger = setup_logger(__name__)
audio_dialog_repository = AudioDialogRepository()
check_list = [
    'продажа шин',
    'продажа услуг шиномонтажа',
    'продажа услуг установки дисков',
    'продажа услуг по замене масла',
    'продажа дисков',
    'продажа колес',
    'продажа аккумулятора',
    'продажа покрышек',
    'оформление возврата',
    'вопрос по доставке'','
    'вопрос по оформлению',
    'вопрос по шинам',
    'вопрос по колеса',
    'вопрос по дискам',
    'вопрос по заказу',
    'вопрос по возврату',
    'вопрос по аккумулятору',
    'вопрос по шиномонтажу',
    'вопрос по установке дисков',
    'вопрос по замене масла',
    'вопрос по гарантии',
    'вопрос по хранению колес'
]


def check_theme(theme):
    if 'жалоба' in theme.lower():
        return True

    phrases = [phrase.strip() for phrase in theme.split(', ')]

    cleaned = ''
    for phrase in phrases:
        if phrase in check_list:
            if cleaned != '':
                cleaned = cleaned + ' ' + phrase
            else:
                cleaned = phrase

    return cleaned

if __name__ == "__main__":
    dialogs = audio_dialog_repository.find_all()
    to_process = dialogs
    total = len(to_process)
    logger.info(f"to process: {total}")


    def worker(dialog):
        try:
            if dialog.theme is not None and dialog.theme != '':
                return dialog.id, None
            dialog_text = print_dialog_to_text(dialog.id)
            theme = resolve_theme(dialog_text)
            print(theme)
            theme = check_theme(theme)
            if theme == '':
                return dialog.id, None

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
                                                             'Тема диалога - ', ' ').replace('Тема диалога: ', '').replace('-', ''))
                except Exception as e:
                    logger.exception(f"failed on {dialog_id}: {e}")
                    continue
