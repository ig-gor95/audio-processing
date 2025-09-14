from log_utils import setup_logger
from saiga import SaigaClient

logger = setup_logger(__name__)

def resolve_theme(dialog_text: str) -> str:
    saiga = SaigaClient()

    try:
        return saiga.ask(
            f"""
            Я дам тебе диалог. После фразы НАЧАЛО начнется диалог. Завершится словом конец.
            Нужно в ответе вывести только тему диалога. Очень кратко. Не более трех слов.
            НАЧАЛО
            {dialog_text}
            КОНЕЦ
            
            Ответь ОЧЕНЬ коротко. Назови тему диалога. Ничего больше не говори
            """
        )
    except Exception as e:
        logger.info(e)
        raise e
