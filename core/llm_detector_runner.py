import json
import re
from core.post_processors.llm_processing.objections_resolver import resolve_llm_data
from core.repository.audio_dialog_repository import AudioDialogRepository
from core.service.dialog_row_util_service import print_dialog_to_text
from log_utils import setup_logger

logger = setup_logger(__name__)
audio_dialog_repository = AudioDialogRepository()


_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)

def _strip_code_fences(s: str) -> str:
    return _CODE_FENCE_RE.sub("", s).strip()

def _first_json_blob(s: str) -> str | None:
    start = None
    stack = []
    for i, ch in enumerate(s):
        if ch in "{[":
            start = i
            stack = [ch]
            break
    if start is None:
        return None
    close = { "{": "}", "[": "]" }
    opener = s[start]
    need = close[opener]
    depth = 0
    for j in range(start, len(s)):
        c = s[j]
        if c == opener:
            depth += 1
        elif c == need:
            depth -= 1
            if depth == 0:
                return s[start:j+1]
    return None

def _to_jsonb_dict(raw_text: str) -> dict:
    """
    resolve_llm_data возвращает строку с JSON -> приводим к dict для JSONB.
    Если пришёл массив, заворачиваем его под ключ 'data'.
    Если парс не удался — {}.
    """
    if raw_text is None:
        return {}
    candidates = [raw_text, _strip_code_fences(raw_text)]
    blob = _first_json_blob(candidates[-1])
    if blob:
        candidates.append(blob)

    for c in candidates:
        try:
            val = json.loads(c)
            if isinstance(val, dict):
                return val
            if isinstance(val, list):
                return {"data": val}
        except Exception:
            continue
    logger.warning("LLM вернул непарсимый JSON. Сохраняю пустой объект.")
    return {}

if __name__ == "__main__":
    dialogs = audio_dialog_repository.find_all()
    total = len(dialogs)
    logger.info(f"to process: {total}")

    for idx, dialog in enumerate(dialogs, start=1):
        if dialog.llm_data_short is not None:
            continue

        logger.info(f"{idx}: {total}")
        if idx in [380, 1313, 1275] :
            continue
        dialog_text = print_dialog_to_text(dialog.id)
        if len(dialog_text) < 5:
            continue
        llm_text = resolve_llm_data(dialog_text)

        llm_data_short = _to_jsonb_dict(llm_text)

        audio_dialog_repository.update_llm_data(dialog.id, llm_data_short)

