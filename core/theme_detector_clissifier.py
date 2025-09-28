# tag_dialogs_with_model.py
# ----------------------------------------------------------
# Проставляет темы диалогам, используя обученную модель
# (см. train_dialog_tagger.py -> dialog_tagger.joblib).
#
# Зависимости: pandas, numpy, scikit-learn, joblib
#
# Пример запуска:
#   python tag_dialogs_with_model.py --model dialog_tagger.joblib --batch 128 --max-workers 1
# ----------------------------------------------------------

import argparse
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
from uuid import UUID

import numpy as np
from joblib import load

from core.repository.audio_dialog_repository import AudioDialogRepository
from core.service.dialog_row_util_service import print_dialog_to_text
from log_utils import setup_logger

logger = setup_logger(__name__)
audio_dialog_repository = AudioDialogRepository()

# Белый список допустимых тем (совпадает с обучением)
ALLOWED_TOPICS = {
    # продажи/оформление
    "продажа шин",
    "продажа услуг шиномонтажа",
    "продажа услуг установки дисков",
    "продажа услуг по замене масла",
    "продажа дисков",
    "продажа колес",
    "продажа аккумулятора",
    "продажа покрышек",
    "оформление возврата",
    # вопросы
    "вопрос по доставке",
    "вопрос по оформлению",
    "вопрос по шинам",
    "вопрос по колеса",
    "вопрос по дискам",
    "вопрос по возврату",
    "вопрос по аккумулятору",
    "вопрос по шиномонтажу",
    "вопрос по установке дисков",
    "вопрос по замене масла",
    "вопрос по гарантии",
    "вопрос по заказу",
    "вопрос по хранению колес",
    # жалоба
    "жалоба",
}

def _clean_topics(pred_labels: List[str]) -> str:
    """
    Фильтруем предсказанные метки по белому списку и
    возвращаем строку через запятую в нижнем регистре.
    """
    if not pred_labels:
        return ""
    # нормализация + фильтрация
    norm = []
    for t in pred_labels:
        s = (t or "").strip().lower().replace("ё", "е")
        if not s:
            continue
        # сводим любые "жалоба ..." к просто "жалоба"
        if s.startswith("жалоба"):
            s = "жалоба"
        if s in ALLOWED_TOPICS:
            norm.append(s)
    # уникальные в исходном порядке
    uniq = list(dict.fromkeys(norm))
    return ", ".join(uniq)

def _predict_labels_batch(model: Dict[str, Any], texts: List[str]) -> List[List[str]]:
    """
    Батчевый предикт. Возвращает список списков меток (строки).
    """
    wv = model["word_vect"]
    cv = model["char_vect"]
    clf = model["clf"]
    mlb = model["mlb"]
    thr = float(model["threshold"])

    # Быстрая чистка (как в train_dialog_tagger.py: basic_clean)
    def _prep(t: str) -> str:
        if not isinstance(t, str):
            t = str(t or "")
        tt = t.lower().replace("ё", "е")
        return " ".join(tt.split())

    Xw = wv.transform([_prep(t) for t in texts])
    Xc = cv.transform([_prep(t) for t in texts])
    from scipy.sparse import hstack
    X = hstack([Xw, Xc]).tocsr()

    # предпочтительно decision_function у логрег (liblinear)
    if hasattr(clf, "decision_function"):
        scores = clf.decision_function(X)
    else:
        scores = clf.predict_proba(X)

    # нормируем к [0..1] если пришли "логиты"
    if (np.min(scores) < 0) or (np.max(scores) > 1):
        probs = 1.0 / (1.0 + np.exp(-scores))
    else:
        probs = scores

    pred_bin = (probs >= thr).astype(int)
    label_lists = mlb.inverse_transform(pred_bin)
    return [list(lbls) for lbls in label_lists]

def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i+size]

def main():
    # 1) загрузка модели
    logger.info(f"Loading model: dialog_tagger.joblib")
    model = load('dialog_tagger.joblib')

    # 2) выбираем диалоги
    dialogs = audio_dialog_repository.find_all()
    to_process = [d for d in dialogs if not d.theme]  # только без темы
    total = len(to_process)
    logger.info(f"to process: {total}")

    # 3) предикт батчами
    updates = []  # (dialog_id, theme_str)
    for batch in _chunks(to_process, 128):
        texts = []
        ids = []
        for d in batch:
            try:
                txt = print_dialog_to_text(d.id)
            except Exception as e:
                logger.exception(f"failed to build text for {d.id}: {e}")
                txt = ""
            texts.append(txt)
            ids.append(d.id)

        # предикт
        try:
            pred_lists = _predict_labels_batch(model, texts)
        except Exception as e:
            logger.exception(f"batch predict failed: {e}")
            continue

        # пост-обработка (фильтр + строка)
        for dlg_id, labels in zip(ids, pred_lists):
            theme = _clean_topics(labels)
            if theme:
                updates.append((dlg_id, theme))

    logger.info(f"ready to update: {len(updates)} / {total}")

    # 4) запись результатов (можно параллелить)
    def writer(item):
        dlg_id, theme = item
        try:
            # чуть подчистим дефисы/скобки, на всякий
            to_save = (theme or "").replace("[", "").replace("]", "").replace("Тема диалога - ", " ") \
                                   .replace("Тема диалога: ", " ").replace("-", "").strip().lower()
            if to_save:
                audio_dialog_repository.update_theme(dlg_id, to_save)
            return True
        except Exception as e:
            logger.exception(f"failed on {dlg_id}: {e}")
            return False

    ok = 0
    with ThreadPoolExecutor(max_workers=1) as ex:
        for i, done in enumerate(ex.map(writer, updates), 1):
            if done:
                ok += 1
            if i % 50 == 0 or i == len(updates):
                logger.info(f"updated {i}/{len(updates)} (ok={ok})")

    logger.info("done.")

if __name__ == "__main__":
    main()
