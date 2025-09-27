# build_training_dataset.py
# ----------------------------------------------------------
# Готовит обучающий датасет диалогов для мультилейбл-классификации тем.
# Берёт построчные данные из AudioDialogRepository.get_all_for_report()
# и агрегирует в один текст на диалог + темы (через запятую).
#
# Установка: pip install pandas numpy
# Запуск:
#   python build_training_dataset.py --out_csv dialogs_ds.csv --out_jsonl dialogs_ds.jsonl
# ----------------------------------------------------------

import argparse
import re
import json
from typing import List, Dict

import numpy as np
import pandas as pd

# ваш репозиторий
from core.repository.audio_dialog_repository import AudioDialogRepository

# --- ожидаемые имена колонок в DataFrame из репозитория
DIALOG_COL = "audio_dialog_fk_id"
TEXT_COL   = "row_text"
THEME_COL  = "theme"
SPEAKER1   = "detected_speaker_id"
SPEAKER2   = "speaker_id"

# допустимые темы (опц., можно оставить пустым, чтобы не фильтровать)
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
    # жалоба (унифицируем «жалоба на …» в «жалоба»)
    "жалоба",
}

def _norm_topic(t: str) -> str:
    """Нормализация темы: нижний регистр, ё→е, обрезка пробелов, «жалоба на …» -> «жалоба»."""
    s = (t or "").strip().lower().replace("ё", "е")
    if not s:
        return ""
    if s.startswith("жалоба"):
        return "жалоба"
    return s

def _split_topics(cell: str) -> List[str]:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    parts = [p.strip() for p in str(cell).split(",")]
    out: List[str] = []
    for p in parts:
        n = _norm_topic(p)
        if not n:
            continue
        if ALLOWED_TOPICS and n not in ALLOWED_TOPICS:
            # если хотите не фильтровать — уберите следующую строку
            continue
        out.append(n)
    # уникальные в порядке появления
    return list(dict.fromkeys(out))

def _speaker_col(df: pd.DataFrame) -> str:
    if SPEAKER1 in df.columns: return SPEAKER1
    if SPEAKER2 in df.columns: return SPEAKER2
    return ""

def _fmt_line(row: pd.Series, spk_col: str) -> str:
    txt = str(row.get(TEXT_COL, "") or "")
    txt = re.sub(r"\s+", " ", txt).strip()
    if not txt:
        return ""
    if spk_col:
        who = str(row.get(spk_col, "") or "").strip()
        return f"{who}: {txt}" if who else txt
    return txt

def build_dataset_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """Склеивает построчный df → по одному ряду на диалог: dialog_id, text, topics(list)."""
    need_cols = [DIALOG_COL, TEXT_COL, THEME_COL]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"В данных нет обязательной колонки: {c}")

    # удалим пустые строки текста
    df = df[df[TEXT_COL].astype(str).str.strip().ne("")].copy()
    # сортировка внутри диалога по порядковому номеру, если есть
    if "row_num" in df.columns:
        df = df.sort_values([DIALOG_COL, "row_num"], na_position="last")
    elif "start" in df.columns and "end" in df.columns:
        df = df.sort_values([DIALOG_COL, "start", "end"], na_position="last")
    else:
        df = df.sort_values([DIALOG_COL], na_position="last")

    spk_col = _speaker_col(df)

    rows: List[Dict] = []
    for did, sub in df.groupby(DIALOG_COL, dropna=False):
        # текст
        lines = [_fmt_line(r, spk_col) for _, r in sub.iterrows()]
        lines = [ln for ln in lines if ln]
        text = "\n".join(lines).strip()

        # темы: собираем со всех строк диалога, затем нормализуем
        topics: List[str] = []
        for _, r in sub.iterrows():
            topics.extend(_split_topics(r.get(THEME_COL, "")))
        topics = list(dict.fromkeys(topics))

        rows.append({
            "dialog_id": str(did),
            "text": text,
            "topics": topics,                          # список
            "topics_csv": ", ".join(topics) if topics else ""  # строка через запятую (удобно для CSV)
        })

    ds = pd.DataFrame(rows)
    # оставим только диалоги, у которых есть разметка
    ds = ds[ds["topics"].map(len) > 0].reset_index(drop=True)
    return ds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", default="dialogs_ds.csv", help="куда сохранить CSV (dialog_id,text,topics_csv)")
    ap.add_argument("--out_jsonl", default="", help="(опц.) JSONL с полями dialog_id,text,topics(list)")
    args = ap.parse_args()

    repo = AudioDialogRepository()
    # Берём построчные данные (как в отчёте): включает audio_dialog_fk_id, row_text, theme, speaker и т.п.
    df = repo.get_all_for_report()

    ds = build_dataset_from_df(df)

    # CSV (компактный)
    ds[["dialog_id", "text", "topics_csv"]].to_csv(args.out_csv, index=False)
    print(f"Saved CSV: {args.out_csv}  (rows: {len(ds)})")

    # JSONL (полные списки меток)
    if args.out_jsonl:
        with open(args.out_jsonl, "w", encoding="utf-8") as f:
            for _, r in ds.iterrows():
                rec = {"dialog_id": r["dialog_id"], "text": r["text"], "topics": r["topics"]}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Saved JSONL: {args.out_jsonl}")

if __name__ == "__main__":
    main()
