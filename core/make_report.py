# main orchestrator: 1) загрузка, 2) расплющивание LLM, 3) агрегация, 4) запись 2 листов

import pandas as pd
import numpy as np
import xlsxwriter

from report_common import (
    THEME_COL, TEXT_COL, DIALOG_COL, SPEAKER_COL, ALT_SPEAKER_COL, ROWNUM_COL,
    START_COL, END_COL, FILE_COL, DUR_COL, LOUD_COL, LLM_COL,
    make_unique_columns, _safe_json_loads, llm_flat,
    detect_speaker_col, build_dialog_text_block, sales_mean_for_dialog
)
from build_summary import aggregate_per_dialog, compute_blocks, write_summary_sheet
from build_dialogs import write_dialogs_sheet

try:
    from core.repository.audio_dialog_repository import AudioDialogRepository
except Exception:
    AudioDialogRepository = None

def _pick_col(df: pd.DataFrame, name: str) -> str:
    if name in df.columns: return name
    for c in df.columns:
        if c == name or c.startswith(name + "."):
            return c
    return name

def make_report(df: pd.DataFrame, out_path: str = "dialogs_report.xlsx") -> str:
    if df.empty:
        raise ValueError("Пустой DataFrame — нечего выгружать.")

    df = df.copy()
    df.columns = make_unique_columns(df.columns)

    # нормализуем названия столбцов
    ren = {}
    for name in (TEXT_COL, THEME_COL, SPEAKER_COL, ALT_SPEAKER_COL,
                 ROWNUM_COL, START_COL, END_COL, FILE_COL, DUR_COL, LOUD_COL):
        pick = _pick_col(df, name)
        if pick in df.columns and pick != name:
            ren[pick] = name
    if ren:
        df = df.rename(columns=ren)

    if DIALOG_COL not in df.columns:
        # на случай, если поле тоже «сплющено»
        pick = _pick_col(df, DIALOG_COL)
        if pick in df.columns:
            if pick != DIALOG_COL:
                df = df.rename(columns={pick: DIALOG_COL})
        else:
            raise ValueError(f"Нет колонки '{DIALOG_COL}' (идентификатор диалога).")

    df[DIALOG_COL] = df[DIALOG_COL].astype(str)

    # сортировка строк диалогов
    if ROWNUM_COL in df.columns:
        df = df.sort_values([DIALOG_COL, ROWNUM_COL], na_position="last")
    elif START_COL in df.columns and END_COL in df.columns:
        df = df.sort_values([DIALOG_COL, START_COL, END_COL], na_position="last")
    else:
        df = df.sort_values([DIALOG_COL], na_position="last")

    # ---- расплющиваем новую LLM-схему ----
    llm_col = None
    for c in df.columns:
        if c == LLM_COL or c.startswith(LLM_COL + "."):
            llm_col = c; break

    if llm_col and llm_col in df.columns:
        llm_flat_rows = []
        for v in df[llm_col]:
            obj = _safe_json_loads(v)
            llm_flat_rows.append(llm_flat(obj) if isinstance(obj, dict) else {})
        llm_df = pd.DataFrame(llm_flat_rows)
        df = pd.concat([df.reset_index(drop=True), llm_df.reset_index(drop=True)], axis=1)

        # типизация флагов/скор
        for c in ["llm_obj_detected","llm_obj_ack","llm_obj_sol","llm_obj_check",
                  "llm_next_set","llm_wrap_up","llm_addon_offered","llm_addon_accepted","llm_obj_score"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # текст блоки — отдельно (дорого, поэтому после сортировки)
    dialog_text_map = {}
    for did, sub in df.groupby(DIALOG_COL, dropna=False):
        dialog_text_map[str(did)] = build_dialog_text_block(sub)
    # агрегация по диалогу
    summary = aggregate_per_dialog(df)
    summary["dialog_text"] = summary["dialog_id"].map(dialog_text_map).fillna("")

    # метрики для "Сводка"
    total_dialogs, total_hours, theme_counts, rates, dist_blocks, corr_matrix = compute_blocks(summary)

    # запись в xlsx: только 2 листа
    wb = xlsxwriter.Workbook(out_path)

    write_summary_sheet(wb, summary, theme_counts, rates, dist_blocks, corr_matrix,
                        total_dialogs, total_hours)
    write_dialogs_sheet(wb, summary)

    wb.close()
    return out_path

if __name__ == "__main__":
    repo = AudioDialogRepository()
    df = repo.get_all_for_report()
    out = make_report(df, "dialogs_report.xlsx")
    print(f"Готово: {out}")
