# make_dialog_report.py
# -----------------------------------------------
# Excel-отчёт по диалогам:
#  - Сводка (в столбик) над основной таблицей
#  - Одна строка на диалог + скрываемая строка с полным текстом (высота авто-рассчитана)
#  - Критерии (только по Sales) в основной таблице, с русскими названиями
#  - Листы с графиками по лексике (кроме «Уменьшительные», «Междометия»)
#  - Лист "Громкость" с потенциально громкими репликами (Sales)
#
# Установка: pip install pandas numpy xlsxwriter
# Запуск:    python make_dialog_report.py --csv dialogs.csv --out dialogs_report.xlsx --top 20
# -----------------------------------------------

import argparse
import re
import math
import textwrap
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import xlsxwriter

from core.repository.audio_dialog_repository import AudioDialogRepository

# ---- Базовые колонки ----
THEME_COL = "theme"
TEXT_COL = "row_text"
DIALOG_COL = "audio_dialog_fk_id"
SPEAKER_COL = "detected_speaker_id"   # если только speaker_id — код учтёт
ALT_SPEAKER_COL = "speaker_id"
ROWNUM_COL = "row_num"
START_COL, END_COL = "start", "end"
FILE_COL, STATUS_COL, DUR_COL = "file_name", "status", "duration"
LOUD_COL = "mean_loudness"

SALES_VALUE = "SALES"
EXCEL_MAX_CELL = 32767

# ---- Критерии: внутр. имена -> отображаемые русские заголовки ----
CRITERIA_LABELS: List[Tuple[str, str]] = [
    ("greeting_phrase",          "Приветствие"),
    ("telling_name_phrases",     "Обращение по имени"),
    ("found_name",               "Назвал имя"),
    ("order_offer",              "Предложение заказа"),
    ("order_processing",         "Оформление заказа"),
    ("order_resume",             "Подведение итогов"),
    ("working_hours",            "Режим работы"),
    ("reserve_terms",            "Сроки резерва"),
    ("delivery_terms",           "Сроки доставки"),
    ("axis_attention",           "Акцент (ось)"),
    ("await_requests",           "Вход в ожидание"),
    ("await_requests_exit",      "Выход из ожидания"),
    # лексика
    ("parasite_words",           "Слова-паразиты"),
    ("stop_words",               "Стоп-слова"),
    ("slang",                    "Сленг"),
    ("non_professional_phrases", "Непроф. фразы"),
    ("inappropriate_phrases",    "Неприемлемые"),
    ("swear_words",              "Мат"),
    ("order_type",               "Тип заказа/подбор"),
]

# Лексика для отдельных листов (без «Уменьшительные» и «Междометия»)
LEXICON_COLS = [
    ("parasite_words",           "Паразиты"),
    ("stop_words",               "Стоп-слова"),
    ("slang",                    "Сленг"),
    ("swear_words",              "Мат"),
    ("inappropriate_phrases",    "Неприемлемые"),
    ("non_professional_phrases", "Непроф. фразы"),
]

_SPLIT_RE = re.compile(r"[;,|/\\]+")
_TRIM = " \"'“”«»·-–—"


# ---------- Утилиты ----------

def make_unique_columns(cols: pd.Index) -> pd.Index:
    seen: Dict[str, int] = {}
    out: List[str] = []
    for c in cols:
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}.{seen[c]}")
    return pd.Index(out)

def _as_str(x) -> str:
    if isinstance(x, pd.Series):
        x = x.iloc[0] if not x.empty else ""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x)

def detect_speaker_col(df: pd.DataFrame) -> Optional[str]:
    if SPEAKER_COL in df.columns:
        return SPEAKER_COL
    if ALT_SPEAKER_COL in df.columns:
        return ALT_SPEAKER_COL
    return None

def classify_theme_by_rules(s: str) -> str:
    s_low = (s or "").lower().replace("ё", "е")
    if "оформление возврата" in s_low or "возврат" in s_low:
        return "Возврат"
    if "жалоб" in s_low or "претенз" in s_low:
        return "Жалоба"
    if "вопрос" in s_low:
        return "Вопрос"
    if "покупка" in s_low or "продаж" in s_low or "sale" in s_low:
        return "Покупка"
    return ""

# 3+ подряд одно и то же слово -> оставить 2
_REPEAT_WORD_RX = re.compile(r'(?iu)\b(\w+)(?:[\s,\-–—.?!…]+\1\b){2,}')

def _collapse_repeats(line: str, keep: int = 2) -> str:
    if keep < 1:
        keep = 1
    def _repl(m: re.Match) -> str:
        w = m.group(1)
        sep = " "
        return (w + sep) * (keep - 1) + w
    prev, cur = None, line
    while prev != cur:
        prev, cur = cur, _REPEAT_WORD_RX.sub(_repl, cur)
    return cur

# aaaaaa -> aaa  (один символ повторяется 6+ раз)
_REPEAT_CHAR_RX = re.compile(r'(?iu)(\w)\1{5,}')

# worthworthworth...  (повторяющийся паттерн длиной 2-10 символов, 4+ раз)
_REPEAT_CHUNK_RX = re.compile(r'(?iu)\b([a-zа-яё0-9]{2,10})\1{3,}\b')

# очень длинные «слова» (без пробелов/знаков) — будем тримить/удалять
_LONG_TOKEN_RX = re.compile(r'(?iu)\b\w{41,}\b')  # >40 символов

def _shrink_long_noise(line: str, soft_trunc_len: int = 40, hard_drop_len: int = 80) -> str:
    # 1) сжать повторы одиночных символов до трёх
    line = _repeat_char_shrink(line)
    # 2) сжать повторяющиеся куски до двух повторов
    line = _REPEAT_CHUNK_RX.sub(lambda m: m.group(1) * 2, line)
    # 3) трим/удаление очень длинных токенов
    def _token_fix(m: re.Match) -> str:
        tok = m.group(0)
        if len(tok) >= hard_drop_len:
            return ""  # удалить
        return tok[:soft_trunc_len] + "…"
    return _LONG_TOKEN_RX.sub(_token_fix, line)

def _repeat_char_shrink(s: str) -> str:
    return _REPEAT_CHAR_RX.sub(lambda m: m.group(1) * 3, s)

# === обновлённая сборка текста диалога ===
def build_dialog_text_block(df_one_dialog: pd.DataFrame) -> str:
    spk_col = detect_speaker_col(df_one_dialog)
    parts: List[str] = []
    for _, r in df_one_dialog.iterrows():
        speaker = _as_str(r.get(spk_col, "")) if spk_col else ""
        text = _as_str(r.get(TEXT_COL, ""))
        text = re.sub(r"\s+\n", "\n", text).strip()
        # 1) схлопываем галлюцинации слов (3+ подряд -> 2)
        text = _collapse_repeats(text, keep=2)
        # 2) чистим «длинный мусор»: aaaaaa, worthworth..., сверх-длинные токены
        text = _shrink_long_noise(text, soft_trunc_len=40, hard_drop_len=80)
        # аккуратный пробел после двоеточия уже в формате
        parts.append(f"{speaker}: {text}" if speaker else text)

    big = "\n".join(parts).strip()
    if len(big) > EXCEL_MAX_CELL:
        big = big[:EXCEL_MAX_CELL - 20] + "\n...[TRUNCATED]"
    return big

def extract_phrases(series: pd.Series) -> List[str]:
    out: List[str] = []
    for v in series.dropna().astype(str):
        s = v.strip()
        if not s or s.lower() in {"nan", "none", '""'}:
            continue
        for p in _SPLIT_RE.split(s):
            p2 = p.strip(_TRIM).lower().replace("ё", "е")
            p2 = re.sub(r"\s+", " ", p2).strip()
            if p2:
                out.append(p2)
    return out

def aggregate_criterion_values_sales_only(sub: pd.DataFrame, colname: str) -> str:
    """Агрегируем значения критерия ТОЛЬКО по строкам SALES; уникальные, через запятую."""
    spk_col = detect_speaker_col(sub)
    if spk_col is None or colname not in sub.columns:
        return ""
    sub_sales = sub[sub[spk_col].astype(str) == SALES_VALUE]
    if sub_sales.empty:
        return ""
    vals = []
    for v in sub_sales[colname].dropna():
        if isinstance(v, (list, tuple, set)):
            items = list(v)
        else:
            s = str(v)
            items = _SPLIT_RE.split(s) if _SPLIT_RE.search(s) else [s]
        for it in items:
            it2 = str(it).strip(_TRIM)
            it2 = re.sub(r"\s+", " ", it2)
            if it2:
                vals.append(it2)
    uniq = list(dict.fromkeys(vals))
    return ", ".join(uniq)

def estimate_merged_row_height(text: str, total_chars_width: int, line_height_pt: float = 14.0) -> float:
    """
    Приблизительно оценивает высоту строки (pt) для объединённой ячейки с переносами.
    total_chars_width — сумма ширин колонок (в символах) в merge-диапазоне.
    """
    if not text:
        return line_height_pt
    lines = 0
    for raw_line in text.split("\n"):
        seg = raw_line.strip()
        if not seg:
            lines += 1
            continue
        wrapped = textwrap.wrap(seg, width=max(8, total_chars_width))
        lines += max(1, len(wrapped))
    return max(line_height_pt, lines * line_height_pt * 1.05)  # небольшой запас


# ---------- Основная функция отчёта ----------

def make_report(
    df: pd.DataFrame,
    out_path: str = "dialogs_report.xlsx",
    top_n: int = 20,
) -> str:
    if df.empty:
        raise ValueError("Пустой DataFrame — нечего выгружать.")

    df = df.copy()
    df.columns = make_unique_columns(df.columns)

    # найти реальные имена ключевых колонок и переименовать в «стандарт»
    def pick_col(name: str) -> Optional[str]:
        if name in df.columns:
            return name
        for c in df.columns:
            if c == name or c.startswith(name + "."):
                return c
        return None

    _DIALOG = pick_col(DIALOG_COL) or DIALOG_COL
    _TEXT   = pick_col(TEXT_COL)   or TEXT_COL
    _THEME  = pick_col(THEME_COL)  or THEME_COL
    _SPEAK  = pick_col(SPEAKER_COL) or pick_col(ALT_SPEAKER_COL)
    _ROWNUM = pick_col(ROWNUM_COL)
    _START, _END = pick_col(START_COL), pick_col(END_COL)
    _FILE, _STATUS, _DUR = pick_col(FILE_COL), pick_col(STATUS_COL), pick_col(DUR_COL)
    _LOUD = pick_col(LOUD_COL)

    ren = {}
    if _TEXT   and _TEXT   != TEXT_COL:   ren[_TEXT]   = TEXT_COL
    if _THEME  and _THEME  != THEME_COL:  ren[_THEME]  = THEME_COL
    if _SPEAK  and _SPEAK  != SPEAKER_COL: ren[_SPEAK] = SPEAKER_COL
    if _ROWNUM and _ROWNUM != ROWNUM_COL: ren[_ROWNUM] = ROWNUM_COL
    if _START  and _START  != START_COL:  ren[_START]  = START_COL
    if _END    and _END    != END_COL:    ren[_END]    = END_COL
    if _FILE   and _FILE   != FILE_COL:   ren[_FILE]   = FILE_COL
    if _STATUS and _STATUS != STATUS_COL: ren[_STATUS] = STATUS_COL
    if _DUR    and _DUR    != DUR_COL:    ren[_DUR]    = DUR_COL
    if _LOUD   and _LOUD   != LOUD_COL:   ren[_LOUD]   = LOUD_COL
    if ren:
        df = df.rename(columns=ren)

    if _DIALOG not in df.columns:
        raise ValueError(f"Нет колонки '{DIALOG_COL}' (идентификатор диалога).")

    # сортировка
    df[_DIALOG] = df[_DIALOG].astype(str)
    if ROWNUM_COL in df.columns:
        df = df.sort_values([_DIALOG, ROWNUM_COL], na_position="last")
    elif START_COL in df.columns and END_COL in df.columns:
        df = df.sort_values([_DIALOG, START_COL, END_COL], na_position="last")
    else:
        df = df.sort_values([_DIALOG], na_position="last")

    # ---- Подготовка громкости (по SALES) ----
    loud_df = pd.DataFrame()
    loud_threshold = None
    if (LOUD_COL in df.columns) and (SPEAKER_COL in df.columns):
        sales_rows = df[df[SPEAKER_COL].astype(str) == SALES_VALUE].copy()
        sales_rows["__loud"] = pd.to_numeric(sales_rows[LOUD_COL], errors="coerce")
        if sales_rows["__loud"].notna().sum() > 0:
            # порог = 95-й перцентиль по SALES
            loud_threshold = float(sales_rows["__loud"].quantile(0.95))
            loud_df = sales_rows.loc[sales_rows["__loud"] >= loud_threshold, [
                _DIALOG, SPEAKER_COL, START_COL, END_COL, LOUD_COL, TEXT_COL
            ]].copy().rename(columns={_DIALOG: "dialog_id"})
            loud_df = loud_df.sort_values("__loud", ascending=False)
            loud_df["row_text_short"] = loud_df[TEXT_COL].astype(str).str.replace(r"\s+", " ", regex=True).str.slice(0, 200)

    # ---- Агрегация на уровень диалога (включая критерии по SALES) ----
    present_criteria = [c for c, _ in CRITERIA_LABELS if c in df.columns]
    rows = []
    for did, sub in df.groupby(_DIALOG, dropna=False):
        themes = sub.get(THEME_COL, pd.Series([], dtype=str)).astype(str).map(str.strip)
        themes = [t for t in themes if t and t.lower() not in {"nan", "none", "null"}]
        theme_joined = ", ".join(dict.fromkeys(themes)) if themes else ""
        theme_class = classify_theme_by_rules(theme_joined)

        duration = sub.get(DUR_COL).iloc[0] if DUR_COL in sub.columns and len(sub) else np.nan
        if (pd.isna(duration) or duration == "") and START_COL in sub.columns and END_COL in sub.columns:
            try:
                s = pd.to_datetime(sub[START_COL], errors="coerce")
                e = pd.to_datetime(sub[END_COL], errors="coerce")
                if s.notna().any() and e.notna().any():
                    duration = float((e.max() - s.min()).total_seconds())
            except Exception:
                pass

        # громкость по диалогу (Sales)
        loud_cnt = 0
        max_loud = np.nan
        if not loud_df.empty:
            sel = loud_df[loud_df["dialog_id"].astype(str) == str(did)]
            loud_cnt = int(len(sel))
            if len(sel):
                max_loud = float(sel["__loud"].max())

        base = dict(
            dialog_id=did,
            theme_class=theme_class,
            theme=theme_joined,
            file_name=_as_str(sub.get(FILE_COL, pd.Series([""])).iloc[0]) if FILE_COL in sub.columns else "",
            status=_as_str(sub.get(STATUS_COL, pd.Series([""])).iloc[0]) if STATUS_COL in sub.columns else "",
            duration_sec=duration,
            rows_count=int(len(sub)),
            dialog_text=build_dialog_text_block(sub),
            loud_peaks=loud_cnt,
            loud_max=max_loud,
        )

        for c in present_criteria:
            base[c] = aggregate_criterion_values_sales_only(sub, c)

        rows.append(base)

    summary = pd.DataFrame(rows)

    # ---- Метрики ----
    total_dialogs = summary["dialog_id"].nunique()
    total_seconds = float(pd.to_numeric(summary["duration_sec"], errors="coerce").fillna(0).sum())
    total_hours = total_seconds / 3600.0

    theme_counts = (
        summary["theme_class"].value_counts(dropna=False)
        .rename_axis("Тема").reset_index(name="Диалогов")
        .sort_values("Диалогов", ascending=False)
    )

    # ---- Пишем Excel ----
    wb = xlsxwriter.Workbook(out_path)

    fmt_header = wb.add_format({"bold": True, "bg_color": "#F2F2F2"})
    fmt_wrap   = wb.add_format({"text_wrap": True, "valign": "top"})
    fmt_norm   = wb.add_format({"valign": "top"})
    fmt_bold   = wb.add_format({"bold": True})
    fmt_num    = wb.add_format({"num_format": "0"})
    fmt_dur    = wb.add_format({"num_format": "0.0"})
    fmt_hlink  = wb.add_format({"font_color": "blue", "underline": 1})

    ws = wb.add_worksheet("Summary")
    ws.set_default_row(13)

    # ---- Сводка в столбик (сверху) ----
    r = 0
    ws.write(r, 0, "Сводка", fmt_header); r += 1
    ws.write(r, 0, "Диалогов:", fmt_bold);     ws.write_number(r, 1, int(total_dialogs), fmt_num); r += 1
    ws.write(r, 0, "Суммарно, часов:", fmt_bold); ws.write_number(r, 1, round(total_hours, 1), fmt_dur); r += 1
    if loud_threshold is not None:
        ws.write(r, 0, "Порог «громко» (95-й перц.),", fmt_bold); ws.write_number(r, 1, loud_threshold, fmt_dur); r += 1

    # Таблица тем под сводкой (вертикально)
    ws.write(r, 0, "Диалоги по темам", fmt_header); r += 1
    ws.write(r, 0, "Тема", fmt_bold); ws.write(r, 1, "Кол-во", fmt_bold); r += 1
    for rec in theme_counts.itertuples(index=False):
        ws.write(r, 0, rec.Тема or "", fmt_norm)
        ws.write_number(r, 1, int(rec.Диалогов), fmt_num)
        r += 1

    start_row_table = r + 2  # отступ до основной таблицы

    # ---- Заголовок основной таблицы ----
    base_headers = [
        ("dialog_id",      "dialog_id"),
        ("theme_class",    "Класс темы"),
        ("theme",          "Тема"),
        ("file_name",      "Файл"),
        ("status",         "Статус"),
        ("duration_sec",   "Длит., с"),
        ("rows_count",     "Строк"),
        ("loud_peaks",     "Громкие реплики (шт)"),
        ("loud_max",       "Макс. громкость"),
    ]
    # критерии (русские названия)
    crit_headers = [(c, ru) for c, ru in CRITERIA_LABELS if c in summary.columns]

    headers = base_headers + crit_headers

    for c, (_key, title) in enumerate(headers):
        ws.write(start_row_table, c, title, fmt_header)

    # запишем строки (одна строка на диалог + скрытая детальная)
    row = start_row_table + 1
    # Ширины колонок (в символах)
    col_widths = {
        "dialog_id": 36, "Класс темы": 12, "Тема": 28, "Файл": 28, "Статус": 10,
        "Длит., с": 10, "Строк": 8, "Громкие реплики (шт)": 8, "Макс. громкость": 12,
        "Превью": 42
    }
    # Применим базовые ширины
    for idx, (_key, title) in enumerate(headers):
        if title in col_widths:
            ws.set_column(idx, idx, col_widths[title], fmt_wrap if title in ("Тема","Превью") else fmt_norm)
        else:
            ws.set_column(idx, idx, 18, fmt_wrap)

    # вычислим суммарную ширину объединённого диапазона с полным текстом (со 2-го столбца до конца)
    # (в символах для estimate_merged_row_height)
    total_chars_width = 0
    for idx in range(1, len(headers)):
        # берём заданную ширину или 18 по умолчанию
        title = headers[idx][1]
        total_chars_width += col_widths.get(title, 18)

    for _, rsum in summary.iterrows():

        values = [
            rsum["dialog_id"],
            rsum["theme_class"],
            rsum["theme"],
            rsum["file_name"],
            rsum["status"],
            float(rsum["duration_sec"]) if pd.notna(rsum["duration_sec"]) else 0.0,
            int(rsum["rows_count"]),
            int(rsum.get("loud_peaks", 0)),
            (float(rsum.get("loud_max")) if pd.notna(rsum.get("loud_max")) else np.nan),
        ]

        # базовые поля
        for c, v in enumerate(values):
            title = headers[c][1]
            if title in ("Длит., с", "Макс. громкость"):
                if (isinstance(v, float) or isinstance(v, (np.floating,))) and not math.isnan(v):
                    ws.write_number(row, c, float(v), fmt_dur)
                else:
                    ws.write(row, c, "", fmt_norm)
            elif title in ("Строк","Громкие реплики (шт)"):
                ws.write_number(row, c, int(v), fmt_num)
            else:
                ws.write(row, c, v, fmt_wrap if title in ("Тема","Превью") else fmt_norm)

        # ссылка для разворота
        ws.write_url(row, len(values), f"internal:'Summary'!A{row+2}", fmt_hlink, string="↓")

        # критерии
        col_idx = len(base_headers)
        for key, _ru in crit_headers:
            ws.write(row, col_idx, _as_str(rsum.get(key, "")), fmt_wrap)
            col_idx += 1

        # компактная высота основной строки
        ws.set_row(row, 12)

        # скрытая строка «Полный текст» с высотой по содержимому
        detail_row = row + 1
        ws.write(detail_row, 0, "Полный текст:", fmt_bold)
        # высота (pt) оценивается из суммарной ширины merge-диапазона
        est_h = estimate_merged_row_height(rsum["dialog_text"], total_chars_width, line_height_pt=14.0)
        ws.merge_range(detail_row, 1, detail_row, len(headers) - 1, rsum["dialog_text"], fmt_wrap)
        ws.set_row(detail_row, est_h, None, {"hidden": True, "level": 1})

        row += 2

    ws.freeze_panes(start_row_table + 1, 1)
    ws.autofilter(start_row_table, 0, row - 1, len(headers) - 1)
    ws.outline_settings(True, False, True, False)

    # ---- Листы лексики (таблица + график) ----
    def build_lexicon_table(colname: str) -> pd.DataFrame:
        if colname not in df.columns:  # нет такой колонки — пропускаем
            return pd.DataFrame(columns=["phrase","freq"])
        spk_col = detect_speaker_col(df)
        ser = df[colname]
        if spk_col is not None:
            ser = ser[df[spk_col].astype(str) == SALES_VALUE]  # только SALES
        from collections import Counter
        return pd.DataFrame(Counter(extract_phrases(ser)).most_common(top_n), columns=["phrase","freq"])

    for colname, nice in LEXICON_COLS:
        tbl = build_lexicon_table(colname)
        if tbl.empty:
            continue
        sheet_name = nice[:31]
        wsx = wb.add_worksheet(sheet_name)
        wsx.set_default_row(13)
        wsx.write(0, 0, "Слово/фраза", fmt_header)
        wsx.write(0, 1, "Частота", fmt_header)
        for i, rr in enumerate(tbl.itertuples(index=False), start=1):
            wsx.write(i, 0, rr.phrase, fmt_norm)
            wsx.write_number(i, 1, int(rr.freq), fmt_num)
        wsx.set_column(0, 0, 50, fmt_norm)
        wsx.set_column(1, 1, 12, fmt_num)
        wsx.freeze_panes(1, 0)
        last = min(10, len(tbl))
        if last > 0:
            ch = wb.add_chart({"type": "column"})
            ch.add_series({
                "name": f"Топ-{min(10, top_n)}: {nice}",
                "categories": [sheet_name, 1, 0, last, 0],
                "values":     [sheet_name, 1, 1, last, 1],
                "data_labels": {"value": True},
            })
            ch.set_title({"name": f"{nice}: Топ-{min(10, top_n)}"})
            ch.set_y_axis({"major_gridlines": {"visible": False}})
            wsx.insert_chart(1, 3, ch, {"x_scale": 1.2, "y_scale": 1.05})

    # ---- Лист «Громкость» ----
    if not loud_df.empty:
        wsl = wb.add_worksheet("Громкость")
        wsl.set_default_row(13)
        cols = ["dialog_id", SPEAKER_COL, START_COL, END_COL, LOUD_COL, "row_text_short"]
        headers_ru = ["dialog_id", "Спикер", "Начало", "Конец", "Громкость", "Текст (сокр.)"]
        for j, h in enumerate(headers_ru):
            wsl.write(0, j, h, fmt_header)
        for i, (_, rr_) in enumerate(loud_df.iterrows(), start=1):
            wsl.write(i, 0, _as_str(rr_["dialog_id"]), fmt_norm)
            wsl.write(i, 1, _as_str(rr_.get(SPEAKER_COL, "")), fmt_norm)
            wsl.write(i, 2, _as_str(rr_.get(START_COL, "")), fmt_norm)
            wsl.write(i, 3, _as_str(rr_.get(END_COL, "")), fmt_norm)
            val = rr_.get("__loud", rr_.get(LOUD_COL))
            try:
                wsl.write_number(i, 4, float(val), fmt_dur)
            except Exception:
                wsl.write(i, 4, _as_str(val), fmt_norm)
            wsl.write(i, 5, _as_str(rr_.get("row_text_short", "")), fmt_wrap)
        # ширины
        wsl.set_column(0, 0, 36, fmt_norm)
        wsl.set_column(1, 1, 10, fmt_norm)
        wsl.set_column(2, 3, 18, fmt_norm)
        wsl.set_column(4, 4, 12, fmt_dur)
        wsl.set_column(5, 5, 60, fmt_wrap)
        wsl.freeze_panes(1, 0)

    wb.close()
    return out_path


# ---------- CLI ----------

if __name__ == "__main__":
    repo = AudioDialogRepository()
    df = repo.get_all_for_report()
    out = make_report(df, "dialogs_report.xlsx")
    print(f"Готово: {out}")
