# -----------------------------------------------
# Excel-отчёт по диалогам:
#  - Сводка (в столбик) над основной таблицей
#  - Одна строка на диалог + скрываемая строка с полным текстом (высота авто-рассчитана)
#  - Критерии (только по Sales) в основной таблице, с русскими названиями
#  - Листы с графиками по лексике (кроме «Уменьшительные», «Междометия»)
#  - Лист "Громкость" с потенциально громкими репликами (Sales)
#  - Лист "Критерии%" с долей диалогов по каждому критерию
#  - Новые флаги громкости: локальные всплески (внутри диалога) и глобальные крайности (по всем диалогам)
#  - В Summary добавлены «Имя SALES» и «Имя CLIENT», а telling_name_phrases/found_name скрыты
# -----------------------------------------------

import re
import textwrap
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import xlsxwriter

try:
    from core.repository.audio_dialog_repository import AudioDialogRepository
except Exception:
    AudioDialogRepository = None

THEME_COL = "theme"
TEXT_COL = "row_text"
DIALOG_COL = "audio_dialog_fk_id"
SPEAKER_COL = "speaker_id"
ALT_SPEAKER_COL = "detected_speaker_id"
ROWNUM_COL = "row_num"
START_COL, END_COL = "start", "end"
FILE_COL, DUR_COL = "file_name", "duration"
LOUD_COL = "mean_loudness"

SALES_VALUE = "SALES"
CLIENT_VALUE = "CLIENT"
SPEAKER_1 = "Speaker_1"
SPEAKER_2 = "Speaker_2"
EXCEL_MAX_CELL = 32767

LOUD_SCALE = 1000.0

# ---- Критерии ----
CRITERIA_LABELS: List[Tuple[str, str]] = [
    ("greeting_phrase",          "Приветствие"),
    ("telling_name_phrases",     "Назвал свое имя"),
    ("found_name",               "Имя"),
    ("ongoing_sale",             "Доп. продажи"),
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

# --- очистка диалогового текста ---

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

_REPEAT_CHAR_RX = re.compile(r'(?iu)(\w)\1{5,}')
_REPEAT_CHUNK_RX = re.compile(r'(?iu)\b([a-zа-яё0-9]{2,10})\1{3,}\b')
_LONG_TOKEN_RX = re.compile(r'(?iu)\b\w{41,}\b')

def _repeat_char_shrink(s: str) -> str:
    return _REPEAT_CHAR_RX.sub(lambda m: m.group(1) * 3, s)

def _shrink_long_noise(line: str, soft_trunc_len: int = 40, hard_drop_len: int = 80) -> str:
    line = _repeat_char_shrink(line)
    line = _REPEAT_CHUNK_RX.sub(lambda m: m.group(1) * 2, line)
    def _token_fix(m: re.Match) -> str:
        tok = m.group(0)
        if len(tok) >= hard_drop_len:
            return ""
        return tok[:soft_trunc_len] + "…"
    return _LONG_TOKEN_RX.sub(_token_fix, line)

def build_dialog_text_block(df_one_dialog: pd.DataFrame) -> str:
    spk_col = detect_speaker_col(df_one_dialog)
    parts: List[str] = []
    for _, r in df_one_dialog.iterrows():
        speaker = _as_str(r.get(spk_col, "")) if spk_col else ""
        text = _as_str(r.get(TEXT_COL, ""))
        text = re.sub(r"\s+\n", "\n", text).strip()
        text = _collapse_repeats(text, keep=2)
        text = _shrink_long_noise(text, soft_trunc_len=40, hard_drop_len=80)
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

def _first_token(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = name.strip().strip(_TRIM)
    if not name:
        return ""

    token = re.split(r"[\s,;/\\|]+", name, maxsplit=1)[0]
    return token.strip(_TRIM)

def _cap(s: str) -> str:
    return s.capitalize() if isinstance(s, str) and s else s

def extract_names_from_sales(sub: pd.DataFrame) -> Tuple[str, str]:
    """
    Возвращает (sales_name, client_name) по правилам:
      - Имя SALES: строка SALES с заполненными found_name И telling_name_phrases;
                   если нет — первое found_name у SALES, встреченное при row_num < 3.
      - Имя CLIENT: любое found_name у SALES, где telling_name_phrases пусто и имя != Имя SALES.
    """
    spk = detect_speaker_col(sub)
    if spk is None:
        return "", ""

    rows = sub[sub[spk].astype(str).isin([SALES_VALUE, SPEAKER_1, SPEAKER_2])].copy()
    if rows.empty:
        return "", ""

    # сортируем по row_num (если есть), иначе по start
    if ROWNUM_COL in rows.columns:
        rows = rows.sort_values(ROWNUM_COL, kind="mergesort")
    elif START_COL in rows.columns:
        rows = rows.sort_values(START_COL, kind="mergesort")

    sales_name = ""
    first_found_early = ""
    client_candidates: List[str] = []

    for _, r in rows.iterrows():
        fn = _first_token(_as_str(r.get("found_name", "")))
        tp = _as_str(r.get("telling_name_phrases", ""))
        rn = r.get(ROWNUM_COL, np.nan)

        # Кейс 1: одновременно есть found_name и telling_name_phrases -> Имя SALES
        if not sales_name and fn and tp:
            sales_name = fn

        # Кейс 2: первый found_name при row_num < 3
        if not first_found_early and fn:
            try:
                if pd.notna(rn) and float(rn) < 3:
                    first_found_early = fn
            except Exception:
                pass

        # Кандидаты в Имя CLIENT: found_name задан, telling_name_phrases пуст
        if fn and not tp:
            client_candidates.append(fn)

    if not sales_name and first_found_early:
        sales_name = first_found_early

    client_name = ""
    if client_candidates:
        for cand in client_candidates:
            if not sales_name or cand.lower() != sales_name.lower():
                client_name = cand
                break

    return _cap(sales_name), _cap(client_name)

def aggregate_criterion_values_sales_only(sub: pd.DataFrame, colname: str) -> str:
    spk_col = detect_speaker_col(sub)
    if spk_col is None or colname not in sub.columns:
        return ""
    sub_sales = sub[sub[spk_col].astype(str).isin([SALES_VALUE, SPEAKER_1, SPEAKER_2])]
    if sub_sales.empty:
        return ""
    vals: List[str] = []
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
    return max(line_height_pt, lines * line_height_pt * 1.05)

# ---------- Громкость: вспомогательные ----------

def _series_loud(df: pd.DataFrame, speaker_value: str) -> pd.Series:
    spk = detect_speaker_col(df)
    if spk is None or LOUD_COL not in df.columns:
        return pd.Series([], dtype=float)
    ser = pd.to_numeric(df.loc[df[spk].astype(str) == speaker_value, LOUD_COL], errors="coerce")
    return ser.dropna()

def _concat_nonempty(parts: List[pd.Series]) -> pd.Series:
    parts = [s for s in parts if s is not None and not s.empty]
    if not parts:
        return pd.Series([], dtype=float)
    return pd.concat(parts, ignore_index=True)

def _sales_mean_for_dialog(sub: pd.DataFrame) -> float:
    ser = _concat_nonempty([
        _series_loud(sub, SALES_VALUE),
        _series_loud(sub, SPEAKER_1),
        _series_loud(sub, SPEAKER_2),
    ])
    return float(ser.mean()) if not ser.empty else np.nan

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
    _FILE, _DUR = pick_col(FILE_COL), pick_col(DUR_COL)
    _LOUD = pick_col(LOUD_COL)

    ren = {}
    if _TEXT   and _TEXT   != TEXT_COL:    ren[_TEXT]   = TEXT_COL
    if _THEME  and _THEME  != THEME_COL:   ren[_THEME]  = THEME_COL
    if _SPEAK  and _SPEAK  != SPEAKER_COL: ren[_SPEAK]  = SPEAKER_COL
    if _ROWNUM and _ROWNUM != ROWNUM_COL:  ren[_ROWNUM] = ROWNUM_COL
    if _START  and _START  != START_COL:   ren[_START]  = START_COL
    if _END    and _END    != END_COL:     ren[_END]    = END_COL
    if _FILE   and _FILE   != FILE_COL:    ren[_FILE]   = FILE_COL
    if _DUR    and _DUR    != DUR_COL:     ren[_DUR]    = DUR_COL
    if _LOUD   and _LOUD   != LOUD_COL:    ren[_LOUD]   = LOUD_COL
    if ren:
        df = df.rename(columns=ren)

    if _DIALOG not in df.columns:
        raise ValueError(f"Нет колонки '{DIALOG_COL}' (идентификатор диалога).")

    df[_DIALOG] = df[_DIALOG].astype(str)
    if ROWNUM_COL in df.columns:
        df = df.sort_values([_DIALOG, ROWNUM_COL], na_position="last")
    elif START_COL in df.columns and END_COL in df.columns:
        df = df.sort_values([_DIALOG, START_COL, END_COL], na_position="last")
    else:
        df = df.sort_values([_DIALOG], na_position="last")

    # ---- Глобальные пороги по среднему SALES (для «тише/громче 95% всех») ----
    sales_means_list: List[float] = []
    for _, sub_d in df.groupby(_DIALOG, dropna=False):
        sales_means_list.append(_sales_mean_for_dialog(sub_d))
    sales_means_per_dialog = pd.Series(sales_means_list, dtype=float).dropna()
    global_sales_q05 = float(sales_means_per_dialog.quantile(0.05)) if len(sales_means_per_dialog) else np.nan
    global_sales_q95 = float(sales_means_per_dialog.quantile(0.95)) if len(sales_means_per_dialog) else np.nan

    # ---- Агрегация на диалог (+ имена) ----
    present_criteria = [c for c, _ in CRITERIA_LABELS if c in df.columns]
    rows: List[Dict[str, object]] = []

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

        # глобальные флаги по среднему SALES в данном диалоге
        sales_ser = _concat_nonempty([
            _series_loud(sub, SALES_VALUE),
            _series_loud(sub, SPEAKER_1),
            _series_loud(sub, SPEAKER_2),
        ])
        sales_mean_this_dialog = float(sales_ser.mean()) if sales_ser.size > 0 else np.nan
        sales_quieter_95_global = int(pd.notna(sales_mean_this_dialog) and pd.notna(global_sales_q05)
                                      and sales_mean_this_dialog <= global_sales_q05)
        sales_louder_95_global  = int(pd.notna(sales_mean_this_dialog) and pd.notna(global_sales_q95)
                                      and sales_mean_this_dialog >= global_sales_q95)

        # Имена
        name_sales, name_client = extract_names_from_sales(sub)

        base: Dict[str, object] = dict(
            dialog_id=did,
            theme_class=theme_class,
            theme=theme_joined,
            file_name=_as_str(sub.get(FILE_COL, pd.Series([""])).iloc[0]) if FILE_COL in sub.columns else "",
            duration_sec=duration,
            rows_count=int(len(sub)),
            dialog_text=build_dialog_text_block(sub),
            sales_quieter_95_global=sales_quieter_95_global,
            sales_louder_95_global=sales_louder_95_global,
            name_sales=name_sales,
            name_client=name_client,
        )

        # критерии (как раньше), но в Summary НЕ будем выводить found_name/telling_name_phrases
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
    fmt_int    = wb.add_format({"num_format": "0"})
    fmt_pct    = wb.add_format({"num_format": "0.0%"})

    # ===== Summary =====
    ws = wb.add_worksheet("Summary")
    ws.set_default_row(13)

    r = 0
    ws.write(r, 0, "Сводка", fmt_header); r += 1
    ws.write(r, 0, "Диалогов:", fmt_bold);        ws.write_number(r, 1, int(total_dialogs), fmt_num); r += 1
    ws.write(r, 0, "Суммарно, часов:", fmt_bold); ws.write_number(r, 1, round(total_hours, 1), fmt_dur); r += 1

    ws.write(r, 0, "Диалоги по темам", fmt_header); r += 1
    ws.write(r, 0, "Тема", fmt_bold); ws.write(r, 1, "Кол-во", fmt_bold); r += 1
    for rec in theme_counts.itertuples(index=False):
        ws.write(r, 0, rec.Тема or "", fmt_norm)
        ws.write_number(r, 1, int(rec.Диалогов), fmt_num)
        r += 1

    start_row_table = r + 2

    # ---- Заголовок основной таблицы ----
    # УБРАЛИ telling_name_phrases и found_name, ДОБАВИЛИ Имя SALES / Имя CLIENT
    base_headers = [
        ("dialog_id",      "Идентификатор диалога"),
        ("theme_class",    "Класс темы"),
        ("theme",          "Тема"),
        ("file_name",      "Имя файла"),
        ("name_sales",     "Имя SALES"),
        ("name_client",    "Обращение к клиенту"),
        ("duration_sec",   "Длит., с"),
        ("rows_count",     "Строк"),
        ("sales_quieter_95_global", "SALES тише 95% всех"),
        ("sales_louder_95_global",  "SALES громче 95% всех"),
        ("_expand",        "▼"),
    ]
    # критерии выводим, НО исключаем found_name/telling_name_phrases из Summary
    present_for_summary = [
        (c, ru) for c, ru in CRITERIA_LABELS
        if c in summary.columns and c not in ("found_name", "telling_name_phrases")
    ]
    headers = base_headers + present_for_summary

    for c, (_key, title) in enumerate(headers):
        ws.write(start_row_table, c, title, fmt_header)

    col_widths = {
        "dialog_id": 36, "Класс темы": 12, "Тема": 28, "Файл": 28,
        "Имя SALES": 16, "Имя CLIENT": 16,
        "Длит., с": 10, "Строк": 8,
        "SALES тише 95% всех": 18, "SALES громче 95% всех": 20,
        "▼": 4
    }
    for idx, (_key, title) in enumerate(headers):
        ws.set_column(idx, idx, col_widths.get(title, 18), fmt_wrap if title in ("Тема",) else fmt_norm)

    total_chars_width = sum(col_widths.get(headers[idx][1], 18) for idx in range(1, len(headers)))

    # ---- строки отчёта ----
    row = start_row_table + 1
    for _, rsum in summary.iterrows():
        values = [
            rsum["dialog_id"],
            rsum["theme_class"],
            rsum["theme"],
            rsum["file_name"],
            _as_str(rsum.get("name_sales", "")),
            _as_str(rsum.get("name_client", "")),
            float(rsum["duration_sec"]) if pd.notna(rsum["duration_sec"]) else 0.0,
            int(rsum["rows_count"]),
            int(rsum.get("sales_quieter_95_global", 0)),
            int(rsum.get("sales_louder_95_global", 0)),
        ]

        for c, v in enumerate(values):
            title = headers[c][1]
            if title == "Длит., с":
                ws.write_number(row, c, float(v), fmt_dur)
            elif title in ("Строк", "SALES тише 95% всех", "SALES громче 95% всех"):
                ws.write_number(row, c, int(v), fmt_int)
            else:
                ws.write(row, c, v, fmt_wrap if title in ("Тема",) else fmt_norm)

        # колонка «▼» — ссылка для разворота
        ws.write_url(row, len(base_headers)-1, f"internal:'Summary'!A{row+2}", fmt_bold, string="↓")

        # критерии (без found_name/telling_name_phrases)
        col_idx = len(base_headers)
        for key, _ru in present_for_summary:
            ws.write(row, col_idx, _as_str(rsum.get(key, "")), fmt_wrap)
            col_idx += 1

        # компактная высота основной строки
        ws.set_row(row, 12)

        # скрытая строка «Полный текст» с высотой по содержимому
        detail_row = row + 1
        ws.write(detail_row, 0, "Полный текст:", fmt_bold)
        est_h = estimate_merged_row_height(rsum["dialog_text"], total_chars_width, line_height_pt=14.0)
        ws.merge_range(detail_row, 1, detail_row, len(headers) - 1, rsum["dialog_text"], fmt_wrap)
        ws.set_row(detail_row, est_h, None, {"hidden": True, "level": 1})

        row += 2

    ws.freeze_panes(start_row_table + 1, 1)
    ws.autofilter(start_row_table, 0, row - 1, len(headers) - 1)
    ws.outline_settings(True, False, True, False)

    # ===== Критерии% =====
    crit_sheet = wb.add_worksheet("Критерии%")
    crit_sheet.set_default_row(13)

    present_criteria = [c for c, _ in CRITERIA_LABELS if c in summary.columns]
    crit_title = dict(CRITERIA_LABELS)
    total_dialogs_safe = max(1, int(summary["dialog_id"].nunique()))

    stat_rows = []
    for c in present_criteria:
        mask = summary[c].astype(str).str.strip().ne("")
        cnt  = int(mask.sum())
        pct  = cnt / total_dialogs_safe
        stat_rows.append({"Критерий": crit_title.get(c, c), "Диалогов": cnt, "Доля": pct})

    crit_df = pd.DataFrame(stat_rows).sort_values("Доля", ascending=False)

    crit_sheet.write(0, 0, "Критерий", fmt_header)
    crit_sheet.write(0, 1, "Диалогов", fmt_header)
    crit_sheet.write(0, 2, "Доля", fmt_header)

    for i, rec in enumerate(crit_df.itertuples(index=False), start=1):
        crit_sheet.write(i, 0, rec.Критерий, fmt_norm)
        crit_sheet.write_number(i, 1, int(rec.Диалогов), fmt_num)
        crit_sheet.write_number(i, 2, float(rec.Доля), fmt_pct)

    crit_sheet.set_column(0, 0, 40, fmt_norm)
    crit_sheet.set_column(1, 1, 12, fmt_num)
    crit_sheet.set_column(2, 2, 12, fmt_pct)
    crit_sheet.freeze_panes(1, 0)

    # ===== Лексика =====
    def build_lexicon_table(colname: str) -> pd.DataFrame:
        if colname not in df.columns:
            return pd.DataFrame(columns=["phrase","freq"])
        spk_col = detect_speaker_col(df)
        ser = df[colname]
        if spk_col is not None:
            ser = ser[df[spk_col].astype(str).isin([SALES_VALUE, SPEAKER_1, SPEAKER_2])]
        from collections import Counter
        return pd.DataFrame(Counter(extract_phrases(ser)).most_common(top_n), columns=["phrase","freq"])

    for colname, nice in LEXICON_COLS:
        tbl = build_lexicon_table(colname)
        if tbl.empty:
            continue
        sheet_name = nice[:31]
        wsx = wb.add_worksheet(sheet_name)
        wsx.set_default_row(13)
        hfmt = wb.add_format({"bold": True, "bg_color": "#F2F2F2"})
        nfmt = wb.add_format({"num_format": "0"})
        wfmt = wb.add_format({"valign": "top"})
        wsx.write(0, 0, "Слово/фраза", hfmt)
        wsx.write(0, 1, "Частота", hfmt)
        for i, rr in enumerate(tbl.itertuples(index=False), start=1):
            wsx.write(i, 0, rr.phrase, wfmt)
            wsx.write_number(i, 1, int(rr.freq), nfmt)
        wsx.set_column(0, 0, 50, wfmt)
        wsx.set_column(1, 1, 12, nfmt)
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

    wb.close()
    return out_path

# ---------- CLI (main не меняем) ----------
if __name__ == "__main__":
    repo = AudioDialogRepository()
    df = repo.get_all_for_report()
    out = make_report(df, "dialogs_report.xlsx")
    print(f"Готово: {out}")
