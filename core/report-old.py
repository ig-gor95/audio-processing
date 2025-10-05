# -----------------------------------------------
# Excel-отчёт по диалогам (новая схема LLM JSON, русификация)
# - Лист 1: "Сводка" — метрики, проценты, графики, корреляции
# - Лист 2: "Диалоги" — таблица по диалогам + скрываемый полный текст
# -----------------------------------------------

import re
import textwrap
from typing import Optional, List, Dict, Tuple
import json
from collections import Counter

import numpy as np
import pandas as pd
import xlsxwriter

try:
    from core.repository.audio_dialog_repository import AudioDialogRepository
except Exception:
    AudioDialogRepository = None

# ===== Базовые колоноки входного фрейма =====
THEME_COL = "theme"
TEXT_COL = "row_text"
DIALOG_COL = "audio_dialog_fk_id"
SPEAKER_COL = "speaker_id"
ALT_SPEAKER_COL = "detected_speaker_id"
ROWNUM_COL = "row_num"
START_COL, END_COL = "start", "end"
FILE_COL, DUR_COL = "file_name", "duration"
LOUD_COL = "mean_loudness"
LLM_COL = "llm_data_short"

SALES_VALUE = "SALES"
CLIENT_VALUE = "CLIENT"
SPEAKER_1 = "Speaker_1"
SPEAKER_2 = "Speaker_2"
EXCEL_MAX_CELL = 32767

# ===== Русификация значений =====
MAP_AVAIL = {"in_stock": "в наличии", "backorder": "под заказ", "unknown": "неизвестно", "n/a": "н/д"}
MAP_SEASON = {"летняя": "летняя", "зимняя": "зимняя", "всесезонная": "всесезонная"}
MAP_OBJ_STATUS = {
    "ignored": "проигнорировано",
    "unresolved": "не решено",
    "partially_resolved": "частично решено",
    "resolved": "решено",
}
MAP_OBJ_TYPE = {
    "price": "цена",
    "availability": "наличие",
    "time_sla": "сроки/запись",
    "trust": "доверие",
    "warranty_claim": "гарантия",
    "fitment": "подбор/совместимость",
    "location_logistics": "локация/логистика",
    "process": "процесс/оформление",
}
MAP_OWNER = {"client": "клиент", "sales": "оператор"}

# ---- Критерии (как было, для вывода в "Диалоги") ----
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
    ("parasite_words",           "Слова-паразиты"),
    ("stop_words",               "Стоп-слова"),
    ("slang",                    "Сленг"),
    ("non_professional_phrases", "Непроф. фразы"),
    ("inappropriate_phrases",    "Неприемлемые"),
    ("swear_words",              "Мат"),
    ("order_type",               "Тип заказа/подбор"),
]

# --- парсинг фраз (для лексики — пока не используем, оставлено на будущее) ---
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

__REPEAT_CHAR_RX = re.compile(r'(?iu)(\w)\1{5,}')
_REPEAT_CHUNK_RX = re.compile(r'(?iu)\b([a-zа-яё0-9]{2,10})\1{3,}\b')
_LONG_TOKEN_RX = re.compile(r'(?iu)\b\w{41,}\b')

def _repeat_char_shrink(s: str) -> str:
    return __REPEAT_CHAR_RX.sub(lambda m: m.group(1) * 3, s)

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

def _first_token(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = name.strip().strip(_TRIM)
    if not name:
        return ""
    token = re.split(r"[\s,;/\\|]+", name, maxsplit=1)[0]
    return token.strip(_TRIM)

def _safe_json_loads(s):
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return None
    if isinstance(s, (dict, list, bool, int, float)):
        return s
    s = str(s).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    s2 = (s.replace("'", '"')
            .replace(" None", " null").replace(": None", ": null")
            .replace(" True", " true").replace(": True", ": true")
            .replace(" False", " false").replace(": False", ": false"))
    try:
        return json.loads(s2)
    except Exception:
        pass
    import re as _re
    s3 = _re.sub(r",\s*([}\]])", r"\1", s2)
    try:
        return json.loads(s3)
    except Exception:
        return None

def _nn(x):
    return "" if x is None else x

def _b2i(b):
    return 1 if b is True else 0

def _aslist(x):
    return x if isinstance(x, list) else []

def _j(x):
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return ""

# ===== Плоский парсер новой схемы + русификация =====
def _llm_flat(row_obj: dict) -> dict:
    if not isinstance(row_obj, dict):
        return {}

    _intent_allowed = {"покупка","уточнение","запись","гарантия"}
    _cat_allowed = {"Шины","Диски","АКБ","Масла/Жидкости","Услуги","Другое"}
    _avail_allowed = set(MAP_AVAIL.keys())
    _season_allowed = set(MAP_SEASON.keys())
    _obj_type_allowed = set(MAP_OBJ_TYPE.keys())
    _obj_status_allowed = set(MAP_OBJ_STATUS.keys())
    _owner_allowed = set(MAP_OWNER.keys())

    def _enum(x, allowed):
        x = _nn(x)
        return x if x in allowed else ""

    objection = row_obj.get("objection", {}) or {}
    handling = (objection.get("handling") or {}) if isinstance(objection, dict) else {}
    next_set = row_obj.get("next_set", {}) or {}
    add_on   = row_obj.get("add_on_sale", {}) or {}

    intent  = _enum(row_obj.get("intent"), _intent_allowed)
    cat     = _enum(row_obj.get("category"), _cat_allowed)
    catdet  = _nn(row_obj.get("category_detail"))
    tire_rim= row_obj.get("tire_rim")

    avail   = _enum(row_obj.get("availability"), _avail_allowed)
    season  = _enum(row_obj.get("season"), _season_allowed)

    obj_detected = bool(isinstance(objection, dict) and objection.get("detected") is True)
    obj_type     = _enum(objection.get("type") if isinstance(objection, dict) else "", _obj_type_allowed)
    obj_status   = _enum(objection.get("status") if isinstance(objection, dict) else "", _obj_status_allowed)
    obj_score    = (objection.get("score") if isinstance(objection, dict) else 0) or 0
    obj_ack      = bool(handling.get("acknowledge") if isinstance(handling, dict) else False)
    obj_sol      = bool(handling.get("solution") if isinstance(handling, dict) else False)
    obj_check    = bool(handling.get("check_close") if isinstance(handling, dict) else False)

    nxt_set   = bool(next_set.get("set") if isinstance(next_set, dict) else False)
    nxt_action= _nn(next_set.get("action") if isinstance(next_set, dict) else "")
    nxt_dt    = _nn(next_set.get("datetime") if isinstance(next_set, dict) else "")
    nxt_place = _nn(next_set.get("place") if isinstance(next_set, dict) else "")
    nxt_owner = _enum(next_set.get("owner") if isinstance(next_set, dict) else "", _owner_allowed)

    wrap_up  = bool(row_obj.get("wrap_up") is True)

    addon_offered  = bool(add_on.get("offered") if isinstance(add_on, dict) else False)
    addon_accepted = bool(add_on.get("accepted") if isinstance(add_on, dict) else False)
    addon_items    = _aslist(add_on.get("items") if isinstance(add_on, dict) else [])
    addon_evid     = _aslist(add_on.get("evidence_quotes") if isinstance(add_on, dict) else [])

    # русификация отображаемых строк
    out = {
        # исходники
        "llm_intent": intent,
        "llm_category": cat,
        "llm_category_detail": catdet,
        "llm_tire_rim": tire_rim if tire_rim in (None, "") else str(tire_rim),

        "llm_availability": avail,
        "llm_season": season,

        "llm_obj_detected": _b2i(obj_detected),
        "llm_obj_type": obj_type,
        "llm_obj_status": obj_status,
        "llm_obj_score": int(obj_score),
        "llm_obj_ack": _b2i(obj_ack),
        "llm_obj_sol": _b2i(obj_sol),
        "llm_obj_check": _b2i(obj_check),

        "llm_next_set": _b2i(nxt_set),
        "llm_next_action": nxt_action,
        "llm_next_datetime": nxt_dt,
        "llm_next_place": nxt_place,
        "llm_next_owner": nxt_owner,

        "llm_wrap_up": _b2i(wrap_up),
        "llm_addon_offered": _b2i(addon_offered),
        "llm_addon_accepted": _b2i(addon_accepted),
        "llm_addon_items_json": _j(addon_items),
        "llm_addon_evid_json": _j(addon_evid),

        # витрины (РУССКИЙ)
        "llm_availability_ru": MAP_AVAIL.get(avail, ""),
        "llm_season_ru": MAP_SEASON.get(season, ""),
        "llm_obj_type_ru": MAP_OBJ_TYPE.get(obj_type, ""),
        "llm_obj_status_ru": MAP_OBJ_STATUS.get(obj_status, ""),
        "llm_next_owner_ru": MAP_OWNER.get(nxt_owner, ""),
    }
    return out

def _cap(s: str) -> str:
    return s.capitalize() if isinstance(s, str) and s else s

def _mode_nonempty(series: pd.Series, allowed: set = None) -> str:
    if series is None or series.empty:
        return ""
    s = series.astype(str).str.strip()
    s = s[~s.isin(["", "nan", "none", "null"])]
    if allowed is not None and len(allowed) > 0:
        s = s[s.isin(list(allowed))]
    if s.empty:
        return ""
    return s.mode(dropna=True).iloc[0]

def _any_flag(series: pd.Series) -> int:
    if series is None or series.empty:
        return 0
    return int(pd.to_numeric(series, errors="coerce").fillna(0).max() > 0)

def _median_num(series: pd.Series):
    if series is None or series.empty:
        return np.nan
    x = pd.to_numeric(series, errors="coerce").dropna()
    return float(x.median()) if len(x) else np.nan

def extract_names_from_sales(sub: pd.DataFrame) -> Tuple[str, str]:
    spk = detect_speaker_col(sub)
    if spk is None:
        return "", ""
    rows = sub[sub[spk].astype(str).isin([SALES_VALUE, SPEAKER_1, SPEAKER_2])].copy()
    if rows.empty:
        return "", ""
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
        if not sales_name and fn and tp:
            sales_name = fn
        if not first_found_early and fn:
            try:
                if pd.notna(rn) and float(rn) < 3:
                    first_found_early = fn
            except Exception:
                pass
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

# ---------- Громкость (для глобальных флагов — при наличии) ----------
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

    # ---- Разворачиваем LLM JSON (новая схема) ----
    _llm_col = None
    for c in df.columns:
        if c == LLM_COL or c.startswith(LLM_COL + "."):
            _llm_col = c
            break

    if _llm_col and _llm_col in df.columns:
        llm_flat_rows = []
        for v in df[_llm_col]:
            obj = _safe_json_loads(v)
            llm_flat_rows.append(_llm_flat(obj) if isinstance(obj, dict) else {})
        llm_df = pd.DataFrame(llm_flat_rows)
        df = pd.concat([df.reset_index(drop=True), llm_df.reset_index(drop=True)], axis=1)

        # типизация для int-флагов
        int_cols = [
            "llm_obj_detected","llm_obj_ack","llm_obj_sol","llm_obj_check","llm_obj_score",
            "llm_next_set","llm_wrap_up","llm_addon_offered","llm_addon_accepted"
        ]
        for c in int_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # ---- Глобальные пороги громкости (если есть) ----
    sales_means_list: List[float] = []
    for _, sub_d in df.groupby(_DIALOG, dropna=False):
        sales_means_list.append(_sales_mean_for_dialog(sub_d))
    sales_means_per_dialog = pd.Series(sales_means_list, dtype=float).dropna()
    global_sales_q05 = float(sales_means_per_dialog.quantile(0.05)) if len(sales_means_per_dialog) else np.nan
    global_sales_q95 = float(sales_means_per_dialog.quantile(0.95)) if len(sales_means_per_dialog) else np.nan

    # ---- Агрегация на диалог ----
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

        for c in present_criteria:
            base[c] = aggregate_criterion_values_sales_only(sub, c)

        # агрегаты/моды
        base.update(dict(
            llm_intent=_mode_nonempty(sub.get("llm_intent"), {"покупка","уточнение","запись","гарантия"}),
            llm_category=_mode_nonempty(sub.get("llm_category"),
                                        {"Шины","Диски","АКБ","Масла/Жидкости","Услуги","Другое"}),
            llm_category_detail=_mode_nonempty(sub.get("llm_category_detail")),
            llm_tire_rim=_mode_nonempty(sub.get("llm_tire_rim")),
            llm_season_ru=_mode_nonempty(sub.get("llm_season_ru"), set(MAP_SEASON.values())),
            llm_availability_ru=_mode_nonempty(sub.get("llm_availability_ru"), set(MAP_AVAIL.values())),
            llm_obj_type_ru=_mode_nonempty(sub.get("llm_obj_type_ru"), set(MAP_OBJ_TYPE.values())),
            llm_obj_status_ru=_mode_nonempty(sub.get("llm_obj_status_ru"), set(MAP_OBJ_STATUS.values())),
            llm_next_owner_ru=_mode_nonempty(sub.get("llm_next_owner_ru"), set(MAP_OWNER.values())),

            llm_obj_detected=_any_flag(sub.get("llm_obj_detected")),
            llm_obj_score=_median_num(sub.get("llm_obj_score")),
            llm_obj_ack=_any_flag(sub.get("llm_obj_ack")),
            llm_obj_sol=_any_flag(sub.get("llm_obj_sol")),
            llm_obj_check=_any_flag(sub.get("llm_obj_check")),
            llm_next_set=_any_flag(sub.get("llm_next_set")),
            llm_next_action=_mode_nonempty(sub.get("llm_next_action")),
            llm_next_datetime=_mode_nonempty(sub.get("llm_next_datetime")),
            llm_next_place=_mode_nonempty(sub.get("llm_next_place")),
            llm_wrap_up=_any_flag(sub.get("llm_wrap_up")),
            llm_addon_offered=_any_flag(sub.get("llm_addon_offered")),
            llm_addon_accepted=_any_flag(sub.get("llm_addon_accepted")),
        ))

        rows.append(base)

    summary = pd.DataFrame(rows)

    # ===== Метрики для "Сводка" =====
    total_dialogs = summary["dialog_id"].nunique()
    total_seconds = float(pd.to_numeric(summary["duration_sec"], errors="coerce").fillna(0).sum())
    total_hours = total_seconds / 3600.0

    theme_counts = (
        summary["theme_class"].value_counts(dropna=False)
        .rename_axis("Тема").reset_index(name="Диалогов")
        .sort_values("Диалогов", ascending=False)
    )

    def _rate(col):
        return float(pd.to_numeric(summary.get(col, pd.Series([])), errors="coerce").fillna(0).mean())

    rates = {
        "Назначен шаг (next_set)": _rate("llm_next_set"),
        "Есть резюме (wrap_up)": _rate("llm_wrap_up"),
        "Есть возражение": _rate("llm_obj_detected"),
        "Допродажа предложена": _rate("llm_addon_offered"),
        "Допродажа принята": _rate("llm_addon_accepted"),
        "Признал возражение": _rate("llm_obj_ack"),
        "Дал решение по возражению": _rate("llm_obj_sol"),
        "Проверил закрытие возраж.": _rate("llm_obj_check"),
    }

    def _dist(series):
        s = series.fillna("").astype(str)
        s = s[s != ""]
        if s.empty:
            return pd.Series(dtype=float)
        return s.value_counts(normalize=True)

    dist_blocks = {
        "Интент": _dist(summary["llm_intent"]),
        "Категория": _dist(summary["llm_category"]),
        "Сезон (LLM)": _dist(summary["llm_season_ru"]),
        "Наличие (LLM)": _dist(summary["llm_availability_ru"]),
        "Тип возражения": _dist(summary["llm_obj_type_ru"]),
        "Статус возражения": _dist(summary["llm_obj_status_ru"]),
        "Ответственный за шаг": _dist(summary["llm_next_owner_ru"]),
    }

    # Корреляции
    corr_df = pd.DataFrame()
    base_bin_cols = [
        "llm_next_set","llm_wrap_up","llm_addon_offered","llm_addon_accepted",
        "llm_obj_detected","llm_obj_ack","llm_obj_sol","llm_obj_check"
    ]
    for c in base_bin_cols:
        if c in summary.columns:
            corr_df[c] = pd.to_numeric(summary[c], errors="coerce").fillna(0).astype(int)
    if "llm_availability_ru" in summary.columns:
        corr_df = pd.concat([corr_df, pd.get_dummies(summary["llm_availability_ru"], prefix="Наличие", dtype=int)], axis=1)
    if "llm_intent" in summary.columns:
        corr_df = pd.concat([corr_df, pd.get_dummies(summary["llm_intent"], prefix="Интент", dtype=int)], axis=1)
    if "llm_category" in summary.columns:
        corr_df = pd.concat([corr_df, pd.get_dummies(summary["llm_category"], prefix="Категория", dtype=int)], axis=1)
    corr_matrix = corr_df.corr(method="pearson") if not corr_df.empty else pd.DataFrame()

    # ===== Пишем Excel =====
    wb = xlsxwriter.Workbook(out_path)

    # форматы
    fmt_header = wb.add_format({"bold": True, "bg_color": "#F2F2F2"})
    fmt_wrap   = wb.add_format({"text_wrap": True, "valign": "top"})
    fmt_norm   = wb.add_format({"valign": "top"})
    fmt_bold   = wb.add_format({"bold": True})
    fmt_num    = wb.add_format({"num_format": "0"})
    fmt_dur    = wb.add_format({"num_format": "0.0"})
    fmt_int    = wb.add_format({"num_format": "0"})
    fmt_pct    = wb.add_format({"num_format": "0.0%"})
    fmt_corr   = wb.add_format({"num_format": "0.00"})

    # ---------- Лист 1: Сводка ----------
    ws_sum = wb.add_worksheet("Сводка")
    ws_sum.set_default_row(13)
    r = 0
    ws_sum.write(r, 0, "Сводка по диалогам", fmt_header); r += 1
    ws_sum.write(r, 0, "Диалогов:", fmt_bold);        ws_sum.write_number(r, 1, int(total_dialogs), fmt_num); r += 1
    ws_sum.write(r, 0, "Суммарно, часов:", fmt_bold); ws_sum.write_number(r, 1, round(total_hours, 1), fmt_dur); r += 2

    # Итоги по LLM-критериям
    ws_sum.write(r, 0, "Итоги по LLM-критериям", fmt_header); r += 1
    start_rates = r
    ws_sum.write(r, 0, "Показатель", fmt_bold); ws_sum.write(r, 1, "Доля", fmt_bold); r += 1
    for k, v in rates.items():
        ws_sum.write(r, 0, k, fmt_norm); ws_sum.write_number(r, 1, float(v), fmt_pct); r += 1
    end_rates = r - 1

    # график по критериям
    if end_rates >= start_rates + 1:
        ch = wb.add_chart({"type": "column"})
        ch.set_title({"name": "Доля по LLM-критериям"})
        ch.add_series({
            "name": "Доля",
            "categories": ["Сводка", start_rates + 1, 0, end_rates, 0],
            "values":     ["Сводка", start_rates + 1, 1, end_rates, 1],
            "data_labels": {"value": True},
        })
        ch.set_y_axis({"major_gridlines": {"visible": False}})
        ws_sum.insert_chart(start_rates, 3, ch, {"x_scale": 1.2, "y_scale": 1.15})
    r = end_rates + 2

    # Диалоги по темам
    ws_sum.write(r, 0, "Диалоги по темам", fmt_header); r += 1
    ws_sum.write(r, 0, "Тема", fmt_bold); ws_sum.write(r, 1, "Кол-во", fmt_bold); r += 1
    start_themes = r
    for rec in theme_counts.itertuples(index=False):
        ws_sum.write(r, 0, rec.Тема or "", fmt_norm)
        ws_sum.write_number(r, 1, int(rec.Диалогов), fmt_num)
        r += 1
    end_themes = r - 1
    if end_themes >= start_themes:
        ch_t = wb.add_chart({"type": "column"})
        ch_t.set_title({"name": "Темы диалогов"})
        ch_t.add_series({
            "name": "Диалогов",
            "categories": ["Сводка", start_themes, 0, end_themes, 0],
            "values":     ["Сводка", start_themes, 1, end_themes, 1],
            "data_labels": {"value": True},
        })
        ws_sum.insert_chart(start_themes, 3, ch_t, {"x_scale": 1.2, "y_scale": 1.1})
    r = end_themes + 2

    # Распределения (только основные блоки с графиками)
    dist_for_charts_order = ["Интент", "Категория", "Наличие (LLM)", "Статус возражения"]
    for block_name in dist_for_charts_order:
        ser = dist_blocks.get(block_name, pd.Series(dtype=float))
        ws_sum.write(r, 0, block_name, fmt_header); r += 1
        ws_sum.write(r, 0, "Значение", fmt_bold); ws_sum.write(r, 1, "Доля", fmt_bold); r += 1
        start_blk = r
        if ser.empty:
            ws_sum.write(r, 0, "— нет данных —", fmt_norm); r += 2
            continue
        max_rows = 8
        for idx, (val, pct) in enumerate(ser.items()):
            if idx >= max_rows:
                break
            ws_sum.write(r, 0, str(val), fmt_norm)
            ws_sum.write_number(r, 1, float(pct), fmt_pct)
            r += 1
        end_blk = r - 1
        ch_b = wb.add_chart({"type": "column"})
        ch_b.set_title({"name": f"{block_name} — распределение"})
        ch_b.add_series({
            "name": "Доля",
            "categories": ["Сводка", start_blk, 0, end_blk, 0],
            "values":     ["Сводка", start_blk, 1, end_blk, 1],
            "data_labels": {"value": True},
        })
        ch_b.set_y_axis({"major_gridlines": {"visible": False}})
        ws_sum.insert_chart(start_blk, 3, ch_b, {"x_scale": 1.2, "y_scale": 1.1})
        r = end_blk + 2

    # Корреляционная матрица (с цветовой шкалой)
    ws_sum.write(r, 0, "Корреляции LLM-показателей", fmt_header); r += 1
    if not corr_matrix.empty:
        # заголовки
        ws_sum.write(r, 0, "", fmt_header)
        for j, col in enumerate(corr_matrix.columns, start=1):
            ws_sum.write(r, j, col, fmt_header)
        for i, row_name in enumerate(corr_matrix.index, start=1):
            ws_sum.write(r + i, 0, row_name, fmt_header)
            for j, col_name in enumerate(corr_matrix.columns, start=1):
                ws_sum.write_number(r + i, j, float(corr_matrix.loc[row_name, col_name]), fmt_corr)
        # условное форматирование "тепло-карта"
        ws_sum.conditional_format(r + 1, 1, r + len(corr_matrix.index), 1 + len(corr_matrix.columns), {
            "type": "3_color_scale",
            "min_value": -1, "max_value": 1
        })
        ws_sum.freeze_panes(r + 1, 1)
        r = r + len(corr_matrix.index) + 2
    else:
        ws_sum.write(r, 0, "Недостаточно данных для расчёта корреляций.", fmt_norm); r += 1

    ws_sum.set_column(0, 0, 28, fmt_norm)
    ws_sum.set_column(1, 1, 12, fmt_norm)
    # оставшиеся колонки шире под графики
    for c in range(3, 10):
        ws_sum.set_column(c, c, 24)

    # ---------- Лист 2: Диалоги ----------
    ws = wb.add_worksheet("Диалоги")
    ws.set_default_row(13)

    base_headers = [
        ("dialog_id", "Идентификатор диалога"),
        ("theme_class", "Класс темы"),
        ("theme", "Тема"),
        ("file_name", "Имя файла"),
        ("name_sales", "Имя SALES"),
        ("name_client", "Обращение к клиенту"),

        ("llm_intent", "Интент"),
        ("llm_category", "Категория"),
        ("llm_category_detail", "Деталь категории"),
        ("llm_tire_rim", "Диаметр, дюйм"),
        ("llm_season_ru", "Сезон"),
        ("llm_availability_ru", "Наличие"),

        ("llm_obj_detected", "Возражение?"),
        ("llm_obj_type_ru", "Тип возражения"),
        ("llm_obj_status_ru", "Статус возражения"),
        ("llm_obj_score", "Оценка обр."),  # 0–3
        ("llm_obj_ack", "Признал"),
        ("llm_obj_sol", "Решение"),
        ("llm_obj_check", "Проверка закрытия"),

        ("llm_next_set", "Назначен шаг"),
        ("llm_next_action", "Действие"),
        ("llm_next_datetime", "Дата/время"),
        ("llm_next_place", "Место"),
        ("llm_next_owner_ru", "Ответственный"),

        ("llm_wrap_up", "Резюме/итог"),
        ("llm_addon_offered", "Допродажа предложена"),
        ("llm_addon_accepted", "Допродажа принята"),

        ("duration_sec", "Длит., с"),
        ("rows_count", "Строк"),
        ("sales_quieter_95_global", "SALES тише 95% всех"),
        ("sales_louder_95_global", "SALES громче 95% всех"),
        ("_expand", "▼"),
    ]
    present_for_summary = [
        (c, ru) for c, ru in CRITERIA_LABELS
        if c in summary.columns and c not in ("found_name", "telling_name_phrases")
    ]
    headers = base_headers + present_for_summary

    # заголовки
    for c, (_key, title) in enumerate(headers):
        ws.write(0, c, title, fmt_header)

    col_widths = {
        "Идентификатор диалога": 36, "Класс темы": 12, "Тема": 28, "Имя файла": 28,
        "Имя SALES": 16, "Обращение к клиенту": 18,
        "Интент": 12, "Категория": 16, "Деталь категории": 18,
        "Диаметр, дюйм": 14, "Сезон": 12, "Наличие": 12,
        "Возражение?": 12, "Тип возражения": 18, "Статус возражения": 18, "Оценка обр.": 12,
        "Признал": 10, "Решение": 10, "Проверка закрытия": 16,
        "Назначен шаг": 14, "Действие": 18, "Дата/время": 18, "Место": 16, "Ответственный": 14,
        "Резюме/итог": 14, "Допродажа предложена": 18, "Допродажа принята": 16,
        "Длит., с": 10, "Строк": 8, "SALES тише 95% всех": 18, "SALES громче 95% всех": 20, "▼": 4
    }
    for idx, (_key, title) in enumerate(headers):
        ws.set_column(idx, idx, col_widths.get(title, 16),
                      fmt_wrap if title in ("Тема","Деталь категории","Действие","Дата/время","Место") else fmt_norm)

    num_keys_int = {
        "rows_count", "sales_quieter_95_global", "sales_louder_95_global",
        "llm_obj_detected", "llm_obj_ack", "llm_obj_sol", "llm_obj_check",
        "llm_next_set", "llm_wrap_up", "llm_addon_offered", "llm_addon_accepted"
    }
    num_keys_float = {"duration_sec", "llm_obj_score"}

    row = 1
    total_chars_width = sum(col_widths.get(headers[idx][1], 16) for idx in range(1, len(headers)))
    for _, rsum in summary.iterrows():
        val_by_key = {k: rsum.get(k, "") for k, _ in headers}
        # нормализуем числа
        val_by_key["duration_sec"] = float(val_by_key.get("duration_sec") or 0.0)
        for k in num_keys_int:
            val_by_key[k] = int(pd.to_numeric(val_by_key.get(k), errors="coerce") if str(val_by_key.get(k)) != "" else 0)
        for k in num_keys_float:
            v = val_by_key.get(k)
            try:
                val_by_key[k] = float(v) if pd.notna(v) else 0.0
            except Exception:
                val_by_key[k] = 0.0
        val_by_key["_expand"] = "↓"

        # запись строки
        for c, (key, title) in enumerate(headers):
            v = val_by_key.get(key, "")
            if key in num_keys_float:
                ws.write_number(row, c, float(v), fmt_dur if key == "duration_sec" else fmt_corr)
            elif key in num_keys_int:
                ws.write_number(row, c, int(v), fmt_int)
            elif key == "_expand":
                # ссылка на скрытую строку ниже на этом же листе
                ws.write_url(row, c, f"internal:'Диалоги'!A{row + 2}", fmt_bold, string="↓")
            else:
                ws.write(row, c, v, fmt_wrap if title in ("Тема","Деталь категории","Действие","Дата/время","Место") else fmt_norm)

        ws.set_row(row, 12)

        # скрытая строка «Полный текст»
        detail_row = row + 1
        ws.write(detail_row, 0, "Полный текст:", fmt_bold)
        est_h = estimate_merged_row_height(_as_str(rsum.get("dialog_text","")), total_chars_width, line_height_pt=14.0)
        ws.merge_range(detail_row, 1, detail_row, len(headers) - 1, _as_str(rsum.get("dialog_text","")), fmt_wrap)
        ws.set_row(detail_row, est_h, None, {"hidden": True, "level": 1})

        row += 2

    ws.freeze_panes(1, 1)
    ws.autofilter(0, 0, row - 1, len(headers) - 1)
    ws.outline_settings(True, False, True, False)

    wb.close()
    return out_path

# ---------- CLI ----------
if __name__ == "__main__":
    repo = AudioDialogRepository()
    df = repo.get_all_for_report()
    out = make_report(df, "dialogs_report.xlsx")
    print(f"Готово: {out}")
