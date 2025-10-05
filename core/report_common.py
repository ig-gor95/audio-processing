import re, json, textwrap
from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd

# ===== Входные колонки =====
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
SPEAKER_1 = "Speaker_1"
SPEAKER_2 = "Speaker_2"
EXCEL_MAX_CELL = 32767

# ===== Русификация =====
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

# ===== Служебные =====
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

# --- очистка текста ---
_REPEAT_WORD_RX = re.compile(r'(?iu)\b(\w+)(?:[\s,\-–—.?!…]+\1\b){2,}')
def _collapse_repeats(line: str, keep: int = 2) -> str:
    if keep < 1:
        keep = 1
    def _repl(m: re.Match) -> str:
        w = m.group(1); sep = " "
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

def _nn(x): return "" if x is None else x
def _b2i(b): return 1 if b is True else 0
def _aslist(x): return x if isinstance(x, list) else []
def _j(x):
    try: return json.dumps(x, ensure_ascii=False)
    except Exception: return ""

# ===== Плоский парсер новой LLM-схемы + RU витрины =====
def llm_flat(row_obj: dict) -> dict:
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
        x = _nn(x); return x if x in allowed else ""

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
    obj_score    = int((objection.get("score") if isinstance(objection, dict) else 0) or 0)
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

    return {
        # исходники
        "llm_intent": intent,
        "llm_category": cat,
        "llm_category_detail": catdet,
        "llm_tire_rim": (str(tire_rim) if tire_rim not in (None, "") else ""),
        "llm_availability": avail,
        "llm_season": season,

        "llm_obj_detected": _b2i(obj_detected),
        "llm_obj_type": obj_type,
        "llm_obj_status": obj_status,
        "llm_obj_score": obj_score,
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

        # витрины (RU)
        "llm_availability_ru": MAP_AVAIL.get(avail, ""),
        "llm_season_ru": MAP_SEASON.get(season, ""),
        "llm_obj_type_ru": MAP_OBJ_TYPE.get(obj_type, ""),
        "llm_obj_status_ru": MAP_OBJ_STATUS.get(obj_status, ""),
        "llm_next_owner_ru": MAP_OWNER.get(nxt_owner, ""),
    }

def _mode_nonempty(series: pd.Series, allowed: set = None) -> str:
    if series is None or series.empty: return ""
    s = series.astype(str).str.strip()
    s = s[~s.isin(["", "nan", "none", "null"])]
    if allowed: s = s[s.isin(list(allowed))]
    if s.empty: return ""
    return s.mode(dropna=True).iloc[0]

def _any_flag(series: pd.Series) -> int:
    if series is None or series.empty: return 0
    return int(pd.to_numeric(series, errors="coerce").fillna(0).max() > 0)

def _median_num(series: pd.Series):
    if series is None or series.empty: return np.nan
    x = pd.to_numeric(series, errors="coerce").dropna()
    return float(x.median()) if len(x) else np.nan

def _first_token(name: str) -> str:
    if not isinstance(name, str): return ""
    name = name.strip().strip(_TRIM)
    if not name: return ""
    token = re.split(r"[\s,;/\\|]+", name, maxsplit=1)[0]
    return token.strip(_TRIM)

def extract_names_from_sales(sub: pd.DataFrame) -> Tuple[str, str]:
    spk = detect_speaker_col(sub)
    if spk is None: return "", ""
    rows = sub[sub[spk].astype(str).isin([SALES_VALUE, SPEAKER_1, SPEAKER_2])].copy()
    if rows.empty: return "", ""
    if ROWNUM_COL in rows.columns:
        rows = rows.sort_values(ROWNUM_COL, kind="mergesort")
    elif START_COL in rows.columns:
        rows = rows.sort_values(START_COL, kind="mergesort")
    sales_name, first_found_early, client_candidates = "", "", []
    for _, r in rows.iterrows():
        fn = _first_token(_as_str(r.get("found_name", "")))
        tp = _as_str(r.get("telling_name_phrases", ""))
        rn = r.get(ROWNUM_COL, np.nan)
        if not sales_name and fn and tp:
            sales_name = fn
        if not first_found_early and fn:
            try:
                if pd.notna(rn) and float(rn) < 3: first_found_early = fn
            except Exception:
                pass
        if fn and not tp:
            client_candidates.append(fn)
    if not sales_name and first_found_early:
        sales_name = first_found_early
    client_name = ""
    for cand in client_candidates:
        if not sales_name or cand.lower() != sales_name.lower():
            client_name = cand; break
    return (sales_name.capitalize() if sales_name else ""), (client_name.capitalize() if client_name else "")

def aggregate_criterion_values_sales_only(sub: pd.DataFrame, colname: str) -> str:
    spk_col = detect_speaker_col(sub)
    if spk_col is None or colname not in sub.columns: return ""
    sub_sales = sub[sub[spk_col].astype(str).isin([SALES_VALUE, SPEAKER_1, SPEAKER_2])]
    if sub_sales.empty: return ""
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
            if it2: vals.append(it2)
    uniq = list(dict.fromkeys(vals))
    return ", ".join(uniq)

def estimate_merged_row_height(text: str, total_chars_width: int, line_height_pt: float = 14.0) -> float:
    if not text: return line_height_pt
    lines = 0
    for raw_line in text.split("\n"):
        seg = raw_line.strip()
        if not seg:
            lines += 1; continue
        wrapped = textwrap.wrap(seg, width=max(8, total_chars_width))
        lines += max(1, len(wrapped))
    return max(line_height_pt, lines * line_height_pt * 1.05)

# ---------- Громкость ----------
def _series_loud(df: pd.DataFrame, speaker_value: str) -> pd.Series:
    spk = detect_speaker_col(df)
    if spk is None or LOUD_COL not in df.columns:
        return pd.Series([], dtype=float)
    ser = pd.to_numeric(df.loc[df[spk].astype(str) == speaker_value, LOUD_COL], errors="coerce")
    return ser.dropna()

def _concat_nonempty(parts: List[pd.Series]) -> pd.Series:
    parts = [s for s in parts if s is not None and not s.empty]
    if not parts: return pd.Series([], dtype=float)
    return pd.concat(parts, ignore_index=True)

def sales_mean_for_dialog(sub: pd.DataFrame) -> float:
    ser = _concat_nonempty([
        _series_loud(sub, SALES_VALUE),
        _series_loud(sub, SPEAKER_1),
        _series_loud(sub, SPEAKER_2),
    ])
    return float(ser.mean()) if not ser.empty else np.nan
