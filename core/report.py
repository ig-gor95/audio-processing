# -*- coding: utf-8 -*-
"""
Один файл: загрузка → LLM-плоский парс → агрегация → запись 2 листов (Summary, Диалоги)

Правки:
- Таблицы начинаются с колонки A
- Графики справа от таблиц, без наложений (динамический отступ по высоте графика)
- Корреляция начинается с A
- Диалоги «раскрываются»: на каждую строку есть скрытая деталь с полным текстом (группировка/outline)
- Объекция/допродажа — словами; если возражения нет, тип/статус пустые; позиции допродажи выводятся
- Имена SALES/клиента — по первым 3 строкам SALES и правилу отличия
"""

import re
import json
import textwrap
from typing import Optional, List, Dict, Tuple, Iterable

import numpy as np
import pandas as pd
import xlsxwriter

# ==== источник данных (опционально) ====
try:
    from core.repository.audio_dialog_repository import AudioDialogRepository
except Exception:
    AudioDialogRepository = None

# ===== колонки-стандарты =====
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
CHART_HEIGHT_ROWS = 18  # визуальная высота графика в строках (для безопасного отступа)
CHART_COL = 6  # колонки для графиков (G=6), чтобы таблицы всегда начинались с A

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

# ===== сервис =====
_SPLIT_RE = re.compile(r"[;,|/\\]+")
_TRIM = " \"'“”«»·-–—"

CH_COL_SIZE = {"width": 478, "height": 300}
CH_COL_SETTING = {"x_offset": 0, "y_offset": 0, "object_position": 1}

ADDON_ITEM_RULES: List[tuple[re.Pattern, str]] = [
    # ПАКЕТЫ
    (re.compile(r'(?i)\bпакет(ы)?\s*(для)?\s*(шин|кол(е|ё)с)\b'), "Пакеты для шин"),

    # ХРАНЕНИЕ
    (re.compile(r'(?i)\bсезонн\w*\s+хранен'), "Сезонное хранение"),
    (re.compile(r'(?i)\bхранен'), "Сезонное хранение"),

    # ШИНОМОНТАЖ / АКЦИИ
    (re.compile(r'(?i)\bшиномонта?ж\b'), "Шиномонтаж"),
    (re.compile(r'(?i)\bбесплатн\w*\s+(шиномонта?ж|переобув)'), "Шиномонтаж (акция)"),
    (re.compile(r'(?i)\bкупон\w*\s+на\s+бесплатн\w*\s+шиномонта?ж'), "Шиномонтаж (акция)"),
    (re.compile(r'(?i)\bшиномонта?ж\s+подар'), "Шиномонтаж (акция)"),
    (re.compile(r'(?i)\bакци(я|и)\b'), "Акции/купоны"),

    # МОЙКА КОЛЁС
    (re.compile(r'(?i)\bтехн(ологическ\w*\s+)?мойк\w*\s*(кол(е|ё)с)?'), "Мойка колёс"),
    (re.compile(r'(?i)\bмойк\w*\s*(кол(е|ё)с)?'), "Мойка колёс"),

    # МОНТАЖ / ДЕМОНТАЖ / КОМПЛЕКС
    (re.compile(r'(?i)\bдемонтаж[-\s]*монтаж[-\s]*балансиров'), "Комплекс ДМБ"),
    (re.compile(r'(?i)\bсняти?е?\s*(и|/)?\s*установ'), "Снятие/установка"),
    (re.compile(r'(?i)\bмонтаж(?!.*балансиров)'), "Монтаж"),
    (re.compile(r'(?i)\bдемонтаж\b'), "Демонтаж"),

    # БАЛАНСИРОВКА / ГРУЗИКИ
    (re.compile(r'(?i)\bбалансиров'), "Балансировка"),
    (re.compile(r'(?i)\bгруз(ы|иков)?\b'), "Балансировочные грузики"),
    (re.compile(r'(?i)\bустановк\w*\s+грузик'), "Балансировочные грузики"),

    # ВЕНТИЛИ / ДАТЧИКИ
    (re.compile(r'(?i)\bвентил(ь|я|и)\b'), "Вентили"),
    (re.compile(r'(?i)\bдатчик(ов)?\b'), "Датчики давления (TPMS)"),
    (re.compile(r'(?i)\bустановк\w*\s+датчик'), "Датчики давления (TPMS)"),

    # ГАРАНТИЯ
    (re.compile(r'(?i)\b(доп(\.|олнительн\w*)?\s+)?расширенн\w*\s+гарант'), "Доп. гарантия (расширенная)"),
    (re.compile(r'(?i)\bдоп(\.|олнительн\w*)?\s+гарант'), "Доп. гарантия (расширенная)"),

    # ПРОЧИЕ СЕРВИСЫ
    (re.compile(r'(?i)\bрезервирован|резервировани|резервировани|резерв\b'), "Резервирование"),
    (re.compile(r'(?i)\bутилиз(аци|ация)\s+шин'), "Утилизация шин"),
    (re.compile(r'(?i)\bонлайн\s+оплат'), "Онлайн оплата"),
    (re.compile(r'(?i)\bподарочн\w*\s+литр\w*\s+бензин'), "Подарочные литры бензина"),
    (re.compile(r'(?i)\bтрейд[-\s]*ин\b'), "Трейд-ин"),

    # ТОВАРЫ / КОМПЛЕКТУХА
    (re.compile(r'(?i)\bкреп(е|ё)ж\b'), "Крепёж"),
    (re.compile(r'(?i)\bпроставк\w+'), "Проставки"),
    (re.compile(r'(?i)\bколпак\w+'), "Колпаки"),
    (re.compile(r'(?i)\bкомпрессор\b'), "Компрессор"),
    (re.compile(r'(?i)\bкол(е|ё)са\b'), "Колёса/комплекты"),
    (re.compile(r'(?i)\bшины\b'), "Шины"),

    # ОШИПОВКА
    (re.compile(r'(?i)\bошиповк'), "Ошиповка"),
]
def _norm_text(s: str) -> str:
    return s.lower().replace("ё", "е").strip()

def normalize_addon_item(raw: str) -> str:
    """Возвращает каноническое имя типа допродажи по строке ввода."""
    if not raw or not str(raw).strip():
        return ""
    s = _norm_text(str(raw))
    for rx, canon in ADDON_ITEM_RULES:
        if rx.search(s):
            return canon
    # не нашли — эвристика по простым ключам «пакет»
    if re.search(r'\bпакет', s):
        return "Пакеты для шин"
    return raw.strip().capitalize()

def normalize_addon_items(items: Iterable[str]) -> List[str]:
    """Нормализует список items и убирает дубликаты, сохраняя порядок."""
    seen = set()
    out: List[str] = []
    for it in items or []:
        canon = normalize_addon_item(it)
        if not canon:
            continue
        if canon not in seen:
            seen.add(canon)
            out.append(canon)
    return out

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


def _corr_ru_label(col: str) -> str:
    base = {
        "llm_next_set": "Назначен шаг",
        "llm_wrap_up": "Резюме/итог",
        "llm_addon_offered": "Допродажа предложена",
        "llm_addon_accepted": "Допродажа принята",
        "llm_obj_detected": "Есть возражение",
        "llm_obj_ack": "Признал возражение",
        "llm_obj_sol": "Решение по возражению",
        "llm_obj_check": "Проверка закрытия возражения",
    }
    if col.startswith("llm_availability_ru_"):
        return "Наличие: " + col.split("llm_availability_ru_", 1)[1]
    if col.startswith("llm_intent_"):
        return "Интент: " + col.split("llm_intent_", 1)[1]
    if col.startswith("llm_category_"):
        return "Категория: " + col.split("llm_category_", 1)[1]
    return base.get(col, col)


def _as_str(x) -> str:
    if isinstance(x, pd.Series):
        x = x.iloc[0] if not x.empty else ""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x)


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


# ===== детектор спикера =====
def detect_speaker_col(df: pd.DataFrame) -> Optional[str]:
    if SPEAKER_COL in df.columns:
        return SPEAKER_COL
    if ALT_SPEAKER_COL in df.columns:
        return ALT_SPEAKER_COL
    return None


# ===== классификация темы =====
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


# ===== очистка текста (длинные ячейки) =====
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


# ===== громкость SALES =====
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


def sales_mean_for_dialog(sub: pd.DataFrame) -> float:
    ser = _concat_nonempty([
        _series_loud(sub, SALES_VALUE),
        _series_loud(sub, SPEAKER_1),
        _series_loud(sub, SPEAKER_2),
    ])
    return float(ser.mean()) if not ser.empty else np.nan


# ===== разбор фразовых колонок =====
def extract_phrases(series: pd.Series) -> List[str]:
    out: List[str] = []
    for v in series.dropna().astype(str):
        s = v.strip()
        if not s or s.lower() in {"nan", "none", '""'}:
            continue
        for p in _SPLIT_RE.split(s):
            p2 = p.strip(_TRIM)
            p2 = re.sub(r"\s+", " ", p2)
            if p2:
                out.append(p2)
    return out


def aggregate_phrases_sales_only(sub: pd.DataFrame, colname: str) -> str:
    spk_col = detect_speaker_col(sub)
    if spk_col is None or colname not in sub.columns:
        return ""
    mask_sales = sub[spk_col].astype(str).isin([SALES_VALUE, SPEAKER_1, SPEAKER_2])
    vals = extract_phrases(sub.loc[mask_sales, colname])
    if not vals:
        return ""
    uniq = list(dict.fromkeys(vals))
    return ", ".join(uniq)


def aggregate_phrases_all_speakers(sub: pd.DataFrame, colname: str) -> str:
    if colname not in sub.columns:
        return ""
    vals = extract_phrases(sub[colname])
    if not vals:
        return ""
    uniq = list(dict.fromkeys(vals))
    return ", ".join(uniq)


# ===== имена =====
def extract_names_from_sales_strict(sub: pd.DataFrame) -> Tuple[str, str]:
    """Имя SALES — если в первых 3 строках SALES есть found_name и telling_name_phrases.
       Обращение к клиенту — первое found_name, отличное от имени SALES."""
    spk = detect_speaker_col(sub)
    if spk is None:
        return "", ""
    if ROWNUM_COL in sub.columns:
        sub = sub.sort_values(ROWNUM_COL, kind="mergesort")
    elif START_COL in sub.columns:
        sub = sub.sort_values(START_COL, kind="mergesort")

    sales_rows = sub[sub[spk].astype(str).isin([SALES_VALUE, SPEAKER_1, SPEAKER_2])].head(3)
    sales_name = ""
    for _, r in sales_rows.iterrows():
        fn = _first_token(_as_str(r.get("found_name", "")))
        tp = _as_str(r.get("telling_name_phrases", ""))
        if fn and tp:
            sales_name = fn
            break

    client_name = ""
    for _, r in sub.iterrows():
        fn = _first_token(_as_str(r.get("found_name", "")))
        if fn and (not sales_name or fn.lower() != sales_name.lower()):
            client_name = fn
            break

    def _cap(s: str) -> str:
        return s.capitalize() if s else s

    return _cap(sales_name), _cap(client_name)


# ===== LLM-плоский парс =====
def llm_flat(row_obj: dict) -> dict:
    if not isinstance(row_obj, dict):
        return {}

    _intent_allowed = {"покупка", "уточнение", "запись", "гарантия"}
    _cat_allowed = {"Шины", "Диски", "АКБ", "Масла/Жидкости", "Услуги", "Другое"}
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
    add_on = row_obj.get("add_on_sale", {}) or {}

    intent = _enum(row_obj.get("intent"), _intent_allowed)
    cat = _enum(row_obj.get("category"), _cat_allowed)
    catdet = _nn(row_obj.get("category_detail"))
    tire_rim = row_obj.get("tire_rim")
    avail = _enum(row_obj.get("availability"), _avail_allowed)
    season = _enum(row_obj.get("season"), _season_allowed)

    obj_detected = bool(isinstance(objection, dict) and objection.get("detected") is True)
    obj_type = _enum(objection.get("type") if isinstance(objection, dict) else "", _obj_type_allowed)
    obj_status = _enum(objection.get("status") if isinstance(objection, dict) else "", _obj_status_allowed)
    obj_ack = bool(handling.get("acknowledge") if isinstance(handling, dict) else False)
    obj_sol = bool(handling.get("solution") if isinstance(handling, dict) else False)
    obj_check = bool(handling.get("check_close") if isinstance(handling, dict) else False)

    # если возражения нет — тип/статус очищаем
    if not obj_detected:
        obj_type = ""
        obj_status = ""

    nxt_set = bool(next_set.get("set") if isinstance(next_set, dict) else False)
    nxt_action = _nn(next_set.get("action") if isinstance(next_set, dict) else "")
    nxt_dt = _nn(next_set.get("datetime") if isinstance(next_set, dict) else "")
    nxt_place = _nn(next_set.get("place") if isinstance(next_set, dict) else "")
    nxt_owner = _enum(next_set.get("owner") if isinstance(next_set, dict) else "", _owner_allowed)

    wrap_up = bool(row_obj.get("wrap_up") is True)

    addon_offered = bool(add_on.get("offered") if isinstance(add_on, dict) else False)
    addon_accepted = bool(add_on.get("accepted") if isinstance(add_on, dict) else False)
    addon_items = _aslist(add_on.get("items") if isinstance(add_on, dict) else [])

    # витрины RU
    avail_ru = MAP_AVAIL.get(avail, "")
    season_ru = MAP_SEASON.get(season, "")
    obj_type_ru = MAP_OBJ_TYPE.get(obj_type, "") if obj_type else ""
    obj_status_ru = MAP_OBJ_STATUS.get(obj_status, "") if obj_status else ""
    nxt_owner_ru = MAP_OWNER.get(nxt_owner, "")

    # текстовые поля для человекочитаемого вывода
    addon_offered_text = "да" if addon_offered else "нет"
    addon_accepted_text = "да" if addon_accepted else "нет"
    obj_detected_text = "да" if obj_detected else "нет"
    obj_ack_text = "да" if obj_ack else "нет"
    obj_sol_text = "да" if obj_sol else "нет"
    obj_check_text = "да" if obj_check else "нет"

    return {
        "llm_intent": intent,
        "llm_category": cat,
        "llm_category_detail": catdet,
        "llm_tire_rim": tire_rim,
        "llm_availability": avail,
        "llm_availability_ru": avail_ru,
        "llm_season": season,
        "llm_season_ru": season_ru,

        # objection — словами (+ бинарники для расчётов)
        "llm_obj_detected": _b2i(obj_detected),
        "llm_obj_detected_text": obj_detected_text,
        "llm_obj_type": obj_type,
        "llm_obj_type_ru": obj_type_ru,
        "llm_obj_status": obj_status,
        "llm_obj_status_ru": obj_status_ru,
        "llm_obj_ack": _b2i(obj_ack),
        "llm_obj_ack_text": obj_ack_text,
        "llm_obj_sol": _b2i(obj_sol),
        "llm_obj_sol_text": obj_sol_text,
        "llm_obj_check": _b2i(obj_check),
        "llm_obj_check_text": obj_check_text,

        # next
        "llm_next_set": _b2i(nxt_set),
        "llm_next_action": nxt_action,
        "llm_next_datetime": nxt_dt,
        "llm_next_place": nxt_place,
        "llm_next_owner": nxt_owner,
        "llm_next_owner_ru": nxt_owner_ru,

        # wrap/addon (и слова)
        "llm_wrap_up": _b2i(wrap_up),
        "llm_addon_offered": _b2i(addon_offered),
        "llm_addon_offered_text": addon_offered_text,
        "llm_addon_accepted": _b2i(addon_accepted),
        "llm_addon_accepted_text": addon_accepted_text,
        "llm_addon_items_json": _j(addon_items),
    }


# ===== агрегаты по диалогу =====
def _mode_nonempty(series: pd.Series, allowed: set = None) -> str:
    if series is None or series.empty:
        return ""
    s = series.astype(str).str.strip()
    s = s[~s.isin(["", "nan", "none", "null"])]
    if allowed is not None:
        s = s[s.isin(allowed)]
    if s.empty:
        return ""
    return s.mode(dropna=True).iloc[0]


def _any_flag(series: pd.Series) -> int:
    if series is None or series.empty:
        return 0
    return int(pd.to_numeric(series, errors="coerce").fillna(0).max() > 0)


def _join_items_from_json(series: pd.Series) -> str:
    items: List[str] = []
    for v in series.dropna():
        try:
            arr = v if isinstance(v, list) else json.loads(str(v))
            if isinstance(arr, list):
                for it in arr:
                    s = str(it).strip()
                    if s:
                        items.append(s)
        except Exception:
            pass
    uniq = list(dict.fromkeys(items))
    return ", ".join(uniq)


def aggregate_per_dialog(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # глобальные пороги по громкости
    sales_means = []
    for _, sub in df.groupby(DIALOG_COL, dropna=False):
        sales_means.append(sales_mean_for_dialog(sub))
    sales_means = pd.Series(sales_means, dtype=float).dropna()
    q05 = float(sales_means.quantile(0.05)) if len(sales_means) else np.nan
    q95 = float(sales_means.quantile(0.95)) if len(sales_means) else np.nan

    for did, sub in df.groupby(DIALOG_COL, dropna=False):
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

        # громкость локально/глобально
        mean_local = sales_mean_for_dialog(sub)
        quieter_95 = int(pd.notna(mean_local) and pd.notna(q05) and mean_local <= q05)
        louder_95 = int(pd.notna(mean_local) and pd.notna(q95) and mean_local >= q95)

        base = dict(
            dialog_id=str(did),
            theme_class=theme_class, #todo удалить столбец
            theme=theme_joined, #todo удалить столбец
            file_name=str(sub.get(FILE_COL, pd.Series([""])).iloc[0]) if FILE_COL in sub.columns else "",
            duration_sec=duration,
            rows_count=int(len(sub)),
            sales_quieter_95_global=quieter_95,
            sales_louder_95_global=louder_95,
            name_sales="",  # заполним позже
            name_client="",  # заполним позже
        )

        # LLM агрегаты
        obj_detected_any = _any_flag(sub.get("llm_obj_detected"))
        base.update(dict(
            llm_intent=_mode_nonempty(sub.get("llm_intent"), {"покупка", "уточнение", "запись", "гарантия"}),
            llm_category=_mode_nonempty(sub.get("llm_category"),
                                        {"Шины", "Диски", "АКБ", "Масла/Жидкости", "Услуги", "Другое"}),
            llm_category_detail=_mode_nonempty(sub.get("llm_category_detail")),
            llm_tire_rim=_mode_nonempty(sub.get("llm_tire_rim")),
            llm_season_ru=_mode_nonempty(sub.get("llm_season_ru"), set(MAP_SEASON.values())),
            llm_availability_ru=_mode_nonempty(sub.get("llm_availability_ru"), set(MAP_AVAIL.values())),

            llm_obj_detected=obj_detected_any,
            llm_obj_detected_text=_mode_nonempty(sub.get("llm_obj_detected_text"), {"да", "нет"}),
            llm_obj_type_ru=(_mode_nonempty(sub.get("llm_obj_type_ru"), set(MAP_OBJ_TYPE.values()))
                             if obj_detected_any else ""),
            llm_obj_status_ru=(_mode_nonempty(sub.get("llm_obj_status_ru"), set(MAP_OBJ_STATUS.values()))
                               if obj_detected_any else ""),
            llm_obj_ack=_any_flag(sub.get("llm_obj_ack")),
            llm_obj_ack_text=_mode_nonempty(sub.get("llm_obj_ack_text"), {"да", "нет"}),
            llm_obj_sol=_any_flag(sub.get("llm_obj_sol")),
            llm_obj_sol_text=_mode_nonempty(sub.get("llm_obj_sol_text"), {"да", "нет"}),
            llm_obj_check=_any_flag(sub.get("llm_obj_check")),
            llm_obj_check_text=_mode_nonempty(sub.get("llm_obj_check_text"), {"да", "нет"}),

            llm_next_set=_any_flag(sub.get("llm_next_set")),
            llm_next_action=_mode_nonempty(sub.get("llm_next_action")),
            llm_next_datetime=_mode_nonempty(sub.get("llm_next_datetime")),
            llm_next_place=_mode_nonempty(sub.get("llm_next_place")),
            llm_next_owner_ru=_mode_nonempty(sub.get("llm_next_owner_ru"), set(MAP_OWNER.values())),

            llm_wrap_up=_any_flag(sub.get("llm_wrap_up")),
            llm_addon_offered=_any_flag(sub.get("llm_addon_offered")),
            llm_addon_offered_text=_mode_nonempty(sub.get("llm_addon_offered_text"), {"да", "нет"}),
            llm_addon_accepted=_any_flag(sub.get("llm_addon_accepted")),
            llm_addon_accepted_text=_mode_nonempty(sub.get("llm_addon_accepted_text"), {"да", "нет"}),
            llm_addon_items=_join_items_from_json(sub.get("llm_addon_items_json", pd.Series([], dtype=str))),
        ))
        sales_name, client_name = extract_names_from_sales_strict(sub)
        sales_name_val = sales_name
        if len(sales_name) < 2:
            sales_name_val = None
        client_name_val = client_name
        if len(client_name) < 2:
            client_name_val = None
        base["named_self_sales"] = sales_name_val  # «Назвал имя» по правилам
        base["addressed_by_name_sales"] = client_name_val
        # Критерии:
        #   - Мат — по всем спикерам
        #   - Остальное — только SALES (если колонок нет — вернутся пустые строки)
        crit_cols = [
            "greeting_phrase", "found_name", "telling_name_phrases",
            "parasite_words", "stop_words", "interjections",
            "await_requests", "await_requests_exit", "working_hours",
            "inappropriate_phrases", "non_professional_phrases", "slang",
        ]
        for col in crit_cols:
            base[col + "_sales"] = aggregate_phrases_sales_only(sub, col)
        base["swear_words_all"] = aggregate_phrases_all_speakers(sub, "swear_words")

        rows.append(base)

    return pd.DataFrame(rows)


# ===== блоки для Summary =====
def compute_blocks(summary: pd.DataFrame):
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
        "Назначен шаг": _rate("llm_next_set"),
        "Есть резюме (wrap_up)": _rate("llm_wrap_up"),
        "Есть возражение": _rate("llm_obj_detected"),
        "Допродажа предложена": _rate("llm_addon_offered"),
        "Допродажа принята": _rate("llm_addon_accepted"),
        "Признание возражения": _rate("llm_obj_ack"),
        "Решение по возражению": _rate("llm_obj_sol"),
        "Проверка закрытия возраж.": _rate("llm_obj_check"),
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

    # корреляции
    corr_df = pd.DataFrame()
    for c in ["llm_next_set", "llm_wrap_up", "llm_addon_offered", "llm_addon_accepted",
              "llm_obj_detected", "llm_obj_ack", "llm_obj_sol", "llm_obj_check"]:
        if c in summary.columns:
            corr_df[c] = pd.to_numeric(summary[c], errors="coerce").fillna(0).astype(int)
    for cat in ["llm_availability_ru", "llm_intent", "llm_category"]:
        if cat in summary.columns:
            corr_df = pd.concat([corr_df, pd.get_dummies(summary[cat], prefix=cat, dtype=int)], axis=1)
    corr_matrix = corr_df.corr(method="pearson") if not corr_df.empty else pd.DataFrame()

    return total_dialogs, total_hours, theme_counts, rates, dist_blocks, corr_matrix


# ===== утилита высоты =====
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


# ===== Summary: таблицы + графики справа, без наложений =====
def _write_table_with_chart(ws, wb, start_row: int, title: str,
                            table_rows: List[Tuple[str, float]], pct: bool = False) -> int:
    fmt_header = wb.add_format({"bold": True, "bg_color": "#F2F2F2"})
    fmt_bold = wb.add_format({"bold": True})
    fmt_norm = wb.add_format({"valign": "top"})
    fmt_num = wb.add_format({"num_format": "0"})
    fmt_pct = wb.add_format({"num_format": "0.0%"})

    r = start_row
    ws.write(r, 0, title, fmt_header);
    r += 1
    ws.write(r, 0, "Показатель", fmt_bold);
    ws.write(r, 1, "Значение", fmt_bold);
    r += 1
    r_table_start = r
    for name, val in table_rows:
        ws.write(r, 0, name, fmt_norm)
        if pct:
            ws.write_number(r, 1, float(val), fmt_pct)
        else:
            ws.write_number(r, 1, float(val), fmt_num)
        r += 1
    r_table_end = r - 1

    # график справа
    if table_rows:
        ch = wb.add_chart({"type": "column"})
        ch.add_series({
            "name": title,
            "categories": ["Summary", r_table_start, 0, r_table_end, 0],
            "values": ["Summary", r_table_start, 1, r_table_end, 1],
            "data_labels": {"value": True},
        })
        if pct:
            ch.set_y_axis({"num_format": "0%", "major_gridlines": {"visible": False}})
        else:
            ch.set_y_axis({"major_gridlines": {"visible": False}})
        ch.set_title({"name": title})
        ch.set_size(CH_COL_SIZE)
        ws.insert_chart(r_table_start - 1, CHART_COL, ch, CH_COL_SETTING)

    # сдвигаем вниз ниже графика
    r = max(r, r_table_start - 1 + CHART_HEIGHT_ROWS) + 2
    return r


def write_summary_sheet(wb, summary, theme_counts, rates, dist_blocks, corr_matrix,
                        total_dialogs, total_hours):
    import re
    import numpy as np
    import pandas as pd

    ws = wb.add_worksheet("Summary")

    # ---------- Форматы
    fmt_title = wb.add_format({"bold": True, "font_size": 16})
    fmt_h2 = wb.add_format({"bold": True, "font_size": 12})
    fmt_hdr = wb.add_format({"bold": True, "bg_color": "#F2F2F2", "border": 1})
    fmt_cell = wb.add_format({"border": 1})
    fmt_wrap = wb.add_format({"border": 1, "text_wrap": True, "valign": "top"})
    fmt_num = wb.add_format({"num_format": "0"})
    fmt_pct = wb.add_format({"num_format": "0.0%"})
    fmt_dur = wb.add_format({"num_format": "0.0"})
    fmt_corr = wb.add_format({"num_format": "0.00", "border": 1})

    ws.set_default_row(22)
    ws.set_column("A:A", 22)
    ws.set_column("B:B", 22)
    ws.set_column("C:C", 22)
    ws.set_column("D:D", 22)  # разделитель
    ws.set_column("E:H", 22)
    ws.set_column("I:L", 22)

    def _is_percent_col(n: str) -> bool:
        n = str(n).lower()
        return ("доля" in n) or n.endswith("%") or ("процент" in n)

    def write_table(df: pd.DataFrame, top_row: int, left_col: int, title: str = None,
                    fit_wrap_cols: set = None) -> tuple[int, int]:
        """Выводит DataFrame с чёрными рамками у каждой ячейки.
           Если в df ровно 2 колонки — объединяет 2-ю и 3-ю ячейки каждой строки."""
        fmt_hdr_b = wb.add_format({"bold": True, "bg_color": "#F2F2F2", "border": 1})
        fmt_num_b = wb.add_format({"num_format": "0.00", "border": 1})
        fmt_pct_b = wb.add_format({"num_format": "0.0%", "border": 1})
        fmt_cell_b = wb.add_format({"border": 1})
        fmt_wrap_b = wb.add_format({"border": 1, "text_wrap": True, "valign": "top"})
        fmt_h2_b = wb.add_format({"bold": True, "font_size": 15, "border": 0})

        if title:
            ws.write(top_row, left_col, title, fmt_h2_b)
            top_row += 1

        cols = list(df.columns)
        merge_two_into_three = (len(cols) == 2)

        # Заголовок
        if merge_two_into_three:
            # 1-й заголовок в колонку 1
            ws.write(top_row, left_col + 0, cols[0], fmt_hdr_b)
            # 2-й заголовок объединён в колонки 2..3
            ws.merge_range(top_row, left_col + 1, top_row, left_col + 2, cols[1], fmt_hdr_b)
        else:
            ws.write_row(top_row, left_col, cols, fmt_hdr_b)
        top_row += 1
        start_r = top_row

        # Данные
        for _, r in df.iterrows():
            if merge_two_into_three:
                # кол1 — обычная
                v0 = r[cols[0]]
                if isinstance(v0, (int, float, np.integer, np.floating)) and not pd.isna(v0):
                    ws.write_number(top_row, left_col + 0, float(v0), fmt_num_b)
                else:
                    t0 = "" if pd.isna(v0) else str(v0)
                    ws.write(top_row, left_col + 0, t0, fmt_cell_b)

                # кол2 — объединяем (2..3)
                v1 = r[cols[1]]
                # подберём формат (процент/число/текст)
                if isinstance(v1, (int, float, np.integer, np.floating)) and not pd.isna(v1):
                    # если это «процентная» колонка — отформатируем процентом
                    use_fmt = fmt_pct_b if (cols[1].lower().find('%') != -1 or 'доля' in cols[1].lower()) else fmt_num_b
                    ws.merge_range(top_row, left_col + 1, top_row, left_col + 2, float(v1), use_fmt)
                else:
                    t1 = "" if pd.isna(v1) else str(v1)
                    use_wrap = (fit_wrap_cols and cols[1] in fit_wrap_cols) or len(t1) > 30
                    ws.merge_range(top_row, left_col + 1, top_row, left_col + 2, t1,
                                   fmt_wrap_b if use_wrap else fmt_cell_b)
            else:
                # стандартный вывод без объединений
                for j, c in enumerate(cols):
                    v = r[c]
                    if isinstance(v, (int, float, np.integer, np.floating)) and not pd.isna(v):
                        if ('доля' in str(c).lower()) or str(c).strip().endswith('%'):
                            val = float(v) if float(v) <= 1 else float(v) / 100.0
                            ws.write_number(top_row, left_col + j, val, fmt_pct_b)
                        else:
                            ws.write_number(top_row, left_col + j, float(v), fmt_num_b)
                    else:
                        txt = "" if pd.isna(v) else str(v)
                        use_wrap = (fit_wrap_cols and c in fit_wrap_cols) or len(txt) > 30
                        ws.write(top_row, left_col + j, txt, fmt_wrap_b if use_wrap else fmt_cell_b)
            top_row += 1

        end_r = top_row - 1
        return start_r, end_r

    def insert_combo_count_pct(title, start_r, end_r, cat_c, cnt_c, pct_c, below_row, left_col):
        ch = wb.add_chart({"type": "column"})
        ch.set_title({"name": title})
        ch.add_series({
            "name": "Кол-во",
            "categories": ["Summary", start_r, cat_c, end_r, cat_c],
            "values": ["Summary", start_r, cnt_c, end_r, cnt_c],
            "data_labels": {"value": True},
        })
        ln = wb.add_chart({"type": "line"})
        ln.add_series({
            "name": "Доля",
            "categories": ["Summary", start_r, cat_c, end_r, cat_c],
            "values": ["Summary", start_r, pct_c, end_r, pct_c],
            "data_labels": {"value": True, "num_format": "0%"},
            "y2_axis": True,
        })
        ch.combine(ln)
        ch.set_y2_axis({"num_format": "0%"})
        ch.set_size(CH_COL_SIZE)
        ws.insert_chart(below_row, left_col, ch, CH_COL_SETTING)

    def insert_doughnut(title, start_r, end_r, cat_c, pct_c, below_row, left_col):
        ch = wb.add_chart({"type": "doughnut"})
        ch.set_title({"name": title})
        ch.add_series({
            "categories": ["Summary", start_r, cat_c, end_r, cat_c],
            "values": ["Summary", start_r, pct_c, end_r, pct_c],
            "data_labels": {"value": True, "num_format": "0%"},
        })
        ch.set_size(CH_COL_SIZE)
        ws.insert_chart(below_row, left_col, ch, CH_COL_SETTING)

    def dist_counts(series: pd.Series, denom: int | None = None, keep_empty=False, empty_label="Не указано"):
        s = series.fillna("").astype(str).str.strip()
        if not keep_empty:
            s2 = s[s != ""]
        else:
            s2 = s.replace({"": empty_label})
        cnt = s2.value_counts()
        if cnt.empty:
            return pd.DataFrame({"Значение": [], "Кол-во": [], "Доля": []})
        d = denom if denom is not None else float(len(s2))
        out = (cnt.rename_axis("Значение").reset_index(name="Кол-во")
               .assign(Доля=lambda df: df["Кол-во"] / (d if d else 1.0)))
        return out

    # Заголовок + итоги
    row = 0
    ws.write(row, 0, "Дашборд — сводка по диалогам", fmt_title);
    row += 2
    ws.write(row, 0, "Диалогов:", fmt_h2);
    ws.write_number(row, 1, int(total_dialogs), fmt_num);
    row += 1
    ws.write(row, 0, "Суммарно, часов:", fmt_h2);
    ws.write_number(row, 1, round(total_hours, 1), fmt_dur);
    row += 1

    # ===== KPI (оставляем, как ориентир; без изменений макета)
    kpi = pd.DataFrame([
        {"Метрика": "Назначен шаг", "Значение_%": rates.get("Назначен шаг", 0.0),
         "Примечание": "Фиксация следующего шага."},
        {"Метрика": "Есть wrap-up", "Значение_%": rates.get("Есть резюме (подведение итогов)", 0.0),
         "Примечание": "Подтверждение условий."},
        {"Метрика": "Есть возражение", "Значение_%": rates.get("Есть возражение", 0.0),
         "Примечание": "Наличие возражения."},
        {"Метрика": "Допродажа предложена", "Значение_%": rates.get("Допродажа предложена", 0.0),
         "Примечание": "Дисциплина апсейла."},
        {"Метрика": "Допродажа принята", "Значение_%": rates.get("Допродажа принята", 0.0),
         "Примечание": "Ценность предложения."},
    ])
    st, en = write_table(kpi, row + 1, 0, title="KPI (с пояснениями)")
    # (не строим отдельный график здесь, чтобы экономить место)
    row = en + 2

    # ===== Интент — только круговая диаграмма (таблица + doughnut)
    intents = dist_counts(summary["llm_intent"], denom=total_dialogs, keep_empty=False)
    st_i, en_i = write_table(intents.rename(columns={"Значение": "Интент"}), row, 0, title="Распределение: Интент")
    insert_doughnut("Интент (%)", st_i, en_i, 0, 2, below_row=en_i + 1, left_col=0)

    # ===== Категория — комбо «кол-во + доля»
    cats = dist_counts(summary["llm_category"], denom=total_dialogs, keep_empty=False)
    st_c, en_c = write_table(cats.rename(columns={"Значение": "Категория"}), row, 4, title="Распределение: Категория")
    insert_combo_count_pct("Категория: кол-во + доля", st_c, en_c, 4, 5, 6, below_row=en_c + 1, left_col=4)

    # ===== Сезон (LLM) — учитываем «не указано», тоже комбо
    seasons = dist_counts(summary["llm_season_ru"], denom=total_dialogs, keep_empty=True, empty_label="Не указано")
    st_s, en_s = write_table(seasons.rename(columns={"Значение": "Сезон (LLM)"}), row, 8,
                             title="Распределение: Сезон (LLM)")
    insert_combo_count_pct("Сезон: кол-во + доля", st_s, en_s, 8, 9, 10, below_row=en_s + 1, left_col=8)

    row = max(en_i + 8, en_c + 10, en_s + 10) + 2

    # ===== Тип/Статус возражения — доли считаем от ВСЕХ ВОЗРАЖЕНИЙ
    filt_obj = summary["llm_obj_detected"].fillna(0).astype(int) > 0
    obj_total = int(filt_obj.sum()) if int(filt_obj.sum()) > 0 else 1

    obj_type = dist_counts(summary.loc[filt_obj, "llm_obj_type_ru"], denom=obj_total, keep_empty=True,
                           empty_label="Не указано")
    st_t, en_t = write_table(obj_type.rename(columns={"Значение": "Тип возражения"}), row, 0,
                             title="Распределение: Тип возражения")
    insert_combo_count_pct("Тип возражения", st_t, en_t, 0, 1, 2, below_row=en_t + 1, left_col=0)

    obj_status = dist_counts(summary.loc[filt_obj, "llm_obj_status_ru"], denom=obj_total, keep_empty=True,
                             empty_label="Не указано")
    st_ts, en_ts = write_table(obj_status.rename(columns={"Значение": "Статус возражения"}), row, 4,
                               title="Распределение: Статус возражения")
    insert_combo_count_pct("Статус возражения", st_ts, en_ts, 4, 5, 6, below_row=en_ts + 1, left_col=4)

    # «Доля и качество»: доля возражений + КОЛИЧЕСТВО handling (ACK+SOL+CLOSE суммой), а не среднее
    tmp = summary.assign(
        obj=filt_obj.astype(int),
        handling_cnt=summary[["llm_obj_ack", "llm_obj_sol", "llm_obj_check"]].fillna(0).astype(int).sum(axis=1)
    )
    obj_quality = (tmp.groupby("llm_intent", dropna=False)
                   .agg(Доля_возражений=("obj", "mean"), Количество_handling=("handling_cnt", "sum"))
                   .reset_index().rename(columns={"llm_intent": "Намерение"}))
    st_q, en_q = write_table(obj_quality, row, 8,
                             title="Возражения: доля и качество (Разрешений возражений — количеством)")
    ch_col = wb.add_chart({"type": "column"})
    ch_col.add_series({
        "name": "Количество разрешений возражений",
        "categories": ["Summary", st_q, 8, en_q, 8],
        "values": ["Summary", st_q, 10, en_q, 10],
    })
    ch_line = wb.add_chart({"type": "line"})
    ch_line.add_series({
        "name": "Доля возражений",
        "categories": ["Summary", st_q, 8, en_q, 8],
        "values": ["Summary", st_q, 9, en_q, 9],
        "data_labels": {"value": True, "num_format": "0%"},
        "y2_axis": True,
    })
    ch_col.combine(ch_line)
    ch_col.set_y2_axis({"num_format": "0%"})
    ch_col.set_size(CH_COL_SIZE)
    ws.insert_chart(en_q + 1, 8, ch_col, CH_COL_SETTING)

    row = max(en_t + 10, en_ts + 10, en_q + 10) + 2

    # ===== Сгруппированные действия next step (исправленный график: категории=0, значения=1)
    def _normalize_action(x: str) -> str:
        if not isinstance(x, str) or not x.strip():
            return "Прочее/Не указано"
        s = x.strip().lower()
        rules = [
            (r"\bоформ(ить)?\s+претензи", "Оформить претензию"),
            (r"\bзапис(ать|ь)\s+на\s+шин|\bзапись\s+на\s+шин|\bзапис(ь|ать)\b", "Запись на шиномонтаж"),
            (r"\bпереоформ", "Переоформить заказ"),
            (r"\bоформ(ить)?\s+заявк|\bзаказа?ть\b|\bоформ(ить)?\s+заказ", "Оформить заказ"),
            (r"\bзарезерв|\bрезервирован|\bрезервироват|\bзаброниров|\bрезерв\b", "Забронировать"),
            (r"\bперезвон|\bс\s+вами\s+свяжутс", "Перезвонить"),
            (r"\bсамовывоз\b|\bзабрать\b", "Самовывоз"),
            (r"\bприехат\w*", "Приехать"),
            (r"\bдостав(ка|ить|им)\b", "Доставка"),
        ]
        for pat, lab in rules:
            if re.search(pat, s):
                return lab
        return "Прочее/Не указано"

    actions_ser = summary.loc[summary["llm_next_set"].fillna(0).astype(int) > 0, "llm_next_action"].fillna("").astype(
        str)
    actions_g = actions_ser.map(_normalize_action).value_counts().rename_axis("Действие (группа)").reset_index(
        name="Кол-во")
    st_ng, en_ng = write_table(actions_g, row, 0, title="Сгруппированные действия следующих шагов")
    ch = wb.add_chart({"type": "column"})
    ch.add_series({
        "name": "Кол-во",
        "categories": ["Summary", st_ng, 0, en_ng, 0],  # кол. с категориями
        "values": ["Summary", st_ng, 1, en_ng, 1],  # кол. со значениями
        "data_labels": {"value": True},
    })
    ch.set_size(CH_COL_SIZE)
    ws.insert_chart(en_ng + 1, 0, ch, CH_COL_SETTING)

    # Доля next step по намерениям (исправленный график)
    next_by_intent = (summary.assign(n=summary["llm_next_set"].fillna(0).astype(int))
                      .groupby("llm_intent", dropna=False)["n"].mean()
                      .reset_index().rename(columns={"llm_intent": "Намерение", "n": "Доля следующих шагов"}))
    st_ni, en_ni = write_table(next_by_intent, row, 4, title="Доля назначенных следующих шагов по намерениям")
    ch2 = wb.add_chart({"type": "column"})
    ch2.add_series({
        "name": "Доля (Назначен шаг)",
        "categories": ["Summary", st_ni, 4 + 0, en_ni, 4 + 0],
        "values": ["Summary", st_ni, 4 + 1, en_ni, 4 + 1],
        "data_labels": {"value": True, "num_format": "0%"},
    })
    ch2.set_y_axis({"num_format": "0%"})
    ch2.set_size(CH_COL_SIZE)
    ws.insert_chart(en_ni + 1, 4, ch2, CH_COL_SETTING)

    row = max(en_ng + 10, en_ni + 10) + 2

    # Доля wrap-up по намерениям (исправленный график; без тотала)
    wrap_by_intent = (
        summary.assign(w=summary["llm_wrap_up"].fillna(0).astype(int))
        .groupby("llm_intent", dropna=False)
        .agg(Количество=("w", "sum"), Доля=("w", "mean"))
        .reset_index()
        .rename(columns={"llm_intent": "Намерение", "Доля": "Доля подведенний итогов"})
    )

    # таблица с количеством и долей
    st_wi, en_wi = write_table(wrap_by_intent, row, 7, title="Подведенние итогов по намерениям")

    # график: столбцы = количество, линия = доля
    ch = wb.add_chart({"type": "column"})
    ch.set_title({"name": "Подведенние итогов по намерениям"})

    # количество wrap-up
    ch.add_series({
        "name": "Количество подведенний итогов",
        "categories": ["Summary", st_wi, 7, en_wi, 7],
        "values": ["Summary", st_wi, 8, en_wi, 8],
        "data_labels": {"value": True},
    })

    # доля wrap-up
    line = wb.add_chart({"type": "line"})
    line.add_series({
        "name": "Доля подведенний итогов",
        "categories": ["Summary", st_wi, 7, en_wi, 7],
        "values": ["Summary", st_wi, 9, en_wi, 9],
        "data_labels": {"value": True, "num_format": "0%"},
        "y2_axis": True,
    })

    # объединяем графики
    ch.combine(line)
    ch.set_y2_axis({"num_format": "0%"})
    ch.set_y_axis({"major_gridlines": {"visible": False}})
    ch.set_size(CH_COL_SIZE)
    ws.insert_chart(en_wi + 1, 7, ch, CH_COL_SETTING)

    # ===== Принята допродажа + абсолюты + график (минимальные правки блока) =====
    # ===== Допродажи по типам items: частоты, доли и конверсия =====
    import json
    import re

    def _parse_items(cell) -> list[str]:
        """Возвращает список типов допродажи из ячейки.
        Приоритет: JSON (llm_addon_items_json) → строка с запятыми (llm_addon_items)."""
        if cell is None:
            return []
        # если уже list
        if isinstance(cell, list):
            return [str(x).strip() for x in cell if str(x).strip()]
        s = str(cell).strip()
        if not s or s.lower() in {"nan", "none", "null", '[]'}:
            return []
        # пробуем json
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()]
        except Exception:
            pass
        # иначе — парсим как строку с разделителями
        parts = [p.strip() for p in re.split(r"[;,/|]+", s) if p and p.strip()]
        return parts

    # маски
    offered_mask = summary.get("llm_addon_offered", pd.Series([], dtype=int)).fillna(0).astype(int) > 0
    accepted_mask = summary.get("llm_addon_accepted", pd.Series([], dtype=int)).fillna(0).astype(int) > 0
    total_dlgs = int(summary["dialog_id"].nunique()) if "dialog_id" in summary.columns else max(1, len(summary))

    # извлекаем items (по диалогу)
    items_json_col = summary.get("llm_addon_items_json", pd.Series([], dtype=str))
    items_str_col = summary.get("llm_addon_items", pd.Series([], dtype=str))
    items_lists = []
    for j, s in zip(items_json_col.reindex(summary.index, fill_value=""),
                    items_str_col.reindex(summary.index, fill_value="")):
        parsed = _parse_items(j if str(j).strip() not in {"", "nan", "null"} else s)
        # уникальные типы в одном диалоге, чтобы не раздуть счётчик
        items_lists.append(sorted(set(parsed)))
    items_lists_norm = [normalize_addon_items(lst) for lst in items_lists]
    items_df = pd.DataFrame({
        "items": items_lists_norm,
        "offered": offered_mask.values if len(offered_mask) == len(summary) else False,
        "accepted": accepted_mask.values if len(accepted_mask) == len(summary) else False,
    })
    # разворачиваем
    expl_all = items_df.explode("items", ignore_index=True).dropna(subset=["items"])

    if expl_all.empty:
        items_stats = pd.DataFrame(
            {"Тип допродажи": [], "Кол-во диалогов (всего)": [], "Доля от всех": [],
            "Кол-во (предложено)": [], "Кол-во (принято)": [], "Конверсия принятия": []}
        )
    else:
        # по типам
        cnt_all = expl_all.groupby("items")["items"].count().rename("cnt_all")
        cnt_offered = expl_all.loc[expl_all["offered"]].groupby("items")["items"].count().rename("cnt_offered")
        cnt_accepted = expl_all.loc[expl_all["accepted"]].groupby("items")["items"].count().rename("cnt_accepted")

        items_stats = (
            pd.concat([cnt_all, cnt_offered, cnt_accepted], axis=1).fillna(0).astype(int).reset_index()
            .rename(columns={"items": "Тип допродажи",
                             "cnt_all": "Кол-во диалогов (всего)",
                             "cnt_offered": "Кол-во (предложено)",
                             "cnt_accepted": "Кол-во (принято)"})
            .assign(
                **{
                    "Доля от всех": lambda d: d["Кол-во диалогов (всего)"] / (total_dlgs if total_dlgs else 1),
                    "Конверсия принятия": lambda d: d.apply(
                        lambda r: (r["Кол-во (принято)"] / r["Кол-во (предложено)"]) if r[
                                                                                            "Кол-во (предложено)"] > 0 else 0.0,
                        axis=1
                    )
                }
            )
            .sort_values(["Кол-во (предложено)", "Кол-во (принято)"], ascending=False, kind="mergesort")
        )
    items_stats = items_stats[items_stats["Кол-во диалогов (всего)"] >= 2]

    # вывод таблицы
    st_it, en_it = write_table(
        items_stats,
        row, 0,
        title="Допродажи по типам: частоты, доли и конверсия",
        fit_wrap_cols={"Тип допродажи"}
    )

    # график ПОД таблицей:
    #   — столбцы: «Кол-во (предложено)» и «Кол-во (принято)»
    #   — линия на вторую ось: «Конверсия принятия»
    if not items_stats.empty:
        ch_col = wb.add_chart({"type": "column"})
        ch_col.set_title({"name": "Допродажи по типам: абсолюты и конверсия"})
        # предложено
        ch_col.add_series({
            "name": "Предложено",
            "categories": ["Summary", st_it, 0, en_it, 0],  # Тип допродажи
            "values": ["Summary", st_it, 2, en_it, 2],  # Кол-во (предложено)
            "data_labels": {"value": True},
        })
        # принято
        ch_col.add_series({
            "name": "Принято",
            "categories": ["Summary", st_it, 0, en_it, 0],
            "values": ["Summary", st_it, 3, en_it, 3],  # Кол-во (принято)
            "data_labels": {"value": True},
        })

        # линия: конверсия
        ch_line = wb.add_chart({"type": "line"})
        ch_line.add_series({
            "name": "Конверсия принятия",
            "categories": ["Summary", st_it, 0, en_it, 0],
            "values": ["Summary", st_it, 5, en_it, 5],  # Конверсия принятия
            "data_labels": {"value": True, "num_format": "0%"},
            "y2_axis": True,
        })
        ch_col.combine(ch_line)
        ch_col.set_y_axis({"major_gridlines": {"visible": False}})
        ch_col.set_y2_axis({"num_format": "0%"})
        ch_col.set_size({"width": 955, "height": 600})
        # строго ПОД таблицей, подвиньте левый столбец при необходимости
        ws.insert_chart(en_it + 1, 0, ch_col, {"x_scale": 1.0, "y_scale": 1.0})

    # отступ для следующего блока
    row = en_it + 22

    # ===== Корреляции: «Сезон указан» — по НЕПУСТЫМ строкам
    bin_cols = [
        ("Следующий шаг зафиксирован", "llm_next_set"),
        ("Сделано подведение итогов", "llm_wrap_up"),
        ("Есть возражение", "llm_obj_detected"),
        ("Признали возражение (ack)", "llm_obj_ack"),
        ("Предложили решение (sol)", "llm_obj_sol"),
        ("Задали контрольный вопрос (close)", "llm_obj_check"),
        ("Предложена допродажа", "llm_addon_offered"),
        ("Принята допродажа", "llm_addon_accepted"),
        ("Сезон указан", None),
        ("Товар в наличии", None),
    ]
    corr_df = pd.DataFrame()
    for ru, col in bin_cols:
        if col:
            corr_df[ru] = summary.get(col, pd.Series([], dtype=int)).fillna(0).astype(int)
        else:
            if ru == "Сезон указан":
                corr_df[ru] = summary.get("llm_season_ru", pd.Series([], dtype=str)).astype(str).str.strip().ne(
                    "").astype(int)
            elif ru == "Товар в наличии":
                corr_df[ru] = summary.get("llm_availability_ru", pd.Series([], dtype=str)).astype(str).str.lower().eq(
                    "в наличии").astype(int)
    cm = corr_df.corr(numeric_only=True).round(3)
    cm.insert(0, "Метрика", cm.index)
    st_corr, en_corr = write_table(cm.reset_index(drop=True), row, 0, title="Корреляции бинарных метрик")
    if cm.shape[1] > 1:
        ws.conditional_format(st_corr, 1, en_corr, cm.shape[1] - 1, {
            "type": "3_color_scale",
            "min_color": "#F8696B", "mid_color": "#FFEB84", "max_color": "#63BE7B"
        })
    row = en_corr + 2

    # формируем автоматические выводы
    if not cm.empty:
        pairs = []
        cols = [c for c in cm.columns if c != "Метрика"]
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a, b = cols[i], cols[j]
                val = float(cm.loc[a, b])
                pairs.append((abs(val), val, a, b))
        pairs.sort(reverse=True)
        top_k = pairs[:8]

        # текстовый вывод
        lines = ["Выводы по корреляциям:",
                 "— Метрики с сильной взаимосвязью (|r|):"]
        for _, v, a, b in top_k:
            sign = "↑" if v > 0 else "↓"
            lines.append(f"{sign} r={v:+.2f}  →  {a} ↔ {b}")
        lines.append("")
        lines.append("Чем ближе |r| к 1, тем чаще метрики встречаются вместе.")
        lines.append("Отрицательная корреляция (↓) означает, что события редки в сочетании.")
        text_out = "\n".join(lines)

        # вставляем вывод под таблицей
        ws.insert_textbox(row, 0, text_out, {
            "width": 500,
            "height": 220,
            "font": {"size": 10},
            "fill": {"color": "#FFF7E6"},
            "line": {"color": "#E2B66A"},
        })
        row += 8
    # ===== Критерии: доля и количество
    crit_map = [
        ("Приветствие", "greeting_phrase_sales"),
        ("Сейлз представился", "named_self_sales"),
        ("Обращение по имени", "addressed_by_name_sales"),
        ("Слова-паразиты", "parasite_words_sales"),
        ("Вход в ожидание", "await_requests_sales"),
        ("Стоп-слова", "stop_words_sales"),
        ("Неприемлемые", "inappropriate_phrases_sales"),
        ("Выход из ожидания", "await_requests_exit_sales"),
        ("Режим работы", "working_hours_sales"),
        ("Непроф. фразы", "non_professional_phrases_sales"),
        ("Мат (учтены не только у сейлзов)", "swear_words_all"),
        ("Сленг", "slang_sales"),
    ]
    rows = []
    total = float(total_dialogs)
    for ru, col in crit_map:
        s = summary.get(col)
        if s is None:
            cnt = 0
        else:
            cnt = s.fillna("").astype(str).str.strip().replace({"nan": ""}).ne("").sum()
        rows.append({"Критерий": ru, "Кол-во": int(cnt), "Доля": (cnt / total if total else 0.0)})
    for ru, col in [("SALES тише 95% всех", "sales_quieter_95_global"),
                    ("SALES громче 95% всех", "sales_louder_95_global")]:
        s = summary.get(col, pd.Series([], dtype=int)).fillna(0).astype(int)
        cnt = int((s > 0).sum())
        rows.append({"Критерий": ru, "Кол-во": cnt, "Доля": (cnt / total if total else 0.0)})
    crit_df = pd.DataFrame(rows)
    _ = write_table(crit_df, row, 0, title="Критерии: доля и количество")
    row += len(crit_df) + 4

    # ===== Топ фраз: слова-паразиты (наполняем из нескольких колонок), сленг, неприемлемые/непроф.
    def top_phrases(cols: list[str], title: str, left: int):
        vals = []
        for col in cols:
            s = summary.get(col, pd.Series([], dtype=str)).dropna().astype(str)
            for v in s:
                parts = [p.strip() for p in re.split(r"[;,|/\\]+", v) if p and p.strip()]
                vals.extend(parts)
        if vals:
            vc = pd.Series(vals).value_counts().head(10).rename_axis("Фраза").reset_index(name="Кол-во")
        else:
            vc = pd.DataFrame({"Фраза": [], "Кол-во": []})
        write_table(vc, row, left, title=title)

    top_phrases(["parasite_words_sales", "parasite_words", "interjections_sales", "interjections"],
                "Топ: слова-паразиты", 0)
    top_phrases(["inappropriate_phrases_sales"], "Топ: неприемлемые фразы", 4)
    top_phrases(["non_professional_phrases_sales", ], "Топ: непрофессиональные фразы", 8)

    ws.freeze_panes(1, 0)
    ws.set_paper(9)
    ws.set_landscape()


# ===== Диалоги (со строкой-раскрытием) =====
def write_dialogs_sheet(wb, summary: pd.DataFrame):
    fmt_header = wb.add_format({"bold": True, "bg_color": "#F2F2F2"})
    fmt_wrap = wb.add_format({"text_wrap": True, "valign": "top"})
    fmt_norm = wb.add_format({"valign": "top"})
    fmt_bold = wb.add_format({"bold": True})
    fmt_num = wb.add_format({"num_format": "0"})
    fmt_dur = wb.add_format({"num_format": "0.0"})
    fmt_int = wb.add_format({"num_format": "0"})

    ws = wb.add_worksheet("Диалоги")
    ws.set_default_row(13)

    headers = [
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

        ("llm_obj_detected_text", "Возражение?"),
        ("llm_obj_type_ru", "Тип возражения"),
        ("llm_obj_status_ru", "Статус возражения"),
        ("llm_obj_ack_text", "Признал"),
        ("llm_obj_sol_text", "Решение"),
        ("llm_obj_check_text", "Проверка закрытия"),

        ("llm_next_set", "Назначен шаг"),
        ("llm_next_action", "Действие"),
        ("llm_next_datetime", "Дата/время"),
        ("llm_next_place", "Место"),
        ("llm_next_owner_ru", "Ответственный"),

        ("llm_wrap_up", "Резюме/итог"),
        ("llm_addon_offered_text", "Допродажа предложена"),
        ("llm_addon_accepted_text", "Допродажа принята"),
        ("llm_addon_items", "Допродажа: позиции"),

        ("swear_words_all", "Мат"),
        ("slang_sales", "Сленг"),
        ("inappropriate_phrases_sales", "Неприемлемые"),
        ("non_professional_phrases_sales", "Непроф. фразы"),
        ("await_requests_sales", "Вход в ожидание"),
        ("await_requests_exit_sales", "Выход из ожидания"),

        ("duration_sec", "Длит., с"),
        ("rows_count", "Строк"),
        ("sales_quieter_95_global", "SALES тише 95% всех"),
        ("sales_louder_95_global", "SALES громче 95% всех"),
        ("_expand", "▼"),
    ]

    for c, (_key, title) in enumerate(headers):
        ws.write(0, c, title, fmt_header)

    # ширины
    col_widths = {
        "Идентификатор диалога": 36, "Класс темы": 12, "Тема": 28, "Имя файла": 28,
        "Имя SALES": 16, "Обращение к клиенту": 18,
        "Интент": 12, "Категория": 16, "Деталь категории": 18,
        "Диаметр, дюйм": 14, "Сезон": 12, "Наличие": 12,
        "Возражение?": 12, "Тип возражения": 18, "Статус возражения": 18,
        "Признал": 10, "Решение": 10, "Проверка закрытия": 16,
        "Назначен шаг": 14, "Действие": 18, "Дата/время": 18, "Место": 16, "Ответственный": 14,
        "Резюме/итог": 14, "Допродажа предложена": 18, "Допродажа принята": 16, "Допродажа: позиции": 24,
        "Мат": 18, "Сленг": 18, "Неприемлемые": 18, "Непроф. фразы": 18, "Вход в ожидание": 18, "Выход из ожидания": 18,
        "Длит., с": 10, "Строк": 8, "SALES тише 95% всех": 18, "SALES громче 95% всех": 20, "▼": 4
    }
    for idx, (_key, title) in enumerate(headers):
        ws.set_column(idx, idx, col_widths.get(title, 16),
                      fmt_wrap if title in ("Тема", "Деталь категории", "Действие", "Дата/время", "Место",
                                            "Допродажа: позиции") else fmt_norm)

    num_keys_int = {"rows_count", "sales_quieter_95_global", "sales_louder_95_global", "llm_next_set", "llm_wrap_up"}
    num_keys_float = {"duration_sec"}

    # вывод
    row = 1
    total_chars_width = 140
    for _, rsum in summary.iterrows():
        # основная строка
        for c, (key, title) in enumerate(headers):
            v = rsum.get(key, "")
            if key in num_keys_float:
                try:
                    v = float(v) if pd.notna(v) else 0.0
                except Exception:
                    v = 0.0
                ws.write_number(row, c, v, fmt_dur)
            elif key in num_keys_int:
                try:
                    v = int(v) if pd.notna(v) else 0
                except Exception:
                    v = 0
                ws.write_number(row, c, v, fmt_int)
            elif key == "_expand":
                ws.write_url(row, c, f"internal:'Диалоги'!A{row + 1}", fmt_bold, string="↓")
            else:
                ws.write(row, c, v, fmt_wrap if title in ("Тема", "Деталь категории", "Действие", "Дата/время", "Место",
                                                          "Допродажа: позиции") else fmt_norm)
        ws.set_row(row, 13)

        # строка-деталь (полный текст), скрытая и сгруппированная
        detail_row = row + 1
        full_text = _as_str(rsum.get("dialog_text", ""))
        est_h = estimate_merged_row_height(full_text, total_chars_width, line_height_pt=14.0)
        ws.write(detail_row, 0, "Полный текст:", fmt_bold)
        ws.merge_range(detail_row, 1, detail_row, len(headers) - 1, full_text, fmt_wrap)
        ws.set_row(detail_row, max(40, min(240, est_h)), None, {"hidden": True, "level": 1})

        row += 2

    ws.autofilter(0, 0, row - 1, len(headers) - 1)
    ws.freeze_panes(1, 1)
    ws.outline_settings(True, False, True, False)


# ===== orchestrator =====
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

    # нормализация имён
    ren = {}
    for name in (TEXT_COL, THEME_COL, SPEAKER_COL, ALT_SPEAKER_COL,
                 ROWNUM_COL, START_COL, END_COL, FILE_COL, DUR_COL, LOUD_COL):
        pick = _pick_col(df, name)
        if pick in df.columns and pick != name:
            ren[pick] = name
    if DIALOG_COL not in df.columns:
        pick = _pick_col(df, DIALOG_COL)
        if pick in df.columns and pick != DIALOG_COL:
            ren[pick] = DIALOG_COL
    if ren:
        df = df.rename(columns=ren)

    if DIALOG_COL not in df.columns:
        raise ValueError(f"Нет колонки '{DIALOG_COL}' (идентификатор диалога).")

    df[DIALOG_COL] = df[DIALOG_COL].astype(str)

    # сортировка
    if ROWNUM_COL in df.columns:
        df = df.sort_values([DIALOG_COL, ROWNUM_COL], na_position="last")
    elif START_COL in df.columns and END_COL in df.columns:
        df = df.sort_values([DIALOG_COL, START_COL, END_COL], na_position="last")
    else:
        df = df.sort_values([DIALOG_COL], na_position="last")

    # расплющивание LLM
    llm_col = None
    for c in df.columns:
        if c == LLM_COL or c.startswith(LLM_COL + "."):
            llm_col = c;
            break
    if llm_col and llm_col in df.columns:
        flat_rows = []
        for v in df[llm_col]:
            obj = _safe_json_loads(v)
            flat_rows.append(llm_flat(obj) if isinstance(obj, dict) else {})
        llm_df = pd.DataFrame(flat_rows)
        df = pd.concat([df.reset_index(drop=True), llm_df.reset_index(drop=True)], axis=1)
        # бинарники к int
        for c in ["llm_obj_detected", "llm_obj_ack", "llm_obj_sol", "llm_obj_check",
                  "llm_next_set", "llm_wrap_up", "llm_addon_offered", "llm_addon_accepted"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # текст и имена
    dialog_text_map = {}
    name_sales_map, name_client_map = {}, {}
    for did, sub in df.groupby(DIALOG_COL, dropna=False):
        dialog_text_map[str(did)] = build_dialog_text_block(sub)
        ns, nc = extract_names_from_sales_strict(sub)
        name_sales_map[str(did)] = ns
        name_client_map[str(did)] = nc

    # агрегаты
    summary = aggregate_per_dialog(df)
    summary["dialog_text"] = summary["dialog_id"].map(dialog_text_map).fillna("")
    summary["name_sales"] = summary["dialog_id"].map(name_sales_map).fillna("")
    summary["name_client"] = summary["dialog_id"].map(name_client_map).fillna("")

    # метрики
    total_dialogs, total_hours, theme_counts, rates, dist_blocks, corr_matrix = compute_blocks(summary)

    # запись
    wb = xlsxwriter.Workbook(out_path)
    write_summary_sheet(wb, summary, theme_counts, rates, dist_blocks, corr_matrix,
                        total_dialogs, total_hours)
    write_dialogs_sheet(wb, summary)
    wb.close()
    return out_path


# ---------- CLI ----------
if __name__ == "__main__":
    if AudioDialogRepository is not None:
        repo = AudioDialogRepository()
        df = repo.get_all_for_report()
    else:
        df = pd.DataFrame(columns=[DIALOG_COL, TEXT_COL])
    out = make_report(df, "dialogs_report.xlsx")
    print(f"Готово: {out}")
