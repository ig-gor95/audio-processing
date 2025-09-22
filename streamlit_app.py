# streamlit_app.py
import re
import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from textwrap import shorten
from collections import Counter

st.set_page_config(page_title="Диалог-лендинг", layout="wide")

THEME_COL = "theme"
TEXT_COL  = "row_text"
DIALOG_COL_FALLBACK  = "audio_dialog_fk_id"
SPEAKER_COL_FALLBACK = "detected_speaker_id"
OPERATOR_VALUE = "SALES"   # оператор фиксированный

# ======== Кандидаты критериев (как в CSV) ========
ALL_BOOL_CANDIDATES = [
    "greeting_phrase","found_name","ongoing_sale","working_hours","interjections",
    "parasite_words","abbreviations","slang","telling_name_phrases","inappropriate_phrases",
    "diminutives","stop_words","swear_words","non_professional_phrases","order_offer",
    "order_processing","order_resume","await_requests","await_requests_exit","axis_attention",
    "order_type","reserve_terms","delivery_terms"
]

# «Проблемные» критерии (для частотки)
NEGATIVE_CRITERIA = [
    "parasite_words","swear_words","stop_words","slang","inappropriate_phrases",
    "non_professional_phrases","diminutives"
]

# Порядок и русские подписи для таблицы диалога
CRITERIA_DISPLAY: List[Tuple[str, str]] = [
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
    # негативная лексика
    ("parasite_words",           "Слова-паразиты"),
    ("stop_words",               "Стоп-слова"),
    ("slang",                    "Сленг"),
    ("non_professional_phrases", "Непроф. фразы"),
    ("inappropriate_phrases",    "Неприемлемые"),
    ("swear_words",              "Мат"),
    ("diminutives",              "Уменьшительные"),
]

# ───────────────────────────────── Helpers ─────────────────────────────────
@st.cache_data(show_spinner=False)
def load_df(upload, theme_col: str = THEME_COL) -> Tuple[pd.DataFrame, int]:
    """Загружает CSV/Parquet и фильтрует строки с пустой темой (NaN/пустые/‘nan’/’none’/’null’)."""
    if upload is None:
        return pd.DataFrame(), 0
    name = upload.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(upload)
    elif name.endswith(".parquet"):
        df = pd.read_parquet(upload)
    else:
        return pd.DataFrame(), 0

    dropped = 0
    if theme_col in df.columns:
        s = df[theme_col].astype(str).str.strip()
        mask = df[theme_col].notna() & s.ne("") & ~s.str.lower().isin(["nan","none","null"])
        dropped = int((~mask).sum())
        df = df.loc[mask].copy()
    return df, dropped

def ensure_columns(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]

FALSE_TOKENS = {"", "0", "нет", "false", "none", "nan", "null", "no", "off"}

def cell_to_flag(x) -> int:
    """
    Правило для строковых критериев:
    - NaN/пусто/в FALSE_TOKENS → 0
    - непустая строка → 1
    - число → 1 только если == 1
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return 0
    if isinstance(x, (int, np.integer, float, np.floating)):
        try:
            return 1 if int(float(x)) == 1 else 0
        except Exception:
            return 0
    s = str(x).strip()
    if s == "" or s.lower() in FALSE_TOKENS:
        return 0
    return 1

@st.cache_data(show_spinner=False)
def flags_line_level(df: pd.DataFrame, candidate_cols: List[str]) -> pd.DataFrame:
    """Создаёт копию df с 0/1-флагами на уровне строк (оригинал не трогаем)."""
    out = df.copy()
    for c in candidate_cols:
        if c in out.columns:
            out[c] = out[c].apply(cell_to_flag).astype(int)
        else:
            out[c] = 0
    return out

@st.cache_data(show_spinner=False)
def aggregate_by_dialog(df_flags: pd.DataFrame, dialog_col: str, bool_cols: List[str]) -> pd.DataFrame:
    """Агрегация признаков на уровень диалога (max по 0/1)."""
    agg = {c: "max" for c in bool_cols}
    for meta in ["file_name", "duration", "status", "theme", "audio_dialog_fk_id"]:
        if meta == dialog_col:
            continue
        if meta in df_flags.columns and meta not in agg:
            agg[meta] = "first"
    g = df_flags.groupby(dialog_col, dropna=False, as_index=False).agg(agg)

    # duration по span, если нет готовой
    has_start_end = ("start" in df_flags.columns) and ("end" in df_flags.columns)
    if "duration" not in g.columns and has_start_end:
        dd = df_flags.groupby(dialog_col, dropna=False).agg(
            start_min=("start","min"), end_max=("end","max")
        ).reset_index()
        dd["start_min_dt"] = pd.to_datetime(dd["start_min"], errors="coerce")
        dd["end_max_dt"]   = pd.to_datetime(dd["end_max"], errors="coerce")
        if dd["start_min_dt"].notna().all() and dd["end_max_dt"].notna().all():
            dd["duration"] = (dd["end_max_dt"] - dd["start_min_dt"]).dt.total_seconds().abs()
        else:
            dd["duration"] = (pd.to_numeric(dd["end_max"], errors="coerce")
                              - pd.to_numeric(dd["start_min"], errors="coerce")).abs()
        g = g.merge(dd[[dialog_col, "duration"]], on=dialog_col, how="left")

    return g.drop(columns=["theme"], errors="ignore")

@st.cache_data(show_spinner=False)
def compute_dialog_stats(df: pd.DataFrame, dialog_col: str, speaker_col: str, operator_value: str) -> pd.DataFrame:
    """Расчёт времени оператора/клиента/пауз на диалог."""
    needed = {dialog_col, speaker_col, "start", "end"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()
    rows = []
    sort_cols = []
    if "audio_dialog_fk_id" in df.columns: sort_cols.append("audio_dialog_fk_id")
    if "row_num" in df.columns:            sort_cols.append("row_num")
    if not sort_cols:                       sort_cols = ["start","end"]

    for did, sub in df.sort_values(sort_cols, na_position="last").groupby(dialog_col, dropna=False):
        s = pd.to_datetime(sub["start"], errors="coerce")
        e = pd.to_datetime(sub["end"], errors="coerce")
        if s.notna().all() and e.notna().all():
            seg_dur = (e - s).dt.total_seconds().clip(lower=0)
            span = float((e.max() - s.min()).total_seconds()) if len(sub) else 0.0
        else:
            s_num = pd.to_numeric(sub["start"], errors="coerce")
            e_num = pd.to_numeric(sub["end"], errors="coerce")
            seg_dur = (e_num - s_num).clip(lower=0)
            span = float((e_num.max() - s_num.min())) if len(sub) else 0.0

        op_mask = sub[speaker_col].astype(str) == str(operator_value)
        op_time = float(pd.to_numeric(seg_dur[op_mask], errors="coerce").fillna(0).sum())
        cl_time = float(pd.to_numeric(seg_dur[~op_mask], errors="coerce").fillna(0).sum())
        total_speech = float(pd.to_numeric(seg_dur, errors="coerce").fillna(0).sum())
        total_pause = max(span - total_speech, 0.0) if span else 0.0

        rows.append({
            dialog_col: did,
            "operator_time": op_time,
            "client_time": cl_time,
            "pause_time": total_pause,
            "talk_time": total_speech,
            "operator_activity_pct": (op_time / (op_time + cl_time) * 100) if (op_time + cl_time) > 0 else np.nan,
            "pause_pct_of_dialog": (total_pause / span * 100) if span > 0 else np.nan,
            "dialog_span": span,
        })
    return pd.DataFrame(rows)

# ───────────── Темы: продажа / жалоба / вопрос / возврат ─────────────
def _split_theme_tokens(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return []
    if isinstance(v, list):
        raw = v
    else:
        s = str(v)
        for sep in [";", "|", "/", "\\"]:
            s = s.replace(sep, ",")
        raw = [x for x in (t.strip() for t in s.split(",")) if x]
    toks = []
    for t in raw:
        t = t.lower().replace("ё", "е").strip(" \"'“”«»·-–—")
        t = re.sub(r"\s+", " ", t)
        toks.append(t)
    return toks

@st.cache_data(show_spinner=False)
def detect_theme_sections_exact(df: pd.DataFrame, dialog_col: str, theme_col: str = THEME_COL) -> pd.DataFrame:
    """Флаги is_purchase / is_question / is_return / is_complaint (без возвратов)."""
    if dialog_col not in df.columns or theme_col not in df.columns:
        return pd.DataFrame()
    tmp = pd.DataFrame({dialog_col: df[dialog_col], theme_col: df[theme_col]})
    rows = []
    for did, sub in tmp.groupby(dialog_col, dropna=False):
        toks = []
        for v in sub[theme_col]:
            toks.extend(_split_theme_tokens(v))
        uniq = [t for t in dict.fromkeys(toks) if t]

        def any_match(patterns): return any(any(p.search(t) for p in patterns) for t in uniq)

        sale_patterns      = [re.compile(r"продаж\w*", re.I), re.compile(r"sale\w*", re.I)]
        question_patterns  = [re.compile(r"вопрос\w*", re.I)]
        return_patterns    = [re.compile(r"\bвозврат\w*", re.I), re.compile(r"оформление\s+возврата", re.I)]
        complaint_patterns = [re.compile(r"жалоб\w*", re.I), re.compile(r"претенз\w*", re.I)]

        is_purchase  = any_match(sale_patterns) or ("покупка" in uniq)
        is_question  = any_match(question_patterns)
        is_return    = any_match(return_patterns)
        is_complaint = (any_match(complaint_patterns)) and (not is_return)

        rows.append({
            dialog_col: did,
            "is_purchase":  int(is_purchase),
            "is_complaint": int(is_complaint),
            "is_question":  int(is_question),
            "is_return":    int(is_return),
            "themes_joined": ", ".join(uniq),
        })
    return pd.DataFrame(rows)

# ───────────── Частотка «плохих фраз» из колонок критериев ─────────────
SPLIT_RE = re.compile(r"[;,|/\\]+")
TRIM_CHARS = " \"'“”«»·-–—"

@st.cache_data(show_spinner=False)
def bad_words_stats(
    df: pd.DataFrame,
    criteria_cols: List[str],
    topn: int = 40,
    speaker_col: str | None = None,
    speaker_value: str = "SALES",
) -> pd.DataFrame:
    """
    Берём СОДЕРЖИМОЕ выбранных колонок-критериев (там текст фраз),
    фильтруем строки только для заданного спикера (SALES),
    считаем частоты фраз.
    """
    use_cols = [c for c in criteria_cols if c in df.columns]
    if not use_cols:
        return pd.DataFrame(columns=["Слово/фраза", "Частота", "Критерий"])

    # строки, где в ЛЮБОЙ из колонок есть непустой текст
    any_hit = np.zeros(len(df), dtype=bool)
    for c in use_cols:
        s = df[c].astype(str).str.strip()
        hit = df[c].notna() & s.ne("") & ~s.str.lower().isin(["nan", "none", "null"])
        any_hit |= hit.to_numpy()

    if speaker_col and speaker_col in df.columns and speaker_value is not None:
        any_hit &= (df[speaker_col].astype(str) == str(speaker_value)).to_numpy()

    sub = df.loc[any_hit, use_cols].copy()
    if sub.empty:
        return pd.DataFrame(columns=["Слово/фраза", "Частота", "Критерий"])

    phrase_counter = Counter()
    per_criterion_counter = Counter()
    phrase_best_criterion: dict[str, tuple[str,int]] = {}

    for _, row in sub.iterrows():
        for c in use_cols:
            val = row.get(c)
            if pd.isna(val):
                continue
            parts = [
                re.sub(r"\s+", " ", p.strip(TRIM_CHARS).lower().replace("ё", "е"))
                for p in SPLIT_RE.split(str(val))
                if p and p.strip(TRIM_CHARS)
            ]
            if not parts:
                continue
            phrase_counter.update(parts)
            per_criterion_counter.update([f"{c}||{p}" for p in parts])

    # "основной" критерий для фразы — где она встречалась чаще всего
    for key, cnt in per_criterion_counter.items():
        crit, phrase = key.split("||", 1)
        best = phrase_best_criterion.get(phrase)
        if best is None or cnt > best[1]:
            phrase_best_criterion[phrase] = (crit, cnt)

    rows = []
    for phrase, freq in phrase_counter.most_common(topn):
        rows.append({
            "Слово/фраза": phrase,
            "Частота": freq,
            "Критерий": phrase_best_criterion.get(phrase, ("", 0))[0]
        })
    return pd.DataFrame(rows)

# ───────────── AgGrid логи → выбранный dialog_id ─────────────
def show_logs_and_get_dialog_id(filtered_df: pd.DataFrame, dialog_col: str, title: str) -> str | None:
    st.markdown(f"#### Логи — {title}")

    cols_to_show = []
    for meta in ["file_name","themes_joined","status","duration"]:
        if meta in filtered_df.columns:
            cols_to_show.append(meta)

    grid_df = (
        filtered_df[[dialog_col] + cols_to_show]
        .copy()
        .drop_duplicates(subset=[dialog_col])
        .rename(columns={
            "file_name": "Файл",
            "themes_joined":"Тема",
            "status":"Статус",
            "duration":"Длительность, с",
            dialog_col: "dialog_id"
        })
    )
    if "Тема" in grid_df.columns:
        grid_df["Тема"] = grid_df["Тема"].astype(str).apply(lambda s: shorten(s, width=120, placeholder="…"))

    q = st.text_input("Поиск по логам", "", key=f"q_{title}")

    gb = GridOptionsBuilder.from_dataframe(grid_df)
    gb.configure_default_column(resizable=True, sortable=True, filter=True)
    gb.configure_column("dialog_id", hide=True)
    gb.configure_selection(selection_mode="single", use_checkbox=False)
    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=100)
    gb.configure_grid_options(domLayout="normal", rowHeight=32, quickFilterText=q,
                              animateRows=False, suppressRowClickSelection=False)
    go = gb.build()

    grid_res = AgGrid(
        grid_df, gridOptions=go, height=520, width="100%", theme="balham",
        fit_columns_on_grid_load=True, allow_unsafe_jscode=False,
        update_mode=GridUpdateMode.SELECTION_CHANGED, enable_enterprise_modules=False, reload_data=False
    )

    sel = grid_res.get("selected_rows", [])
    if isinstance(sel, pd.DataFrame):
        sel = sel.to_dict(orient="records")
    elif not isinstance(sel, list):
        sel = []
    if not sel:
        return None
    return str(sel[0].get("dialog_id"))

# ───────────── Таблица строк диалога с критериями (✅/❌) ─────────────
def render_dialog_criteria_table(df: pd.DataFrame, dialog_id: str, dialog_col: str, speaker_col: str, text_col: str):
    """Показывает таблицу строк диалога со столбцами-критериями (✅/❌), без сплошного текста."""
    sub = df[df[dialog_col].astype(str) == str(dialog_id)].copy()
    if sub.empty:
        st.info("Нет строк для этого диалога.")
        return

    # сортировка строк диалога
    sort_cols = []
    if "audio_dialog_fk_id" in sub.columns: sort_cols.append("audio_dialog_fk_id")
    if "row_num"            in sub.columns: sort_cols.append("row_num")
    if not sort_cols:                        sort_cols = ["start","end"]
    sub = sub.sort_values(sort_cols, na_position="last")

    # базовые столбцы
    base_cols = []
    for c in ["start","end"]:
        if c in sub.columns: base_cols.append(c)
    if speaker_col in sub.columns:
        base_cols.append(speaker_col)
    if text_col in sub.columns:
        base_cols.append(text_col)

    # локально сделаем 0/1-флаги для строк
    local = flags_line_level(sub, [c for c, _ in CRITERIA_DISPLAY])

    # визуальные столбцы критериев
    for col, ru in CRITERIA_DISPLAY:
        if col in local.columns:
            local[ru] = local[col].apply(lambda v: "✅" if int(v) == 1 else "❌")

    show_cols = base_cols + [ru for _, ru in CRITERIA_DISPLAY if _ in local.columns]
    st.dataframe(local[show_cols], use_container_width=True, hide_index=True)

# ───────────────────────────── Sidebar & data ─────────────────────────────
st.sidebar.header("Данные")
if st.sidebar.button("♻️ Сбросить кэш"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

upl = st.sidebar.file_uploader("CSV или Parquet", type=["csv","parquet"])
df, dropped = load_df(upl)
if df.empty:
    st.stop()
if dropped:
    st.info(f"Отфильтровано строк без темы: {dropped}")

# Автоопределение колонок
dialog_col  = DIALOG_COL_FALLBACK  if DIALOG_COL_FALLBACK  in df.columns else st.sidebar.selectbox("Колонка диалога", df.columns)
speaker_col = SPEAKER_COL_FALLBACK if SPEAKER_COL_FALLBACK in df.columns else st.sidebar.selectbox("Колонка спикера", df.columns)
text_col    = TEXT_COL             if TEXT_COL             in df.columns else st.sidebar.selectbox("Колонка текста", df.columns)

# создаём df_flags для агрегатов (оригинальный df с текстом НЕ меняем)
df_flags = flags_line_level(df, ALL_BOOL_CANDIDATES)

# Агрегаты диалогов + статы + темы
dlg_flags = aggregate_by_dialog(df_flags, dialog_col, ensure_columns(df_flags, ALL_BOOL_CANDIDATES))
talk_stats = compute_dialog_stats(df, dialog_col, speaker_col, OPERATOR_VALUE)
if not talk_stats.empty:
    dlg_flags = dlg_flags.merge(talk_stats, on=dialog_col, how="left")
try:
    themes_mat = detect_theme_sections_exact(df, dialog_col, theme_col=THEME_COL)
    if not themes_mat.empty:
        dlg_flags = dlg_flags.merge(themes_mat, on=dialog_col, how="left")
except Exception as _e:
    st.warning(f"Не удалось построить матрицу тем: {_e}")

# Верхние метрики
n_dialogs = dlg_flags[dialog_col].nunique()
total_duration = None
if "duration" in dlg_flags.columns:
    total_duration = float(pd.to_numeric(dlg_flags["duration"], errors="coerce").fillna(0).sum())
elif "dialog_span" in dlg_flags.columns:
    total_duration = float(pd.to_numeric(dlg_flags["dialog_span"], errors="coerce").fillna(0).sum())

# ───────────────────────────────── Pages ─────────────────────────────────
page = st.sidebar.radio("Страница", ["Обзор", "Аналитика критериев"], index=0)

if page == "Обзор":
    st.title("📊 Обзор диалогов")
    c1, c2, c3 = st.columns(3)
    c1.metric("Всего диалогов", f"{n_dialogs:,}".replace(",", " "))
    if total_duration is not None:
        hours = int(total_duration // 3600)
        minutes = int((total_duration % 3600) // 60)
        c2.metric("Суммарная длительность", f"{hours} ч {minutes} мин")
    if "operator_activity_pct" in dlg_flags.columns:
        c3.metric("Ср. активность оператора", f"{dlg_flags['operator_activity_pct'].mean():.1f}%")

    st.subheader("Разделы")
    tabs = st.tabs(["Продажа", "Жалоба", "Вопрос", "Возврат"])

    # Продажа
    with tabs[0]:
        orders_df = dlg_flags[dlg_flags.get("is_purchase", 0).fillna(0).astype(int) == 1]
        st.write(f"Количество: **{len(orders_df)}**")
        did = show_logs_and_get_dialog_id(orders_df, dialog_col, "Продажа")
        if did:
            st.markdown("---")
            st.markdown("##### Диалог — строки и критерии")
            render_dialog_criteria_table(df, did, dialog_col, speaker_col, text_col)

    # Жалоба (без возвратов)
    with tabs[1]:
        complaints_df = dlg_flags[
            (dlg_flags.get("is_complaint", 0).fillna(0).astype(int) == 1) &
            (dlg_flags.get("is_return", 0).fillna(0).astype(int) == 0)
        ]
        st.write(f"Количество: **{len(complaints_df)}**")
        did = show_logs_and_get_dialog_id(complaints_df, dialog_col, "Жалоба")
        if did:
            st.markdown("---")
            st.markdown("##### Диалог — строки и критерии")
            render_dialog_criteria_table(df, did, dialog_col, speaker_col, text_col)

    # Вопрос
    with tabs[2]:
        questions_df = dlg_flags[dlg_flags.get("is_question", 0).fillna(0).astype(int) == 1]
        st.write(f"Количество: **{len(questions_df)}**")
        did = show_logs_and_get_dialog_id(questions_df, dialog_col, "Вопрос")
        if did:
            st.markdown("---")
            st.markdown("##### Диалог — строки и критерии")
            render_dialog_criteria_table(df, did, dialog_col, speaker_col, text_col)

    # Возврат
    with tabs[3]:
        returns_df = dlg_flags[dlg_flags.get("is_return", 0).fillna(0).astype(int) == 1]
        st.write(f"Количество: **{len(returns_df)}**")
        did = show_logs_and_get_dialog_id(returns_df, dialog_col, "Возврат")
        if did:
            st.markdown("---")
            st.markdown("##### Диалог — строки и критерии")
            render_dialog_criteria_table(df, did, dialog_col, speaker_col, text_col)

else:
    st.title("🧪 Аналитика критериев")

    # 1) Частотка «плохих фраз» (по значениям в выбранных колонках, только SALES)
    st.subheader("Критерии для анализа лексики")
    available_neg = [c for c in NEGATIVE_CRITERIA if c in df.columns]

    # какая колонка про спикера
    speaker_col_for_stats = (
        "speaker_id" if "speaker_id" in df.columns
        else ("detected_speaker_id" if "detected_speaker_id" in df.columns else None)
    )

    if not available_neg:
        st.info("В данных нет колонок с «проблемной» лексикой.")
    else:
        cols1, cols2, cols3 = st.columns(3)
        chosen = []
        for i, c in enumerate(available_neg):
            with [cols1, cols2, cols3][i % 3]:
                if st.checkbox(c, value=False, key=f"neg_{c}"):
                    chosen.append(c)

        if chosen:
            bw = bad_words_stats(
                df,
                criteria_cols=chosen,   # используем СОДЕРЖИМОЕ этих колонок
                topn=40,
                speaker_col=speaker_col_for_stats,
                speaker_value="SALES",  # считаем только для SALES
            )
            st.markdown("#### Топ-фразы по выбранным критериям (только SALES)")
            st.dataframe(bw, use_container_width=True, hide_index=True)
        else:
            st.info("Выберите хотя бы один критерий выше.")

    st.markdown("---")

    # 2) Доли по каждому критерию (по диалогам) — используем dlg_flags
    st.subheader("Доля диалогов с признаком (по каждому критерию)")
    crit_map: Dict[str, List[str]] = {
        "Поприветствовал и назвал имя": ensure_columns(dlg_flags, ["greeting_phrase","telling_name_phrases"]),
        "Обращение/поиск имени":        ensure_columns(dlg_flags, ["found_name"]),
        "Слова-паразиты":               ensure_columns(dlg_flags, ["parasite_words"]),
        "Маты/неприемлемые":            ensure_columns(dlg_flags, ["swear_words","inappropriate_phrases"]),
        "Стоп-слова":                   ensure_columns(dlg_flags, ["stop_words"]),
        "Личности/сленг":               ensure_columns(dlg_flags, ["slang","non_professional_phrases"]),
        "Предложение заказа":           ensure_columns(dlg_flags, ["order_offer"]),
        "Оформление заказа":            ensure_columns(dlg_flags, ["order_processing"]),
        "Резюме заказа":                ensure_columns(dlg_flags, ["order_resume"]),
        "Режим работы":                 ensure_columns(dlg_flags, ["working_hours"]),
        "Вход в ожидание":              ensure_columns(dlg_flags, ["await_requests"]),
        "Выход из ожидания":            ensure_columns(dlg_flags, ["await_requests_exit"]),
        "Тип заказа/подбор":            ensure_columns(dlg_flags, ["order_type"]),
        "Сроки резерва":                ensure_columns(dlg_flags, ["reserve_terms"]),
        "Сроки доставки":               ensure_columns(dlg_flags, ["delivery_terms"]),
        "Тема: Продажа":                ensure_columns(dlg_flags, ["is_purchase"]),
        "Тема: Жалоба":                 ensure_columns(dlg_flags, ["is_complaint"]),
        "Тема: Вопрос":                 ensure_columns(dlg_flags, ["is_question"]),
        "Тема: Возврат":                ensure_columns(dlg_flags, ["is_return"]),
    }

    dlg_share = dlg_flags[[dialog_col]].drop_duplicates().copy()
    for disp, cols in crit_map.items():
        if not cols:
            continue
        v = np.zeros(len(dlg_flags), dtype=int)
        for c in cols:
            v = np.maximum(v, pd.to_numeric(dlg_flags[c], errors="coerce").fillna(0).astype(int).to_numpy())
        tmp = pd.DataFrame({dialog_col: dlg_flags[dialog_col], disp: v})
        tmp = tmp.groupby(dialog_col, as_index=False)[disp].max()
        dlg_share = dlg_share.merge(tmp, on=dialog_col, how="left")

    share_rows = []
    for disp in [c for c in dlg_share.columns if c != dialog_col]:
        share = pd.to_numeric(dlg_share[disp], errors="coerce").fillna(0).mean() * 100 if len(dlg_share) else 0.0
        share_rows.append({"Критерий": disp, "Доля от тотала, %": round(float(share), 1)})
    summary = pd.DataFrame(share_rows).sort_values("Доля от тотала, %", ascending=False)
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown("---")

    # 3) Быстрый фильтр по критериям → логи → таблица выбранного диалога
    st.subheader("Быстрые фильтры и просмотр логов")

    # 3.1 Фильтр по спикеру (влияет и на отбор диалогов, и на рендер таблицы)
    speaker_choice = st.radio(
        "Показывать реплики спикера",
        ["SALES", "CLIENT"],
        index=0,  # по умолчанию SALES
        horizontal=True,
        key="quick_speaker_choice",
    )

    # 3.2 Выбор условий по критериям (те же русские названия, что в dlg_share/crit_map)
    names = [c for c in dlg_share.columns if c != dialog_col]
    cols = st.columns(3)
    sel = {}
    OPTION_IGNORE, OPTION_T, OPTION_F = "Не учитывать", "Только есть", "Только нет"
    for i, name in enumerate(names):
        with cols[i % 3]:
            sel[name] = st.selectbox(name, [OPTION_IGNORE, OPTION_T, OPTION_F], key=f"cf_{name}")

    # 3.3 Отбор диалогов с учётом speaker_choice и выбранных условий ПО СТРОКАМ
    # Нужна мапа: Русское имя критерия -> список внутренних колонок
    # Используем ту же `crit_map`, что ты формировал выше для dlg_share.
    row_df = df_flags.copy()  # здесь флаги 0/1 уже посчитаны на уровне строк
    if speaker_col in row_df.columns:
        row_df = row_df[row_df[speaker_col].astype(str) == speaker_choice]

    # стартовый набор — все диалоги, присутствующие после фильтра спикера
    candidate_ids = set(row_df[dialog_col].astype(str).unique())

    for disp_name, choice in sel.items():
        if choice == OPTION_IGNORE:
            continue

        cols_for_crit = crit_map.get(disp_name, [])
        cols_for_crit = [c for c in cols_for_crit if c in row_df.columns]
        if not cols_for_crit:
            # если колонок нет, этот критерий никак не влияет
            continue

        # по этому критерию «срабатывание строки» = max по колонкам критерия
        per_row_hit = np.zeros(len(row_df), dtype=int)
        for c in cols_for_crit:
            per_row_hit = np.maximum(per_row_hit,
                                     pd.to_numeric(row_df[c], errors="coerce").fillna(0).astype(int).to_numpy())

        tmp = row_df[[dialog_col]].copy()
        tmp["hit"] = per_row_hit
        per_dialog_hit = tmp.groupby(dialog_col, as_index=False)["hit"].max()

        if choice == OPTION_T:
            keep_ids = set(per_dialog_hit.loc[per_dialog_hit["hit"] == 1, dialog_col].astype(str))
            candidate_ids &= keep_ids
        elif choice == OPTION_F:
            keep_ids = set(per_dialog_hit.loc[per_dialog_hit["hit"] == 0, dialog_col].astype(str))
            candidate_ids &= keep_ids

    # 3.4 Логи + просмотр диалога
    logs = dlg_flags.copy()
    logs[dialog_col] = logs[dialog_col].astype(str)
    logs = logs[logs[dialog_col].isin(candidate_ids)]

    did = show_logs_and_get_dialog_id(logs, dialog_col, "Критерии (после фильтров)")
    if did:
        st.markdown("---")
        st.markdown("##### Диалог — строки и критерии")
        # ⬇️ показываем весь диалог (и SALES, и CLIENT), без спикер-фильтра
        render_dialog_criteria_table(
            df,  # <-- полный df, не фильтруем по speaker_choice
            did,
            dialog_col,
            speaker_col,
            text_col
        )
