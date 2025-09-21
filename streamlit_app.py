# streamlit_app.py
import re
import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from textwrap import shorten

st.set_page_config(page_title="Диалог-лендинг", layout="wide")

THEME_COL = "theme"
TEXT_COL  = "row_text"

# ───────────────────────────────── Helpers ─────────────────────────────────
@st.cache_data(show_spinner=False)
def load_df(upload, theme_col: str = THEME_COL) -> tuple[pd.DataFrame, int]:
    """
    Загружает CSV/Parquet и сразу отфильтровывает строки с пустой темой.
    Пустая: NaN, пустая/пробельная строка, текстовые 'nan'/'none'/'null'.
    Возвращает (df, dropped_count).
    """
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
        mask = df[theme_col].notna() & s.ne("") & ~s.str.lower().isin(["nan", "none", "null"])
        dropped = int((~mask).sum())
        df = df.loc[mask].copy()

    return df, dropped


def ensure_columns(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def aggregate_by_dialog(df: pd.DataFrame, dialog_col: str, bool_cols: List[str]) -> pd.DataFrame:
    # Any-агрегация: max по 0/1
    agg = {c: "max" for c in bool_cols}
    # Метаданные (не агрегируем ключ)
    for meta in ["file_name", "duration", "status", "theme", "audio_dialog_fk_id"]:
        if meta == dialog_col:  # на всякий
            continue
        if meta in df.columns and meta not in agg:
            agg[meta] = "first"

    g = df.groupby(dialog_col, dropna=False, as_index=False).agg(agg)

    # Если нет готовой duration — считаем по span (min(start) → max(end))
    has_start_end = ("start" in df.columns) and ("end" in df.columns)
    if "duration" not in g.columns and has_start_end:
        dd = df.groupby(dialog_col, dropna=False).agg(start_min=("start","min"), end_max=("end","max")).reset_index()
        dd["start_min_dt"] = pd.to_datetime(dd["start_min"], errors="coerce")
        dd["end_max_dt"]   = pd.to_datetime(dd["end_max"], errors="coerce")
        if dd["start_min_dt"].notna().all() and dd["end_max_dt"].notna().all():
            dd["duration"] = (dd["end_max_dt"] - dd["start_min_dt"]).dt.total_seconds().abs()
        else:
            dd["duration"] = (pd.to_numeric(dd["end_max"], errors="coerce") - pd.to_numeric(dd["start_min"], errors="coerce")).abs()
        g = g.merge(dd[[dialog_col, "duration"]], on=dialog_col, how="left")

    # Чтобы нигде в UI не всплывал сырой theme с NaN:
    g = g.drop(columns=["theme"], errors="ignore")
    return g


def compute_dialog_stats(df: pd.DataFrame, dialog_col: str, speaker_col: str, operator_value: str | int) -> pd.DataFrame:
    needed = {dialog_col, speaker_col, "start", "end"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()

    rows = []
    sort_cols = []
    if "audio_dialog_fk_id" in df.columns:
        sort_cols.append("audio_dialog_fk_id")
    if "row_num" in df.columns:
        sort_cols.append("row_num")
    if not sort_cols:
        sort_cols = ["start", "end"]

    for did, sub in df.sort_values(sort_cols, na_position="last").groupby(dialog_col, dropna=False):
        sub = sub.copy()
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

        total_speech = float(pd.to_numeric(seg_dur, errors="coerce").fillna(0).sum())
        total_pause = max(span - total_speech, 0.0) if span else 0.0

        op_mask = sub[speaker_col].astype(str) == str(operator_value)
        op_time = float(pd.to_numeric(seg_dur[op_mask], errors="coerce").fillna(0).sum())
        cl_time = float(pd.to_numeric(seg_dur[~op_mask], errors="coerce").fillna(0).sum())

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


def build_dialog_texts(df: pd.DataFrame, dialog_col: str, speaker_col: str, text_col: str) -> dict:
    texts = {}
    sort_cols = []
    if "audio_dialog_fk_id" in df.columns:
        sort_cols.append("audio_dialog_fk_id")
    if "row_num" in df.columns:
        sort_cols.append("row_num")
    if not sort_cols:
        sort_cols = ["start", "end"]

    def lab(x):
        if pd.isna(x): return "СПИКЕР"
        return str(x)

    for did, sub in df.sort_values(sort_cols, na_position="last").groupby(dialog_col, dropna=False):
        lines = [f"**{lab(r[speaker_col])}:** {r[text_col]}" for _, r in sub.iterrows()]
        texts[str(did)] = "\n\n".join(lines)
    return texts


# ───────────── Темы: продажа / жалоба / вопрос (строгая нормализация) ─────────────
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


def detect_theme_sections_exact(df: pd.DataFrame, dialog_col: str, theme_col: str = THEME_COL) -> pd.DataFrame:
    """
    Метки:
      is_purchase  -> 'продаж*' или токен 'покупка' или англ. 'sale*'
      is_question  -> 'вопрос*'
      is_return    -> '\bвозврат\w*' или 'оформление возврата'
      is_complaint -> 'жалоб\w*' или 'претенз\w*'   (ВАЖНО: без возвратов)
    """
    if dialog_col not in df.columns or theme_col not in df.columns:
        return pd.DataFrame()

    tmp = pd.DataFrame({dialog_col: df[dialog_col], theme_col: df[theme_col]})
    rows = []
    for did, sub in tmp.groupby(dialog_col, dropna=False):
        # собрать нормализованные токены по диалогу
        toks = []
        for v in sub[theme_col]:
            toks.extend(_split_theme_tokens(v))
        uniq = [t for t in dict.fromkeys(toks) if t]

        def any_match(patterns):
            return any(any(p.search(t) for p in patterns) for t in uniq)

        sale_patterns      = [re.compile(r"продажа*", re.I)]
        question_patterns  = [re.compile(r"вопрос*", re.I)]
        return_patterns    = [re.compile(r"оформление возврата*", re.I)]
        complaint_patterns = [re.compile(r"жалоба*", re.I)]

        is_purchase = any_match(sale_patterns) or ("покупка" in uniq)
        is_question = any_match(question_patterns)
        is_return   = any_match(return_patterns)
        # жалоба теперь БЕЗ возвратов
        is_complaint = (any_match(complaint_patterns)) and (not is_return)

        rows.append({
            dialog_col: did,
            "is_purchase": 1 if is_purchase else 0,
            "is_complaint": 1 if is_complaint else 0,
            "is_question": 1 if is_question else 0,
            "is_return": 1 if is_return else 0,
            "themes_joined": ", ".join(uniq),
        })
    return pd.DataFrame(rows)


# ───────────────────────────── Sidebar & data ─────────────────────────────
st.sidebar.header("Данные")
if st.sidebar.button("♻️ Сбросить кэш"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.experimental_rerun()

upl = st.sidebar.file_uploader("CSV или Parquet", type=["csv","parquet"])
df, dropped = load_df(upl)
if df.empty:
    st.stop()
if dropped:
    st.info(f"Отфильтровано строк без темы: {dropped}")

# Автоопределение колонок
dialog_col = "audio_dialog_fk_id" if "audio_dialog_fk_id" in df.columns else st.sidebar.selectbox("Колонка диалога", df.columns)
speaker_col = "detected_speaker_id" if "detected_speaker_id" in df.columns else st.sidebar.selectbox("Колонка спикера", df.columns)
text_col = "row_text" if "row_text" in df.columns else st.sidebar.selectbox("Колонка текста", df.columns)

OPERATOR_VALUE = "SALES"

# Кандидаты булевых флагов
candidate_bool_cols = [
    "greeting_phrase","found_name","ongoing_sale","working_hours","interjections",
    "parasite_words","abbreviations","slang","telling_name_phrases","inappropriate_phrases",
    "diminutives","stop_words","swear_words","non_professional_phrases","order_offer",
    "order_processing","order_resume","await_requests","await_requests_exit","axis_attention",
    "order_type",
    "objection_processed","evaluation_offered","script_hint_present","brand_named",
    "transfer_to_other_operator","self_pickup_address_spoken","client_contacts_taken",
    "reserve_terms","delivery_terms","end_correct","made_accent_on_availability",
    "who_finished_dialog_operator","who_finished_dialog_client","interrupts_client",
    "uncertain_speech"
]
bool_cols = ensure_columns(df, candidate_bool_cols)
for c in bool_cols:
    if pd.api.types.is_bool_dtype(df[c]):
        df[c] = df[c].astype(int)
    else:
        df[c] = df[c].apply(lambda v: bool(v) and str(v).strip().lower() not in {"0","false","нет",""}).astype(int)

# Тексты диалогов
dialog_texts = build_dialog_texts(df, dialog_col, speaker_col, text_col)

# Агрегаты по диалогам
dlg_flags = aggregate_by_dialog(df, dialog_col, bool_cols)

# Время/паузы
talk_stats = compute_dialog_stats(df, dialog_col, speaker_col, OPERATOR_VALUE)
if not talk_stats.empty:
    dlg_flags = dlg_flags.merge(talk_stats, on=dialog_col, how="left")

# Темы
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

# ─────────────────────────── Компонент логов (кликабельно) ───────────────────────────
@st.cache_data(show_spinner=False)
def render_dialog_text(df: pd.DataFrame, dialog_col: str, speaker_col: str, text_col: str, dialog_id: str) -> str:
    """Собираем текст диалога по dialog_id по требованию (быстро)."""
    sub = df[df[dialog_col].astype(str) == str(dialog_id)]
    if sub.empty:
        return ""
    # Пытаемся сохранить правильный порядок фраз
    sort_cols = []
    if "audio_dialog_fk_id" in sub.columns: sort_cols.append("audio_dialog_fk_id")
    if "row_num" in sub.columns:            sort_cols.append("row_num")
    if not sort_cols:                        sort_cols = ["start", "end"]
    sub = sub.sort_values(sort_cols, na_position="last")

    def lab(x): return "СПИКЕР" if pd.isna(x) else str(x)
    # plain text — рендерится заметно быстрее, чем markdown
    lines = [f"{lab(r[speaker_col])}: {r[text_col]}" for _, r in sub.iterrows()]
    return "\n\n".join(lines)


def show_logs(filtered_df: pd.DataFrame, title: str):
    st.markdown(f"#### Логи — {title}")

    # колонки для грида (минимальный набор)
    cols_to_show = []
    for meta in ["file_name", "themes_joined", "status", "duration"]:
        if meta in filtered_df.columns:
            cols_to_show.append(meta)

    # формируем отображаемые данные
    grid_df = (
        filtered_df[[dialog_col] + cols_to_show]
        .copy()
        .drop_duplicates(subset=[dialog_col])
        .rename(columns={
            "file_name": "Файл",
            "themes_joined": "Тема",
            "status": "Статус",
            "duration": "Длительность, с",
            dialog_col: "dialog_id",
        })
    )

    # облегчение DOM: укоротим «Тема»
    if "Тема" in grid_df.columns:
        grid_df["Тема"] = grid_df["Тема"].astype(str).apply(lambda s: shorten(s, width=120, placeholder="…"))

    # быстрый поиск
    q = st.text_input("Поиск по логам", "", key=f"q_{title}")

    # AgGrid options
    gb = GridOptionsBuilder.from_dataframe(grid_df)
    gb.configure_default_column(resizable=True, sortable=True, filter=True)
    gb.configure_column("dialog_id", hide=True)
    gb.configure_selection(selection_mode="single", use_checkbox=False)
    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=100)
    gb.configure_grid_options(
        domLayout="normal",
        rowHeight=32,
        quickFilterText=q,
        animateRows=False,                 # чутка быстрее
        suppressRowClickSelection=False
    )
    go = gb.build()

    grid_res = AgGrid(
        grid_df,
        gridOptions=go,
        height=520,                # скролл
        width="100%",
        theme="balham",
        fit_columns_on_grid_load=True,
        allow_unsafe_jscode=False,
        update_mode=GridUpdateMode.SELECTION_CHANGED,   # без перерендера всей страницы
        enable_enterprise_modules=False,
        reload_data=False,
    )

    # безопасно достаём выбранную строку (в разных версиях бывает list/DF)
    sel = grid_res.get("selected_rows", [])
    # нормализуем к списку словарей
    if isinstance(sel, pd.DataFrame):
        sel = sel.to_dict(orient="records")
    elif not isinstance(sel, list):
        sel = []

    if len(sel) == 0:
        st.info("Кликните по строке, чтобы открыть диалог.")
        return

    did = str(sel[0].get("dialog_id"))

    st.markdown("---")
    st.markdown("##### Полный текст")

    # быстрый on-demand рендер текста (без глобального словаря всех диалогов)
    txt = render_dialog_text(df, dialog_col, speaker_col, text_col, did)
    if not txt:
        st.info("Текст диалога не найден.")
    else:
        st.text(txt)   # быстрее, чем markdown

# ───────────────────────────────── Pages ─────────────────────────────────
page = st.sidebar.radio("Страница", ["Обзор", "Критерии"], index=0)

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
        show_logs(orders_df, "Продажа")

    # Жалоба (без возвратов)
    with tabs[1]:
        complaints_df = dlg_flags[
            (dlg_flags.get("is_complaint", 0).fillna(0).astype(int) == 1) &
            (dlg_flags.get("is_return", 0).fillna(0).astype(int) == 0)
        ]
        st.write(f"Количество: **{len(complaints_df)}**")
        show_logs(complaints_df, "Жалоба")

    # Вопрос
    with tabs[2]:
        questions_df = dlg_flags[dlg_flags.get("is_question", 0).fillna(0).astype(int) == 1]
        st.write(f"Количество: **{len(questions_df)}**")
        show_logs(questions_df, "Вопрос")

    # Возврат
    with tabs[3]:
        returns_df = dlg_flags[dlg_flags.get("is_return", 0).fillna(0).astype(int) == 1]
        st.write(f"Количество: **{len(returns_df)}**")
        show_logs(returns_df, "Возврат")

else:
    st.title("🧩 Критерии и их доля от тотала")

    criteria_map: Dict[str, List[str]] = {
        "Поприветствовал и назвал свое имя": ensure_columns(dlg_flags, ["greeting_phrase","telling_name_phrases"]),
        "Обращение по имени (или спросил имя)": ensure_columns(dlg_flags, ["found_name"]),
        "Слова-паразиты": ensure_columns(dlg_flags, ["parasite_words"]),
        "Маты / неприемлемые фразы": ensure_columns(dlg_flags, ["swear_words","inappropriate_phrases"]),
        "Стоп-слова": ensure_columns(dlg_flags, ["stop_words"]),
        "Переходит на личности / сленг": ensure_columns(dlg_flags, ["slang","non_professional_phrases"]),
        "Предложение оформить заказ": ensure_columns(dlg_flags, ["order_offer"]),
        "Оформление заказа": ensure_columns(dlg_flags, ["order_processing"]),
        "Подведение итогов по заказу": ensure_columns(dlg_flags, ["order_resume"]),
        "Режим ожидания": ensure_columns(dlg_flags, ["await_requests"]),
        "Режим работы магазина": ensure_columns(dlg_flags, ["working_hours"]),
        "Сделал акцент на наличии / ось внимания": ensure_columns(dlg_flags, ["axis_attention"]),
        "Тип заказа / подбор по авто": ensure_columns(dlg_flags, ["order_type"]),
        "Отработано ли возражение": ensure_columns(dlg_flags, ["objection_processed"]),
        "Есть ли подсказка (скрипт доп продаж)": ensure_columns(dlg_flags, ["script_hint_present"]),
        "Предложена ли оценка": ensure_columns(dlg_flags, ["evaluation_offered"]),
        "Контакные данные клиента": ensure_columns(dlg_flags, ["client_contacts_taken"]),
        "Озвучен адрес самовывоза": ensure_columns(dlg_flags, ["self_pickup_address_spoken"]),
        "Перевод на другого оператора": ensure_columns(dlg_flags, ["transfer_to_other_operator"]),
        "Сроки резерва": ensure_columns(dlg_flags, ["reserve_terms"]),
        "Сроки доставки": ensure_columns(dlg_flags, ["delivery_terms"]),
        "Кто завершил диалог — оператор": ensure_columns(dlg_flags, ["who_finished_dialog_operator"]),
        "Кто завершил диалог — клиент": ensure_columns(dlg_flags, ["who_finished_dialog_client"]),
        "Перебивает клиента": ensure_columns(dlg_flags, ["interrupts_client"]),
        "Неуверенность в речи": ensure_columns(dlg_flags, ["uncertain_speech"]),
        "Тема: Продажа": ensure_columns(dlg_flags, ["is_purchase"]),
        "Тема: Жалоба": ensure_columns(dlg_flags, ["is_complaint"]),
        "Тема: Вопрос": ensure_columns(dlg_flags, ["is_question"]),
    }

    crit_df = pd.DataFrame({ "dialog_id": dlg_flags[dialog_col].astype(str) })
    for disp, cols in criteria_map.items():
        if not cols:
            continue
        v = np.zeros(len(dlg_flags), dtype=int)
        for c in cols:
            v = np.maximum(v, pd.to_numeric(dlg_flags[c], errors="coerce").fillna(0).astype(int).to_numpy())
        crit_df[disp] = v

    if "operator_activity_pct" in dlg_flags.columns:
        crit_df["Активность оператора, %"] = dlg_flags["operator_activity_pct"].round(1)
    if "pause_pct_of_dialog" in dlg_flags.columns:
        crit_df["Паузы в диалоге, %"] = dlg_flags["pause_pct_of_dialog"].round(1)

    share_rows = []
    exclude_cols = {"dialog_id","Активность оператора, %","Паузы в диалоге, %"}
    for disp in [c for c in crit_df.columns if c not in exclude_cols]:
        share = crit_df[disp].mean() * 100 if len(crit_df) else 0.0
        share_rows.append({"Критерий": disp, "Доля от тотала": round(share, 1)})
    summary = pd.DataFrame(share_rows).sort_values("Доля от тотала", ascending=False)
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Фильтры по критериям")

    OPTION_IGNORE = "Не учитывать"
    OPTION_T = "Только есть"
    OPTION_F = "Только нет"

    sel_state = {}
    cols = st.columns(3)
    names = [c for c in crit_df.columns if c not in {"dialog_id","Активность оператора, %","Паузы в диалоге, %"}]
    for i, name in enumerate(names):
        with cols[i % 3]:
            sel_state[name] = st.selectbox(name, options=[OPTION_IGNORE, OPTION_T, OPTION_F], index=0, key=f"sel_{name}")

    mask = pd.Series(True, index=crit_df.index)
    for name, choice in sel_state.items():
        if choice == OPTION_T:
            mask &= crit_df[name] == 1
        elif choice == OPTION_F:
            mask &= crit_df[name] == 0

    filtered_ids = set(crit_df.loc[mask, "dialog_id"].astype(str))

    st.markdown("### Логи (после фильтров)")
    logs = dlg_flags.copy()
    logs[dialog_col] = logs[dialog_col].astype(str)
    logs = logs[logs[dialog_col].isin(filtered_ids)]
    show_logs(logs, "Критерии (после фильтров)")  # тот же кликабельный компонент
