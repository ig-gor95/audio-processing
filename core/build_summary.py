import pandas as pd
import numpy as np
import xlsxwriter

from report_common import (
    THEME_COL, DIALOG_COL, FILE_COL, DUR_COL, START_COL, END_COL, ROWNUM_COL,
    LLM_COL, make_unique_columns, _safe_json_loads, llm_flat, _mode_nonempty,
    _any_flag, _median_num, classify_theme_by_rules, sales_mean_for_dialog,
    aggregate_criterion_values_sales_only, MAP_AVAIL, MAP_SEASON, MAP_OBJ_TYPE,
    MAP_OBJ_STATUS, MAP_OWNER
)

# Параметры раскладки графиков
CHART_ROWS = 22   # "высота" чарта в строках для расчёта смещения вниз
GAP_ROWS   = 2

def aggregate_per_dialog(df: pd.DataFrame) -> pd.DataFrame:
    present_criteria = [c for c in df.columns if c in {
        "greeting_phrase","telling_name_phrases","found_name","ongoing_sale","order_offer",
        "order_processing","order_resume","working_hours","reserve_terms","delivery_terms",
        "axis_attention","await_requests","await_requests_exit","parasite_words","stop_words",
        "slang","non_professional_phrases","inappropriate_phrases","swear_words","order_type"
    }]

    rows = []
    for did, sub in df.groupby(DIALOG_COL, dropna=False):
        themes = sub.get(THEME_COL, pd.Series([], dtype=str)).astype(str).str.strip()
        themes = [t for t in themes if t and t.lower() not in {"nan","none","null"}]
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

        base = dict(
            dialog_id=str(did),
            theme_class=theme_class,
            theme=theme_joined,
            file_name=str(sub.get(FILE_COL, pd.Series([""])).iloc[0]) if FILE_COL in sub.columns else "",
            duration_sec=duration,
            rows_count=int(len(sub)),
            dialog_text=""  # наполняется в main (чтобы не тянуть сюда текстовые utils)
        )

        for c in present_criteria:
            base[c] = aggregate_criterion_values_sales_only(sub, c)

        # агрегаты LLM
        base.update(dict(
            llm_intent=_mode_nonempty(sub.get("llm_intent"), {"покупка","уточнение","запись","гарантия"}),
            llm_category=_mode_nonempty(sub.get("llm_category"), {"Шины","Диски","АКБ","Масла/Жидкости","Услуги","Другое"}),
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

    return pd.DataFrame(rows)

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
        if s.empty: return pd.Series(dtype=float)
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

    return total_dialogs, total_hours, theme_counts, rates, dist_blocks, corr_matrix

def write_summary_sheet(wb, summary, theme_counts, rates, dist_blocks, corr_matrix,
                        total_dialogs, total_hours):
    fmt_header = wb.add_format({"bold": True, "bg_color": "#F2F2F2"})
    fmt_wrap   = wb.add_format({"text_wrap": True, "valign": "top"})
    fmt_norm   = wb.add_format({"valign": "top"})
    fmt_bold   = wb.add_format({"bold": True})
    fmt_num    = wb.add_format({"num_format": "0"})
    fmt_dur    = wb.add_format({"num_format": "0.0"})
    fmt_pct    = wb.add_format({"num_format": "0.0%"})
    fmt_corr   = wb.add_format({"num_format": "0.00"})

    ws = wb.add_worksheet("Сводка")
    ws.set_default_row(13)

    r = 0
    ws.write(r, 0, "Сводка по диалогам", fmt_header); r += 1
    ws.write(r, 0, "Диалогов:", fmt_bold);        ws.write_number(r, 1, int(total_dialogs), fmt_num); r += 1
    ws.write(r, 0, "Суммарно, часов:", fmt_bold); ws.write_number(r, 1, round(total_hours, 1), fmt_dur); r += 2

    # Блок: Итоги по LLM
    ws.write(r, 0, "Итоги по LLM-критериям", fmt_header); r += 1
    start_rates = r
    ws.write(r, 0, "Показатель", fmt_bold); ws.write(r, 1, "Доля", fmt_bold); r += 1
    for k, v in rates.items():
        ws.write(r, 0, k, fmt_norm); ws.write_number(r, 1, float(v), fmt_pct); r += 1
    end_rates = r - 1

    # график справа от таблицы (и отступ вниз по высоте чарта)
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
        ws.insert_chart(start_rates, 3, ch, {"x_scale": 1.15, "y_scale": 1.1})
    r = max(end_rates + GAP_ROWS, start_rates + CHART_ROWS + GAP_ROWS)

    # Блок: Темы
    ws.write(r, 0, "Диалоги по темам", fmt_header); r += 1
    ws.write(r, 0, "Тема", fmt_bold); ws.write(r, 1, "Кол-во", fmt_bold); r += 1
    start_themes = r
    for rec in theme_counts.itertuples(index=False):
        ws.write(r, 0, rec.Тема or "", fmt_norm)
        ws.write_number(r, 1, int(rec.Диалогов), fmt_num)
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
        ws.insert_chart(start_themes, 3, ch_t, {"x_scale": 1.15, "y_scale": 1.05})
    r = max(end_themes + GAP_ROWS, start_themes + CHART_ROWS + GAP_ROWS)

    # Блоки распределений (основные)
    for block_name in ["Интент", "Категория", "Наличие (LLM)", "Статус возражения"]:
        ser = dist_blocks.get(block_name, pd.Series(dtype=float))
        ws.write(r, 0, block_name, fmt_header); r += 1
        ws.write(r, 0, "Значение", fmt_bold); ws.write(r, 1, "Доля", fmt_bold); r += 1
        start_blk = r
        if ser.empty:
            ws.write(r, 0, "— нет данных —", fmt_norm); r += 2
            end_blk = r - 1
        else:
            for idx, (val, pct) in enumerate(ser.items()):
                if idx >= 8: break
                ws.write(r, 0, str(val), fmt_norm)
                ws.write_number(r, 1, float(pct), fmt_pct)
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
        ws.insert_chart(start_blk, 3, ch_b, {"x_scale": 1.15, "y_scale": 1.05})

        # двигаемся ниже, учитывая реальную высоту чарта
        r = max(end_blk + GAP_ROWS, start_blk + CHART_ROWS + GAP_ROWS)

    # Корреляции (тепло-карта)
    ws.write(r, 0, "Корреляции LLM-показателей", fmt_header); r += 1
    if not corr_matrix.empty:
        ws.write(r, 0, "", fmt_header)
        for j, col in enumerate(corr_matrix.columns, start=1):
            ws.write(r, j, col, fmt_header)
        for i, row_name in enumerate(corr_matrix.index, start=1):
            ws.write(r + i, 0, row_name, fmt_header)
            for j, col_name in enumerate(corr_matrix.columns, start=1):
                ws.write_number(r + i, j, float(corr_matrix.loc[row_name, col_name]), fmt_corr)
        # цветовая шкала
        ws.conditional_format(r + 1, 1, r + len(corr_matrix.index), 1 + len(corr_matrix.columns), {
            "type": "3_color_scale", "min_value": -1, "max_value": 1
        })
        r = r + len(corr_matrix.index) + GAP_ROWS
    else:
        ws.write(r, 0, "Недостаточно данных для расчёта корреляций.", fmt_norm); r += 1

    # ширины + «якорь» для скролла
    ws.set_column(0, 0, 28, fmt_norm)
    ws.set_column(1, 2, 14, fmt_norm)
    for c in range(3, 12):
        ws.set_column(c, c, 24)

    # принудительно увеличим used range, чтобы скролл точно работал
    ws.write(r + 30, 0, "")  # «якорь» внизу
