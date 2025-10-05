import pandas as pd
import xlsxwriter

from report_common import estimate_merged_row_height

def write_dialogs_sheet(wb, summary: pd.DataFrame):
    fmt_header = wb.add_format({"bold": True, "bg_color": "#F2F2F2"})
    fmt_wrap   = wb.add_format({"text_wrap": True, "valign": "top"})
    fmt_norm   = wb.add_format({"valign": "top"})
    fmt_bold   = wb.add_format({"bold": True})
    fmt_int    = wb.add_format({"num_format": "0"})
    fmt_dur    = wb.add_format({"num_format": "0.0"})
    fmt_corr   = wb.add_format({"num_format": "0.00"})

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
        ("llm_obj_detected", "Возражение?"),
        ("llm_obj_type_ru", "Тип возражения"),
        ("llm_obj_status_ru", "Статус возражения"),
        ("llm_obj_score", "Оценка обр."),
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

    for c, (_key, title) in enumerate(headers):
        ws.write(0, c, title, fmt_header)

    col_widths = {
        "Идентификатор диалога": 36, "Класс темы": 12, "Тема": 28, "Имя файла": 28,
        "Имя SALES": 16, "Обращение к клиенту": 18, "Интент": 12, "Категория": 16,
        "Деталь категории": 18, "Диаметр, дюйм": 14, "Сезон": 12, "Наличие": 12,
        "Возражение?": 12, "Тип возражения": 18, "Статус возражения": 18, "Оценка обр.": 12,
        "Признал": 10, "Решение": 10, "Проверка закрытия": 16, "Назначен шаг": 14,
        "Действие": 18, "Дата/время": 18, "Место": 16, "Ответственный": 14,
        "Резюме/итог": 14, "Допродажа предложена": 18, "Допродажа принята": 16,
        "Длит., с": 10, "Строк": 8, "SALES тише 95% всех": 18, "SALES громче 95% всех": 20, "▼": 4
    }
    for idx, (_key, title) in enumerate(headers):
        ws.set_column(idx, idx, col_widths.get(title, 16),
                      fmt_wrap if title in ("Тема","Деталь категории","Действие","Дата/время","Место") else fmt_norm)

    num_keys_int = {
        "rows_count","sales_quieter_95_global","sales_louder_95_global",
        "llm_obj_detected","llm_obj_ack","llm_obj_sol","llm_obj_check",
        "llm_next_set","llm_wrap_up","llm_addon_offered","llm_addon_accepted"
    }
    num_keys_float = {"duration_sec","llm_obj_score"}

    row = 1
    total_chars_width = sum(col_widths.get(headers[idx][1], 16) for idx in range(1, len(headers)))
    for _, rsum in summary.iterrows():
        val_by_key = {k: rsum.get(k, "") for k, _ in headers}
        val_by_key["duration_sec"] = float(val_by_key.get("duration_sec") or 0.0)
        for k in num_keys_int:
            v = val_by_key.get(k); val_by_key[k] = int(pd.to_numeric(v, errors="coerce") if str(v) != "" else 0)
        for k in num_keys_float:
            v = val_by_key.get(k)
            try: val_by_key[k] = float(v) if pd.notna(v) else 0.0
            except Exception: val_by_key[k] = 0.0
        val_by_key["_expand"] = "↓"

        for c, (key, title) in enumerate(headers):
            v = val_by_key.get(key, "")
            if key in num_keys_float:
                ws.write_number(row, c, float(v), fmt_dur if key == "duration_sec" else fmt_corr)
            elif key in num_keys_int:
                ws.write_number(row, c, int(v), fmt_int)
            elif key == "_expand":
                ws.write_url(row, c, f"internal:'Диалоги'!A{row + 2}", fmt_bold, string="↓")
            else:
                ws.write(row, c, v, fmt_wrap if title in ("Тема","Деталь категории","Действие","Дата/время","Место") else fmt_norm)

        ws.set_row(row, 12)

        # скрытая строка «Полный текст»
        detail_row = row + 1
        ws.write(detail_row, 0, "Полный текст:", fmt_bold)
        dialog_text = str(rsum.get("dialog_text", "") or "")
        est_h = estimate_merged_row_height(dialog_text, total_chars_width, line_height_pt=14.0)
        ws.merge_range(detail_row, 1, detail_row, len(headers) - 1, dialog_text, fmt_wrap)
        ws.set_row(detail_row, est_h, None, {"hidden": True, "level": 1})

        row += 2

    ws.freeze_panes(1, 1)
    ws.autofilter(0, 0, row - 1, len(headers) - 1)
    ws.outline_settings(True, False, True, False)
