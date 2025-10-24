from log_utils import setup_logger
from saiga import SaigaClient

logger = setup_logger(__name__)
saiga = SaigaClient()


def resolve_theme(dialog_text: str) -> str:
    try:
        return saiga.ask(
            f"""
            Я дам тебе диалог. После фразы НАЧАЛО ДИАЛОГА начнется диалог. Завершится словом КОНЕЦ ДИАЛОГА.
            Нужно в ответе вывести только тему этого диалога или темы через запятую в одну строку из предложенных ниже:
            Возможные темы для ответа:
            Если клиент оформил заказ или бронь, можно перечислить через запятую:
             - продажа шин
             - продажа услуг шиномонтажа
             - продажа услуг установки дисков
             - продажа услуг по замене масла
             - продажа дисков
             - продажа колес
             - продажа аккумулятора
             - продажа покрышек
             - оформление возврата
            Если клиент задал вопросы, то выбери ответ из списка, можно перечислить через запятую:
            - вопрос по доставке
            - вопрос по оформлению
            - вопрос по шинам
            - вопрос по колеса
            - вопрос по дискам
            - вопрос по возврату
            - вопрос по аккумулятору
            - вопрос по шиномонтажу
            - вопрос по установке дисков
            - вопрос по замене масла
            - вопрос по гарантии
            - вопрос по заказу
            - вопрос по хранению колес
            Если у клиента жалоба - то жалоба на [тема жалобы]
            Нельзя выводить что либо, не указанное в примерах выше
            
            НАЧАЛО ДИАЛОГА
            {{dialog_text}}
            КОНЕЦ ДИАЛОГА
            Строго обязательно: в ответе нужно вывести результат тз предложенных тем
            Ответ должен быть максимально коротким, ничего кроме предлоденных вариантов перечисленных выше выводить запрещено
            """
        )
    except Exception as e:
        logger.info(e)
        raise e


# def resolve_llm_data(dialog_text: str) -> str:
#     try:
#         return saiga.ask(
#             f"""
#                 Ты — детерминированный парсер входящих звонков (авто-ритейл).Верни ТОЛЬКО валидный JSON по схеме ниже. Если данных нет — null/unknown/uncertain. Без домыслов.
#             ВходTODAY: {{{{YYYY-MM-DD | optional}}}}<<<DIALOGUE_START{{dialog_text}}DIALOGUE_END>>>
#             ASR может ошибаться — восстанавливай смысл, не выдумывай.
#             0) Категория и намерение
#             Категория: "Шины" | "Диски" | "АКБ" | "Масла/Жидкости" | "Услуги" | "Другое".category_detail (если уместно): Масла/Жидкости → "масло" | "антифриз/ОЖ" | "ATF" | "тормозная жидкость" | "омывайка"; Услуги → "шиномонтаж" | "замена масла" | "диагностика" | "доставка" | "прочее".Если в диалоге есть и покупка товара, и запись на работу по нему → intent="покупка" (запись отразить в entities/service и/или next_step). intent="запись" — только если товара нет.
#             1) Нормализация (кратко)
#             Коды склеивать: 714 037 700 → "714037700".Шины: 175/70R14 | 175 70 14 | 175 на 70 d14 | R14 175/70 → size:{{{{width:175,aspect:70,rim:14}}}}. Индексы: \b(\d{{{{2,3}}}})([A-Z])\b.Масла/ОЖ: grade, volume_l, spec(G11..G13), color — только если явно.Деньги рядом с руб|₽|р|цена → price_rub:int (диапазоны → [min,max]).Даты/время: "YYYY-MM(-DD)", "HH:MM".
#             1a) Сезон шин (строго)
#             tire.season = "летняя"|"зимняя"|"всесезонная" только если:• прямое слово/AS/«круглый год»; или зимние маркеры (шипы/липучка/3PMSF/Nordic/Ice/Frict); или летние маркеры («на лето/летняя»);• либо сезонная переобувка + явный период: март–май → летняя; окт–дек → зимняя (используй дату из диалога/TODAY).Если не прямое слово — добавь entities → feature/season_cue с attrs.maps_to.
#             2) Базовые критерии
#             intent ∈ "покупка" | "уточнение" | "запись" | "гарантия" (приоритет: гарантия > запись > покупка > уточнение).required_params_collected — true, если заполнено ≥80% ключевых полей сценария (Шины/АКБ/МЖ/Гарантия — как раньше).solution_concrete — продажа/уточнение: stated_availability ∈ {{{{in_stock|backorder|unknown}}}} И ( price_rub ИЛИ price_range_rub ).next_step_set — true только при одновременном наличии owner и action (желательно place/date/time).contacts_ok — true, если есть contacts.name (телефон не обязателен).
#             3) Возражения и жалобы (усилено)
#             Триггеры недовольства/жалоб (лови хотя бы одно):слова/фразы: жалоба, претензия, не устраивает, надоело, не могу/невозможно, больше не буду покупать, лапшу на уши, обман, напишу заявление, свяжите с руководством, угрозы/эскалации; сильная ругань/брань; явное разочарование по срокам/процессу/доступности/цене.
#             Для КАЖДОГО случая заполни objection.list[]:
#             * type: price|availability|time_sla|trust|warranty_claim|fitment|location_logistics|process
#                 * «далеко/нет записи завтра/только через неделю» → time_sla
#                 * «банк/рассрочка/анкета/процесс оформления» → process
#                 * «обманываете/лапшу/не верю» → trust
#             * quote (≤25 слов, дословно), sales_response (кратко),
#             * steps: {{{{acknowledge, solution_or_argument, check_close}}}} — true/false.
#                 * Примеры «acknowledge»: понимаю/извиняюсь/давайте разберёмся
#                 * «solution_or_argument»: конкретное действие/альтернатива (переоформим заказ; обращайтесь в банк; запись на...)
#                 * «check_close»: это решает вопрос? всё устраивает?
#             * resolution_status: "resolved"|"unresolved"|"ignored".
#             * reasoning_short: 1–2 фразы.objection.detected = true, если objection.list не пуст.handling_score (0–3) — по лучшему кейсу: 0 игнор; 1 признал; 2 признал+решение/альтернатива; 3 признал+решение+проверка снятия.
#             4) Подведение итогов (soft wrap-up) + допродажа + исход продажи
#             wrap_up.closing_ok = true, если выполнены ≥3 из:repeated_items | unit_price_announced | total_price_announced | fulfillment_method | address_or_store | working_hours | delivery_terms | storage_terms | service_offer(может быть declined) | follow_up_sla | for_legal_entities_docs | consumables_notice | storage_contract_notice.Короткие подтверждающие цитаты положи в wrap_up.evidence_quotes (1–3).Допродажа:wrap_up.add_on_sale = {{{{offered, accepted, items[], total_price_rub, evidence_quotes[]}}}} (только по явным фразам).Исход продажи:wrap_up.sale_outcome = "won|pending|lost|unknown" + sale_outcome_reason (например: "finance_process"|"time_sla"|"no_stock"|"client_refusal"|"abusive_language"|"other").
#             * lost — клиент отказывается/уходит/обрыв из-за конфликта/бранной лексики/эскалации без решения.
#             * pending — ждут внешнее действие (ответ банка, перезвон и т. п.).
#             Если intent ∈ {{{{покупка, запись}}}} и closing_ok=false → red_flag no_wrap_up.
#             5) Красные флаги
#             Добавляй при выполнении условий:no_phone_for_appointment | appointment_in_past | no_warranty_sla | bring_list_missing | warranty_facts_missing | size_incomplete | indices_missing | route_contradiction | no_availability_price | no_next_step | no_contacts | no_wrap_up | abusive_language | escalation_requested | complaint_filed
#             * abusive_language — явная нецензурная лексика/оскорбления.
#             * escalation_requested — «свяжите/соедините с руководством», «подам жалобу» и т. п.
#             * complaint_filed — оператор подтвердил, что оформил претензию/обращение.
#             6) Авто-сущности (только явные факты)
#             Любая entities[i] обязана иметь evidence_quote (≤20 слов) и по возможности char_span.etype: "product"|"service"|"store"|"location"|"brand"|"code"|"quantity"|"price"|"feature"|"time"|"contact"|"other".Сезонные выводы — добавляй feature/season_cue с attrs.maps_to.Qty — только штуки; литры/кг в attrs.
#             Evidence-lock
#             Не ставь НЕ-null для: next_step.*, contacts.*, slots.tire.size.* / load_index / speed_index / season, slots.solution.stated_availability / price_*, category_detail — без прямой цитаты (и/или feature season_cue для season).Связь entities → slots: нет подтверждающей сущности — не заполняй связанные поля.Дата/время/адрес — только явно сказанные.
#
#             Выходной JSON (строго эта структура)
#
#             {{{{
#               "intent": "покупка|уточнение|запись|гарантия",
#               "category": "Шины|Диски|АКБ|Масла/Жидкости|Услуги|Другое",
#               "category_detail": null,
#               "slots": {{{{
#                 "battery": {{{{
#                   "car_make": null,
#                   "car_model": null,
#                   "car_year": null,
#                   "capacity_ah": null,
#                   "cca_a": null,
#                   "polarity": null
#                 }}}},
#                 "tire": {{{{
#                   "size": {{{{ "width": null, "aspect": null, "rim": null }}}},
#                   "season": null,
#                   "load_index": null,
#                   "speed_index": null
#                 }}}},
#                 "solution": {{{{
#                   "stated_availability": "in_stock|backorder|unknown|n/a",
#                   "price_rub": null,
#                   "price_range_rub": [null, null],
#                   "offered_alt_same_category": "true|false|uncertain",
#                   "warranty_route": null,
#                   "sla": null,
#                   "bring_list": []
#                 }}}},
#                 "warranty_facts": {{{{
#                   "purchase_date": null,
#                   "receipt_present": null,
#                   "season_used": null,
#                   "usage_conditions": {{{{ "pressure": null, "speed": null, "storage": null }}}},
#                   "defect_text": null
#                 }}}}
#               }}}},
#               "next_step": {{{{ "set": false, "date": null, "time": null, "place": null, "owner": null, "action": null }}}},
#               "contacts": {{{{ "name": null, "phone": null, "email": null, "address": null, "car_plate": null, "order_id": null }}}},
#               "flags": {{{{
#                 "intent_identified": false,
#                 "required_params_collected": false,
#                 "solution_concrete": false,
#                 "next_step_set": false,
#                 "contacts_ok": false
#               }}}},
#               "objection": {{{{
#                 "detected": false,
#                 "list": [],
#                 "steps": {{{{ "acknowledge": false, "solution_or_argument": false, "check_close": false }}}},
#                 "handling_score": 0
#               }}}},
#               "wrap_up": {{{{
#                 "closing_ok": false,
#                 "repeated_items": false,
#                 "unit_price_announced": false,
#                 "total_price_announced": false,
#                 "fulfillment_method": null,
#                 "address_or_store": null,
#                 "working_hours": null,
#                 "delivery_terms": null,
#                 "storage_terms": null,
#                 "service_offer": {{{{ "type": null, "mentioned": false, "declined": false }}}},
#                 "for_legal_entities_docs": false,
#                 "consumables_notice": false,
#                 "follow_up_sla": null,
#                 "storage_contract_notice": false,
#                 "add_on_sale": {{{{ "offered": false, "accepted": false, "items": [], "total_price_rub": null, "evidence_quotes": [] }}}},
#                 "sale_outcome": "won|pending|lost|unknown",
#                 "sale_outcome_reason": null,
#                 "evidence_quotes": []
#               }}}},
#               "score": 0,
#               "red_flags": [],
#               "entities": [],
#               "catalog_items": [],
#               "actions": [],
#               "ambiguities": []
#             }}}} """
#         )
#     except Exception as e:
#         logger.info(e)
#         raise e
def resolve_llm_data(dialog_text: str) -> str:
    try:
        return saiga.ask(
            f""" Ты — детерминированный парсер входящих звонков (авто-ритейл).
            Верни ТОЛЬКО валидный JSON по схеме ниже. Если данных нет — null/unknown/uncertain. Без домыслов.
            
            Вход
            TODAY: {{YYYY-MM-DD | optional}}
            <<<DIALOGUE_START
            {dialog_text}
            DIALOGUE_END>>>
            
            Правила (кратко)
            1) intent: "покупка" | "уточнение" | "запись" | "гарантия".
               Если есть и покупка товара, и запись на работу по нему → intent="покупка".
            
            2) category: "Шины" | "Диски" | "АКБ" | "Масла/Жидкости" | "Услуги" | "Другое".
               category_detail (если уместно):
                 Услуги → "шиномонтаж"|"замена масла"|"диагностика"|"доставка"|"прочее";
                 Масла/Жидкости → "масло"|"антифриз/ОЖ"|"ATF"|"тормозная жидкость"|"омывайка".
            
            3) tire_rim: диаметр диска (из 225/60R17, 225 60 17, «225 на 60 р17») как число. Нет — null.
            
            4) availability: "in_stock" (в наличии) | "backorder" (под заказ) | "unknown" | "n/a".
               Ключи: «в наличии/есть»→in_stock; «под заказ/будет к …»→backorder; иначе unknown.
            
            5) season: "летняя"|"зимняя"|"всесезонная" ТОЛЬКО если:
               • прямое слово (летние/зимние/всесезонные/AS/«круглый год»), или
               • зимние маркеры: шипы/шипованная/липучка/3PMSF/снежинка/Ice/Nordic/Friction, или
               • летние маркеры: «на лето/летняя», или
               • «переобувка/сезонный шиномонтаж» + период: март–май→летняя; окт–дек→зимняя (можно от TODAY).
               Иначе null.
            
            6) objection (жалоба/возражение):
               detected=true, если есть недовольство/жалоба/брань/эскалация/недоверие/«не могу/не устраивает»/проблема со сроками, ценой, наличием, процессом.
               type: price|availability|time_sla|trust|warranty_claim|fitment|location_logistics|process.
               handling — оцени ответ оператора по шагам:
                 acknowledge (признал/извинился/«давайте разберёмся»),
                 solution (конкретное действие/альтернатива/эскалация),
                 check_close (проверка снятия).
               status: "resolved" | "partially_resolved" | "unresolved" | "ignored".
               score (0–3): 0=ignored; 1=acknowledge; 2=acknowledge+solution; 3=acknowledge+solution+check_close.
               Дай короткие цитаты: quote (клиент) и sales_quote (оператор).
            
            7) next_set (ДАЛЬНЕЙШИЕ ШАГИ):
               set=true, если есть подтверждённое действие (прямые формулировки).
               Примеры action: "зарезервировать"|"оформить заказ"|"самовывоз"|"доставка"|"записать на шиномонтаж"|
               "перезвонить"|"с вами свяжутся"|"оформить претензию"|"переоформить заказ"|"отменить заказ"|
               "приехать"|"забрать"|"оплатить".
               owner: "client" если действие делает клиент; "sales" если инициирует оператор.
               datetime/place можно частично (любые из них могут быть null). Нормализуй время "16.30"→"16:30".
            
            8) wrap_up: true, если оператор кратко резюмировал ≥2 из: состав/параметры/цена/сумма; способ/место получения (адрес); дата/срок (доставка/хранение/запись).
            
            9) add_on_sale (допродажа): фиксируй только явные доптовары/услуги, предложенные/принятые сверх основного запроса.
               offered=true при явном предложении (напр. «пакеты для шин нужны?» «доп. гарантия» «хранение»).
               accepted=true при явном согласии клиента («давайте пакеты/гарантию/хранение»).
               items[]: перечисли названия доптоваров/услуг (напр. "пакеты для шин","доп. гарантия","сезонное хранение","мойка","монтаж").
               Если суммы/кол-ва сказаны — можно не указывать (оставь null).
            
            Evidence-lock: любые НЕ-null (кроме wrap_up, objection.status/score) — только при прямых формулировках из диалога.
            
            Выходной JSON (строго эта структура)
            {{
              "intent": "покупка|уточнение|запись|гарантия",
              "category": "Шины|Диски|АКБ|Масла/Жидкости|Услуги|Другое",
              "category_detail": null,
              "tire_rim": null,
              "availability": "in_stock|backorder|unknown|n/a",
              "season": null,
              "objection": {{
                "detected": false,
                "type": null,
                "quote": null,
                "sales_quote": null,
                "handling": {{ "acknowledge": false, "solution": false, "check_close": false }},
                "status": "ignored|unresolved|partially_resolved|resolved",
                "score": 0
              }},
              "next_set": {{
                "set": false,
                "action": null,      // из списка выше
                "datetime": null,    // "YYYY-MM-DD HH:MM" | "YYYY-MM-DD" | "HH:MM" | null
                "place": null,
                "owner": null        // "client"|"sales"|null
              }},
              "wrap_up": false,
              "add_on_sale": {{
                "offered": false,
                "accepted": false,
                "items": [],         // напр. ["пакеты для шин","доп. гарантия"]
                "evidence_quotes": []// короткие цитаты (1–2), опционально
              }}
            }} """
        )
    except Exception as e:
        logger.info(e)
        raise e