from typing import List

from core.entity.audio_to_text_result import ProcessingResults, ObjectionProp
from saiga import SaigaClient


def resolve_objections(results: ProcessingResults) -> ProcessingResults:
    saiga = SaigaClient()

    res = saiga.ask_with_json_response(
        f"""
        Я передам тебе текст диалога, после слов ОСНОВНОЙ ТЕКСТ.
        Нужно понять было ли отработано возражение клиента, когда клиенту не понравились условия,
        а менеджер предоставил условие получше, то есть отработал возражение.
        Виды воражений клинета: это дорого, это неудобно, это не нравится.
        Вернуть результат нужно в формате списка json объектов.
        Пример json:
        [{{
            "objection_str": "",
            "manager_offer_str": "",
            "was_resolved": true"
        }}] 
        objection_str - строка с возражением клиента
        manager_offer_str - строка с отработкой возражения менеджером
        was_resolved - остался ли клиент доволен
        
        Возражений и отработок возражений может быть несколько
        выводи только заполненную структуру
        ОСНОВНОЙ ТЕКСТ:
        {results.to_string()}
        
        Обязательно В ответе выведи только json и больше ничего без форматирования и лишних символов, так как я буду парсить твой вывод
        строго соблюдай наименования полей в json
        """
    )
    if not isinstance(res, List):
        raise ValueError(f"Expected dictionary but got different format {res}")

    objection_props = []

    for i in res:
        try:
            objection_props.append(ObjectionProp(i["objection_str"], i["manager_offer_str"],  i["was_resolved"]))
        except Exception:
            raise ValueError(f"incorrect format {i}")

    results.objection_prop = objection_props

    return results
