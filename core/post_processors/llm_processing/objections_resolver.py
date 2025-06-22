from typing import List

from core.entity.audio_to_text_result import ProcessingResults, ObjectionProp
from saiga import SaigaClient


def resolve_objections(results: ProcessingResults) -> ProcessingResults:
    saiga = SaigaClient()

    res = saiga.ask_with_json_response(
        f"""
        По диалогу с пронумерованными фразами, мне нужно вывести json список,
        в котором o: номер строки, где CLIENT высказал возражение по цене или времени
        r: номер строки, где SALES предложил решение
        w: true или false - остался ли клиент удовлетворен решением
        Если таких ситуаций не было, вывести пустой json список
        кроме списка в ответе ничего не выводи вообще, так как ответ будет парсится
        выводом должен обязательно json список, даже если в нем только 1 элемент
        никаких слов с разъяснениями не нужно
        """
    )
    if not isinstance(res, List):
        raise ValueError(f"Expected dictionary but got different format {res}")

    objection_props = []

    for i in res:
        try:
            objection_props.append(ObjectionProp(i["o"], i["r"], i["w"]))
        except Exception:
            raise ValueError(f"incorrect format {i}")

    results.objection_prop = objection_props

    return results
