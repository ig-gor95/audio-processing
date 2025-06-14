from typing import Dict, List

from saiga import SaigaClient

# Пример использования
if __name__ == "__main__":
    saiga = SaigaClient()

    dialog = """
         Client: Здравствуйте! Я бы хотел приобрести колеса у вас!
         Manager: Добрый день! Меня зовут Евгений, какие колеса вас интересуют?
         Client: Для мазды-шестерки
         Manager: Стоимость будет - 50 тысяч рублей
         Client: Это будет дорого для меня, такое меня не удовлетворит
         Manager: Тогда мы можем предложить вам рассрочку
         Client: Это звучит лучше
         """

    res = saiga.ask_with_json_response(
        f"""
        Я передам тебе текст диалога, после слов ОСНОВНОЙ ТЕКСТ.
        Нужно понять было ли отработано возражение клиента, когда клиенту не понравились условия,
        а менеджер предоставил условие получше, то есть отработал возражение.
        Вернуть результат нужно в формате json.
        Тип json:
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
        {dialog}
        
        В ответе выведи только json и больше ничего без форматирования и лишних символов, так как я буду парсить твой вывод
        """
    )
    if not isinstance(res, List):
        raise ValueError("Expected dictionary but got different format")

        # Work with dictionary data
    for i in res:
        print(i)