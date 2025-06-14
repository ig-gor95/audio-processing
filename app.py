from openai import OpenAI
import json


def get_objections_from_ollama(dialog: str, prompt: str) -> list[dict]:
    """
    Вызывает модель Saiga через Ollama и возвращает список возражений клиента.
    """
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )

    try:
        response = client.chat.completions.create(
            model="cyberlis/saiga-mistral:7b-lora-custom-q4_K",
            messages=[
                {"role": "system", "content": "Ты анализируешь диалоги и возвращаешь JSON."},
                {"role": "user", "content": f"{prompt}\n\nДиалог:\n{dialog}"}
            ],
            temperature=0.2
        )

        content = response.choices[0].message.content
        print("\n🔍 Ответ модели (сырое содержимое):\n", content)

        data = json.loads(content)
        return data.get("objections", [])

    except Exception as e:
        print(f"\n❌ Ошибка API: {str(e)}")
        return []



if __name__ == "__main__":
    custom_prompt = """
        Проанализируй диалог и верни возражения клиента в JSON формате:
        {
            "objections": [
                {"type": "тип_возражения", "text": "цитата"}
            ]
        }
        Типы возражений: цена, доверие, необходимость, ассортимент.
        Внимание: Верни только JSON без пояснений.
        """

    test_dialog = """
        Клиент: Этот ноутбук слишком дорогой
        Оператор: У нас есть рассрочка без %
        Клиент: А если он сломается через неделю?
        Оператор: Гарантия 2 года
        """

    objections = get_objections_from_ollama(test_dialog, custom_prompt)

    print("\n📋 Найденные возражения:")
    for obj in objections:
        print(f"- [{obj['type']}] {obj['text']}")
