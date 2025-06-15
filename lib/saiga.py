from openai import OpenAI

from json_util import text_to_json
from yaml_reader import ConfigLoader

config = ConfigLoader("../configs/config.yaml")

class SaigaClient:

    def __init__(self):
        self.client = OpenAI(
            base_url=config.get("saiga.url"),
            api_key=config.get("saiga.api-key")
        )
        self.model = config.get("saiga.model-name")

    def ask(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": f"{prompt}"}]
        )
        return response.choices[0].message.content

    def ask_with_json_response(self, prompt):
        return text_to_json(self.ask(prompt))
