import os, httpx
from openai import OpenAI
from json_util import text_to_json
from yaml_reader import ConfigLoader

config = ConfigLoader("../configs/config.yaml")

class SaigaClient:
    def __init__(self):
        base_url = (config.get("saiga.url", "http://127.0.0.1:11434/v1")
                    .replace("localhost", "127.0.0.1"))

        # Make this process proxy-proof and prefer loopback
        for var in ("HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy"):
            os.environ.pop(var, None)
        os.environ["NO_PROXY"] = ",".join(filter(None, [
            os.environ.get("NO_PROXY",""),
            "127.0.0.1,localhost,::1",
        ]))

        self.client = OpenAI(
            base_url=base_url,
            api_key=config.get("saiga.api-key", "ollama"),
            http_client=httpx.Client(
                proxies=None,        # <- ignore proxies, even if set by IDE/OS
                trust_env=False,     # <- don't read *_PROXY from environment
                timeout=httpx.Timeout(60, connect=10),
            ),
        )
        self.model = config.get("saiga.model-name", "ilyagusev/saiga_llama3")

    def ask(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content

    def ask_with_json_response(self, prompt: str):
        return text_to_json(self.ask(prompt))
