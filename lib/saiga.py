import os, time, httpx
from openai import OpenAI, APITimeoutError, APIConnectionError, APIStatusError  # исключения из нового SDK
# RateLimitError можно добавить при необходимости

from json_util import text_to_json
from yaml_reader import ConfigLoader

config = ConfigLoader("../configs/config.yaml")

class SaigaClient:
    def __init__(self):
        base_url = (config.get("saiga.url", "http://127.0.0.1:11434/v1")
                    .replace("localhost", "127.0.0.1"))

        # Гарантированно игнорируем прокси
        for var in ("HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy","all_proxy"):
            os.environ.pop(var, None)
        os.environ["NO_PROXY"] = ",".join(filter(None, [
            os.environ.get("NO_PROXY",""),
            "127.0.0.1,localhost,::1",
        ]))

        # Транспорт с ретраями (httpx >= 0.26)
        transport = httpx.HTTPTransport(retries=3)

        self.client = OpenAI(
            base_url=base_url,
            api_key=config.get("saiga.api-key", "ollama"),
            http_client=httpx.Client(
                transport=transport,   # ← ретраи на уровне транспорта
                proxy=None,
                trust_env=False,
                timeout=httpx.Timeout(
                    connect=10.0,       # соединение поднимается быстро
                    read=300.0,         # ждём первый токен/ответ дольше
                    write=120.0,
                    pool=300.0,
                ),
                limits=httpx.Limits(
                    max_connections=20,
                    max_keepalive_connections=10,
                    keepalive_expiry=60.0,
                ),
            ),
        )
        self.model = config.get("saiga.model-name", "ilyagusev/saiga_llama3")

    def ask(self, prompt: str, max_retries: int = 3, backoff: float = 1.8) -> str:
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                )
                return resp.choices[0].message.content
            except (APITimeoutError, httpx.ReadTimeout, httpx.ConnectTimeout) as e:
                last_err = e
            except (APIConnectionError, APIStatusError, httpx.HTTPError) as e:
                last_err = e

            # Повтор с прогревом и бэкоффом
            if attempt < max_retries:
                try:
                    # warm-up: дёрнем список моделей (часто прогружает движок)
                    _ = self.client.models.list()
                except Exception:
                    pass
                time.sleep(backoff ** (attempt - 1))
                continue
            raise last_err  # исчерпали повторы

    def ask_with_json_response(self, prompt: str):
        return text_to_json(self.ask(prompt))
