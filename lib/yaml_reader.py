import yaml
from pathlib import Path

class ConfigLoader:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self._config = self._load_config()

    def _load_config(self):
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get(self, key_path: str, default=None):
        keys = key_path.split(".")
        value = self._config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def get_config(self):
        return self._config
