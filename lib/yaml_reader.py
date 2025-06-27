import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional


class ConfigLoader:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load and parse YAML config file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split(".")
        value = self._config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def get_dict_list(self, key_path: str) -> Dict[str, List[str]]:
        value = self.get(key_path)

        if value is None:
            raise KeyError(f"Key path not found: {key_path}")

        if not isinstance(value, dict) or not all(isinstance(v, list) for v in value.values()):
            raise ValueError(
                f"Config at path '{key_path}' must be a dictionary with list values. "
                f"Got {type(value)} with sample value type: "
                f"{type(next(iter(value.values()))) if value else 'empty'}"
            )

        return value

    def get_config(self) -> Dict[str, Any]:
        """Get entire config dictionary"""
        return self._config