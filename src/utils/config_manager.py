import json
import os
from typing import Any, Union

class ConfigManager:
    def __init__(self, conf_path: str = "conf.json") -> None:
        self.conf_path = conf_path
        self.conf = self._load_conf()

    def _load_conf(self) -> dict:

        if not os.path.exists(self.conf_path):
            raise FileNotFoundError(f"The path to the config file is not existing: {self.conf_path}")

        try:
            with open(self.conf_path, 'r', encoding='utf-8') as file:
                return json.load(file)

        except json.JSONDecodeError as e:
            raise ValueError(f"JSON parsing Error in '{self.conf_path}': {e}")

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        value = self.conf

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        keys = key.split(".")
        conf = self.conf

        for k in keys[:-1]:
            if k not in conf or not isinstance(conf[k], dict):
                conf[k] = {}
            conf = conf[k]

        conf[keys[-1]] = value

    def save(self, conf_path: Union[str, None] = None):
        conf_path = conf_path or self.conf_path

        try:
            with open(conf_path, "w", encoding='utf-8') as file:
                json.dump(self.conf, file, indent=4, ensure_ascii=False)

        except IOError as e:
            raise IOError(f"Error during config file editing '{self.conf_path}': {e}")
