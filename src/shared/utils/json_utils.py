import json
from typing import Union, Dict, List
import re

def text_to_json(
        text: str,
        indent: int = 2,
        ensure_ascii: bool = False,
        sort_keys: bool = False
) -> Union[Dict, List, str]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    try:
        fixed = text.replace(',}', '}').replace(',]', ']').replace("```json", "").replace("```", "")
        fixed = fixed.strip()
        fixed = re.sub(r'(\w+):', r'"\1":', fixed)
        return json.loads(fixed)
    except (json.JSONDecodeError, TypeError):
        pass

    try:
        return json.dumps(
            {"content": text.strip()},
            indent=indent,
            ensure_ascii=ensure_ascii,
            sort_keys=sort_keys
        )
    except Exception:
        return json.dumps({"error": "Failed to convert text to JSON"})