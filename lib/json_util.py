import json
from typing import Union, Dict, List
import re

def text_to_json(
        text: str,
        indent: int = 2,
        ensure_ascii: bool = False,
        sort_keys: bool = False
) -> Union[Dict, List, str]:
    """
    Converts a text string to JSON format with multiple fallback strategies.

    Args:
        text: Input text to convert
        indent: Indentation for pretty-printing
        ensure_ascii: Escape non-ASCII characters
        sort_keys: Sort dictionary keys

    Returns:
        Parsed JSON (dict/list) if successful, or JSON-formatted string as fallback
    """
    # First try: Direct JSON parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Second try: Fix common issues and retry
    try:
        # Remove trailing commas
        fixed = text.replace(',}', '}').replace(',]', ']')
        # Wrap unquoted keys/values
        fixed = re.sub(r'(\w+):', r'"\1":', fixed)
        return json.loads(fixed)
    except (json.JSONDecodeError, TypeError):
        pass

    # Final fallback: Convert entire text to JSON string
    try:
        return json.dumps(
            {"content": text.strip()},
            indent=indent,
            ensure_ascii=ensure_ascii,
            sort_keys=sort_keys
        )
    except Exception:
        return json.dumps({"error": "Failed to convert text to JSON"})