#!/usr/bin/env python3
"""Test script to verify ConfigLoader path resolution works correctly."""

from lib.yaml_reader import ConfigLoader
from pathlib import Path

def test_config_loader():
    """Test various config file paths."""
    test_cases = [
        "../configs/config.yaml",
        "configs/config.yaml", 
        "post_processors/config/sales_patterns.yaml",
        "config/criteria_detector_config.yaml",
    ]
    
    print("Testing ConfigLoader path resolution:\n")
    
    for config_path in test_cases:
        try:
            config = ConfigLoader(config_path)
            resolved_path = config.config_path.relative_to(Path(__file__).parent)
            print(f"✅ '{config_path}'")
            print(f"   → Resolved to: {resolved_path}\n")
        except FileNotFoundError as e:
            print(f"❌ '{config_path}'")
            print(f"   → Error: {e}\n")
        except Exception as e:
            print(f"⚠️  '{config_path}'")
            print(f"   → Unexpected error: {e}\n")

if __name__ == "__main__":
    test_config_loader()

