import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional


class ConfigLoader:
    def __init__(self, config_path: str):
        """
        Initialize ConfigLoader with automatic path resolution.
        
        Args:
            config_path: Path to config file. Can be:
                - Absolute path
                - Relative to project root (e.g., "configs/config.yaml")
                - Relative path with ../ (will be resolved from project root)
        """
        self.config_path = self._resolve_config_path(config_path)
        self._config = self._load_config()
    
    def _resolve_config_path(self, config_path: str) -> Path:
        """Resolve config path relative to project root."""
        path = Path(config_path)
        
        # If absolute path and exists, use it
        if path.is_absolute() and path.exists():
            return path
        
        # Find project root (where this file's parent/parent is)
        # lib/yaml_reader.py -> lib -> project_root
        project_root = Path(__file__).parent.parent
        
        # Remove leading ../ from path if present and resolve from project root
        path_str = str(path)
        while path_str.startswith('../'):
            path_str = path_str[3:]
        
        # Try multiple possible locations
        possible_paths = [
            project_root / path_str,                           # Direct from root
            project_root / config_path,                        # Original path
            project_root / "core" / path_str,                  # In core/
            project_root / "configs" / Path(path_str).name,    # In configs/
            project_root / "config" / Path(path_str).name,     # In config/
        ]
        
        # If path contains subdirectories (e.g., post_processors/config/file.yaml)
        # also try searching for it in common locations
        if '/' in path_str:
            parts = Path(path_str).parts
            filename = parts[-1]
            
            # Search common config directories
            search_dirs = [
                project_root,
                project_root / "core",
                project_root / "configs",
                project_root / "config",
            ]
            
            for search_dir in search_dirs:
                # Use glob to find the file anywhere under this directory
                matches = list(search_dir.rglob(filename))
                if matches:
                    # Prefer exact path match if multiple found
                    for match in matches:
                        if path_str in str(match.relative_to(project_root)):
                            possible_paths.insert(0, match)
                            break
                    else:
                        possible_paths.insert(0, matches[0])
        
        for p in possible_paths:
            if p.exists():
                return p
        
        # If not found, return the first attempt (will error in _load_config)
        return possible_paths[0]

    def _load_config(self) -> Dict[str, Any]:
        """Load and parse YAML config file"""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {self.config_path}\n"
                f"Tried to resolve from project root. "
                f"Make sure the file exists in configs/ or config/ directory."
            )
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
    
    def get_all(self) -> Dict[str, Any]:
        """Alias for get_config() - get entire config dictionary"""
        return self._config