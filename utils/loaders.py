import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List
import tomllib
import json

# Load config file
def load_config(config_file: Path) -> dict:
    """
    Load a TOML configuration file and return its contents as a dictionary.
    """
    with config_file.open("rb") as f:
        config = tomllib.load(f)
    return config

# Load channels file based on extension
def load_channels_txt(channels_file: Path) -> List[str]:
    """
    Load channel names from a TXT file.
    - Strips whitespace
    - Ignores blank lines and lines starting with '#'
    - De-duplicates while preserving order
    """
    seen = set()
    out: List[str] = []
    with channels_file.open("r", encoding="utf-8") as f:
        for line in f:
            # strip comments (allow inline '# ...')
            raw = line.strip()
            if not raw:
                continue
            # If there's an inline comment, cut it
            if "#" in raw:
                raw = raw.split("#", 1)[0].strip()
            if not raw:
                continue
            if raw not in seen:
                seen.add(raw)
                out.append(raw)
    if not out:
        raise ValueError(f"No channels found in {channels_file}")
    return out

def load_channels_csv(channels_file: Path) -> List[str]:
    """
    Load channel names from a CSV file.
    - Expects a single column of channel names
    - Strips whitespace
    - Ignores blank lines and lines starting with '#'
    - De-duplicates while preserving order
    """
    seen = set()
    out: List[str] = []
    with channels_file.open("r", encoding="utf-8") as f:
        for line in f:
            # strip comments (allow inline '# ...')
            raw = line.strip()
            if not raw:
                continue
            # If there's an inline comment, cut it
            if "#" in raw:
                raw = raw.split("#", 1)[0].strip()
            if not raw:
                continue
            if raw not in seen:
                seen.add(raw)
                out.append(raw)
    if not out:
        raise ValueError(f"No channels found in {channels_file}")
    return out

def load_channels_json(channels_file: Path) -> List[str]:
    """
    Load channel names from a JSON file.
    Expects a JSON array of channel names.
    - Strips whitespace
    - Ignores blank entries
    - De-duplicates while preserving order
    """
    seen = set()
    out: List[str] = []
    with channels_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array in {channels_file}")
        for item in data:
            if not isinstance(item, str):
                continue
            raw = item.strip()
            if not raw:
                continue
            if raw not in seen:
                seen.add(raw)
                out.append(raw)
    if not out:
        raise ValueError(f"No channels found in {channels_file}")
    return out

def load_channels(channels_file: Path) -> List[str]:
    """
    Load channel names from a file.
    Supports TXT, CSV, and JSON formats based on file extension.
    """
    if not channels_file.exists():
        raise FileNotFoundError(f"Channels file not found: {channels_file}")
    
    if channels_file.suffix.lower() == ".txt":
        return load_channels_txt(channels_file)
    elif channels_file.suffix.lower() == ".csv":
        return load_channels_csv(channels_file)
    elif channels_file.suffix.lower() == ".json":
        return load_channels_json(channels_file)
    else:
        raise ValueError(f"Unsupported channels file format: {channels_file.suffix}")

# Write utility to save state as JSON
def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    """
    Write JSON to 'path' atomically: write to a temp file in the same directory then rename.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, path)  # atomic on POSIX
    except Exception:
        # Best effort to cleanup on failure
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise