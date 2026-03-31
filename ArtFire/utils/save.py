import json
from pathlib import Path
from typing import Any, Dict


def save_json(obj: Dict[str, Any], path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)
