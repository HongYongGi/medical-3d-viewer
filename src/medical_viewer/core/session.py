from __future__ import annotations

import uuid
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Session:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)
    input_path: Path | None = None
    result_paths: dict[str, Path] = field(default_factory=dict)

    def upload_dir(self, base: Path) -> Path:
        d = base / self.id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def result_dir(self, base: Path) -> Path:
        d = base / self.id
        d.mkdir(parents=True, exist_ok=True)
        return d
