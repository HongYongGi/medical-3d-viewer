"""Session and data cleanup utilities."""
from __future__ import annotations

import shutil
import time
from pathlib import Path


def cleanup_old_sessions(
    data_dir: Path,
    max_age_hours: float = 24,
    subdirs: tuple[str, ...] = ("uploads", "results", "meshes"),
) -> dict[str, int]:
    """Remove session directories older than max_age_hours.

    Returns dict with counts of removed dirs per subdir.
    """
    cutoff = time.time() - max_age_hours * 3600
    removed: dict[str, int] = {}

    for subdir_name in subdirs:
        subdir = data_dir / subdir_name
        if not subdir.exists():
            continue
        count = 0
        for session_dir in subdir.iterdir():
            if not session_dir.is_dir():
                continue
            # Check modification time of directory
            try:
                mtime = session_dir.stat().st_mtime
                if mtime < cutoff:
                    shutil.rmtree(session_dir)
                    count += 1
            except OSError:
                continue
        removed[subdir_name] = count

    return removed


def get_data_usage(data_dir: Path) -> dict[str, float]:
    """Get disk usage in MB for each data subdirectory."""
    usage = {}
    if not data_dir.exists():
        return usage
    for subdir in data_dir.iterdir():
        if subdir.is_dir():
            total = sum(f.stat().st_size for f in subdir.rglob("*") if f.is_file())
            usage[subdir.name] = total / (1024 * 1024)
    return usage
