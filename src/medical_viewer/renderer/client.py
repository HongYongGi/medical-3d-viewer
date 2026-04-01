from __future__ import annotations
import httpx
from pathlib import Path


class RendererClient:
    """HTTP client for communicating with Go 3D renderer service."""

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip("/")

    def health_check(self) -> bool:
        try:
            resp = httpx.get(f"{self.base_url}/health", timeout=2.0)
            return resp.status_code == 200
        except Exception:
            return False

    def load_volume(self, session_id: str, vol_path: str | Path, seg_path: str | Path | None = None) -> dict:
        payload = {
            "session_id": session_id,
            "vol_path": str(vol_path),
            "seg_path": str(seg_path) if seg_path else "",
        }
        resp = httpx.post(f"{self.base_url}/api/v1/volume/load", json=payload, timeout=10.0)
        resp.raise_for_status()
        return resp.json()

    def generate_meshes(self, session_id: str) -> dict:
        resp = httpx.post(f"{self.base_url}/api/v1/mesh/generate/{session_id}", timeout=120.0)
        resp.raise_for_status()
        return resp.json()

    def get_session_info(self, session_id: str) -> dict | None:
        try:
            resp = httpx.get(f"{self.base_url}/api/v1/session/{session_id}", timeout=5.0)
            return resp.json() if resp.status_code == 200 else None
        except Exception:
            return None

    def get_viewer_url(self, session_id: str) -> str:
        return f"{self.base_url}/viewer/{session_id}"
