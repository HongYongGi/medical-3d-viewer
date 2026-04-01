from __future__ import annotations
import time
import streamlit as st


class ProgressTracker:
    """Streamlit progress bar with tqdm-style status text."""

    def __init__(self, label: str = "처리 중..."):
        self.label = label
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        self.start_time = time.time()

    def update(self, fraction: float, message: str = ""):
        fraction = max(0.0, min(1.0, fraction))
        self.progress_bar.progress(fraction)
        elapsed = time.time() - self.start_time
        if fraction > 0:
            estimated_total = elapsed / fraction
            remaining = estimated_total - elapsed
            elapsed_str = self._format_time(elapsed)
            remaining_str = self._format_time(remaining)
            pct = int(fraction * 100)
            bar_len = 20
            filled = int(bar_len * fraction)
            bar = '\u2588' * filled + '\u2591' * (bar_len - filled)
            self.status_text.markdown(
                f"`[{bar}] {pct}% | 경과: {elapsed_str} | "
                f"남은시간: ~{remaining_str} | {message}`"
            )
        else:
            self.status_text.text(message)

    def complete(self, message: str = "완료!"):
        self.progress_bar.progress(1.0)
        elapsed = self._format_time(time.time() - self.start_time)
        self.status_text.markdown(
            f"`[{'█' * 20}] 100% | 총 소요: {elapsed} | {message}`"
        )

    @staticmethod
    def _format_time(seconds: float) -> str:
        m, s = divmod(int(seconds), 60)
        return f"{m:02d}:{s:02d}"
