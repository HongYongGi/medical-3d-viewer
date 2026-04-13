import subprocess
import sys
from pathlib import Path


def main():
    app_path = str(Path(__file__).parent / "app.py")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        app_path,
        "--server.port=8501",
        "--server.headless=true",
    ])


if __name__ == "__main__":
    main()
