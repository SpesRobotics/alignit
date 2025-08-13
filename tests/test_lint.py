import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str]):
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
    return result.returncode == 0


def test_black():
    result = run([sys.executable, "-m", "black", "--check", "--diff", "--verbose", "."])
    assert result, "Black formatting issues found"
