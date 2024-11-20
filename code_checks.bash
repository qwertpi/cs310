root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
find "$root" -type d -name "venv" -prune -o -type f -name "*.py" -exec python3 -m ruff format {} +
find "$root" -type d -name "venv" -prune -o -type f -name "*.py" -exec python3 -m ruff check {} +
find "$root" -type d -name "venv" -prune -o -type f -name "*.py" -exec python3 -m mypy {} +