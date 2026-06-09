from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"


def load_env(path=ENV_FILE):
    if not path.exists():
        return {}

    settings = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        settings[key.strip()] = value.strip().strip('"').strip("'")

    return settings


_settings = load_env()

API_KEY = _settings.get("API_KEY")

if not API_KEY:
    raise RuntimeError("API_KEY is missing from the .env file")
