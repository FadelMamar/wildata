from dotenv import load_dotenv

from .cli import app
from .config import ROOT

if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    app()
