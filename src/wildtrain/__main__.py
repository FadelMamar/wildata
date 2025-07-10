from .cli import app
from .config import ROOT
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv(ROOT / ".env")
    app() 