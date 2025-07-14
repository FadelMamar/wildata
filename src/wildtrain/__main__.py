from dotenv import load_dotenv

from .cli import app
from .config import ROOT
from .logging_config import setup_logging

if __name__ == "__main__":
    load_dotenv(ROOT / ".env", override=True)
    # Setup basic logging for module execution
    setup_logging(level="INFO")
    app()
