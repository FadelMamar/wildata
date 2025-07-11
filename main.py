#!/usr/bin/env python3
"""
WildTrain Data Pipeline - Main entry point
"""

from wildtrain.cli import app
from wildtrain.logging_config import setup_logging

if __name__ == "__main__":
    # Setup basic logging for command line usage
    setup_logging(level="INFO")
    app()
