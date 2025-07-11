import pytest
from typer.testing import CliRunner

from src.wildtrain.cli import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "wildtrain version" in result.output


def test_status():
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "WildTrain Pipeline Status" in result.output


def test_dataset_list():
    result = runner.invoke(app, ["dataset", "list"])
    assert result.exit_code == 0
    assert "No datasets found" in result.output or "Found" in result.output
