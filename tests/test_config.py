import os

import pytest

from app.config import ConfigError, load_dotenv, load_settings


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("AGENT_MAX_STEPS", raising=False)
    monkeypatch.delenv("GEMINI_TEMPERATURE", raising=False)
    monkeypatch.delenv("GEMINI_THINKING", raising=False)


def test_missing_file_and_key_raises(tmp_path, clean_env):
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        load_settings(tmp_path / "absent.env")


def test_quoted_assignment_and_defaults(tmp_path, clean_env):
    (tmp_path / ".env").write_text('export GEMINI_API_KEY="test-key-value"\n# comment\n\n', encoding="utf-8")
    settings = load_settings(tmp_path / ".env")
    assert settings.api_key == "test-key-value"
    assert settings.model == "gemini-3.5-flash-lite"
    assert settings.max_steps == 12
    assert settings.temperature == 0.2
    assert settings.thinking == "low"


def test_existing_environment_is_not_overridden(tmp_path, clean_env, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    (tmp_path / ".env").write_text("GEMINI_API_KEY=from-file\nGEMINI_MODEL=gemini-3.5-flash\n", encoding="utf-8")
    load_dotenv(tmp_path / ".env")
    assert os.environ["GEMINI_API_KEY"] == "from-env"
    assert os.environ["GEMINI_MODEL"] == "gemini-3.5-flash"


def test_bare_token_file_is_the_api_key(tmp_path, clean_env):
    (tmp_path / ".env").write_text("AIzaSyDUMMYKEY0123456789abcdef\n", encoding="utf-8")
    settings = load_settings(tmp_path / ".env")
    assert settings.api_key == "AIzaSyDUMMYKEY0123456789abcdef"


def test_invalid_thinking_level(tmp_path, clean_env, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_THINKING", "ultra")
    with pytest.raises(ConfigError, match="GEMINI_THINKING"):
        load_settings(tmp_path / "absent.env")


def test_invalid_max_steps(tmp_path, clean_env, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("AGENT_MAX_STEPS", "zero")
    with pytest.raises(ConfigError, match="AGENT_MAX_STEPS"):
        load_settings(tmp_path / "absent.env")
