"""Runtime configuration. The API key is read from the environment or ``.env`` and never logged."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from google.genai import types

DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_MAX_STEPS = 12
DEFAULT_TEMPERATURE = 0.2
DEFAULT_THINKING = "low"


class ConfigError(RuntimeError):
    """Missing or invalid local configuration."""


@dataclass(frozen=True)
class Settings:
    api_key: str
    model: str = DEFAULT_MODEL
    max_steps: int = DEFAULT_MAX_STEPS
    temperature: float = DEFAULT_TEMPERATURE
    thinking: str = DEFAULT_THINKING
    summary_model: str = ""

    def thinking_level(self) -> types.ThinkingLevel:
        key = self.thinking.strip().upper()
        try:
            return types.ThinkingLevel[key]
        except KeyError as exc:
            valid = ["minimal", "low", "medium", "high"]
            raise ConfigError(f"GEMINI_THINKING must be one of {valid} (got {self.thinking!r}).") from exc


def load_dotenv(path: Path | None = None) -> None:
    """Load ``.env`` into the environment without overriding variables that are already set.

    Accepts ``KEY=value``, optional ``export`` and quotes. A file whose only
    content is a single token is treated as ``GEMINI_API_KEY``.
    """
    path = Path(path) if path is not None else Path(".env")
    if not path.is_file():
        return
    assignments: list[tuple[str, str]] = []
    bare: list[str] = []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            bare.append(line)
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        assignments.append((key.strip(), value))
    for key, value in assignments:
        if key:
            os.environ.setdefault(key, value)
    if (
        len(bare) == 1
        and " " not in bare[0]
        and len(bare[0]) >= 20
        and "GEMINI_API_KEY" not in os.environ
        and "GOOGLE_API_KEY" not in os.environ
    ):
        os.environ["GEMINI_API_KEY"] = bare[0]


def load_settings(env_file: Path | None = None) -> Settings:
    """Build settings from the environment, loading ``env_file`` (default ``.env``) first."""
    load_dotenv(env_file)
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ConfigError(
            "GEMINI_API_KEY is not set. Add a line to .env: GEMINI_API_KEY=your_key "
            "(a .env file containing only the key is also accepted)."
        )
    max_steps_raw = os.environ.get("AGENT_MAX_STEPS", str(DEFAULT_MAX_STEPS))
    temperature_raw = os.environ.get("GEMINI_TEMPERATURE", str(DEFAULT_TEMPERATURE))
    try:
        max_steps = int(max_steps_raw)
        temperature = float(temperature_raw)
    except ValueError as exc:
        raise ConfigError(f"AGENT_MAX_STEPS and GEMINI_TEMPERATURE must be numbers (got {max_steps_raw!r}, {temperature_raw!r}).") from exc
    if max_steps < 1:
        raise ConfigError("AGENT_MAX_STEPS must be at least 1.")
    settings = Settings(
        api_key=api_key,
        model=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL),
        max_steps=max_steps,
        temperature=temperature,
        thinking=os.environ.get("GEMINI_THINKING", DEFAULT_THINKING),
        summary_model=os.environ.get("GEMINI_SUMMARY_MODEL", ""),
    )
    settings.thinking_level()  # validate now, not on the first request
    return settings
