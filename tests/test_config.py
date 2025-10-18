import pytest

from src import config


@pytest.fixture(autouse=True)
def clear_cache():
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()


def test_settings_loads_from_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    monkeypatch.setenv("DEVICE", "cpu")
    settings = config.get_settings([])
    assert settings.openai_api_key == "test-key"
    assert settings.openai_model == "gpt-test"
    assert settings.device == "cpu"


def test_cpu_flag_overrides_device(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("DEVICE", raising=False)
    settings = config.get_settings(["--cpu"])
    assert settings.device == "cpu"
